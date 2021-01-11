#include <ros/ros.h>

#include <parameters.h>
#include <MapRingBuffer.h>
#include <math_utils.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <tic_toc.h>

#include <StateEstimator.hpp>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <sensor_utils.hpp>

#include "sophus/so3.h"
#include "sophus/se3.h"

class stateTransitionTest{
  public:
    stateTransitionTest(ros::NodeHandle nh, ros::NodeHandle pnh): nh_(nh), pnh_(pnh), cnt_(0) {}
    ~stateTransitionTest() {}

    void run();
    void calcNumericalDerivative(Eigen::MatrixXd& error_state_dot_numerical, const Eigen::MatrixXd& cur_error_state);
    void imuCallback(const sensor_msgs::Imu::ConstPtr& imuIn);

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Subscriber subImu_;
    std::queue<Imu> imuBuf_;
    StatePredictor* filter_;
    GlobalState nominalState_;
    GlobalState trueState_;
    GlobalState state_tmp;
    V3D acc_raw_;
    V3D gyr_raw_;
    V3D ba_init_;
    V3D bw_init_;
    int cnt_;
    Imu imu_last_;
    Imu imu_cur_;
    double last_imu_time_;
    Eigen::MatrixXd last_error_state_;
    Eigen::MatrixXd last_error_state_dot_;
};

void stateTransitionTest::run() {
  ba_init_ = INIT_BA;
  bw_init_ = INIT_BW;
  last_error_state_ = Eigen::MatrixXd::Zero(GlobalState::DIM_OF_STATE_, 1);
  last_error_state_dot_ = Eigen::MatrixXd::Zero(GlobalState::DIM_OF_STATE_, 1);
  filter_ = new StatePredictor();
  subImu_ = pnh_.subscribe<sensor_msgs::Imu>(IMU_TOPIC, 100, &stateTransitionTest::imuCallback, this);
}

void stateTransitionTest::imuCallback(const sensor_msgs::ImuConstPtr& imuMsg) {
  acc_raw_ << imuMsg->linear_acceleration.x, imuMsg->linear_acceleration.y,
      imuMsg->linear_acceleration.z;
  gyr_raw_ << imuMsg->angular_velocity.x, imuMsg->angular_velocity.y,
      imuMsg->angular_velocity.z;
  imu_cur_.time = imuMsg->header.stamp.toSec();
  imu_cur_.acc = acc_raw_;
  imu_cur_.gyr = gyr_raw_;

  if (cnt_ == 0) {
    // Initialize first frame
    last_imu_time_ = imu_cur_.time;
    imu_last_ = imu_cur_;
    filter_->initialization(imu_last_.time, V3D(0, 0, 0), V3D(0, 0, 0),
                            ba_init_, bw_init_, imu_last_.acc,
                            imu_last_.gyr);
    state_tmp = filter_->state_;
    cnt_++;
  }
  else if (cnt_ <= 10) {   
    double dt = imu_cur_.time - last_imu_time_;
    filter_->predict(dt, imu_cur_.acc, imu_cur_.gyr, true);
    nominalState_ = filter_->state_;

    // Create noise matrix n_t^a, n_t^w, n_t^{ba}, n_t^{bw}
    double sqrt_cov[4] = {ACC_N * ug, GYR_N * dph, ACC_W * ugpsHz, GYR_W * dpsh};
    Eigen::MatrixXd error_noise = Eigen::MatrixXd::Zero(GlobalState::DIM_OF_NOISE_, 1);
    V3D noise_tmp;
    cv::RNG rng( cv::getTickCount() );
    cv::Mat tmp = cv::Mat::zeros(3, 1, CV_64FC1);
    for (int i = 0; i < 4; i++) {
      rng.fill(tmp, cv::RNG::NORMAL, 0., sqrt_cov[i]);
      cv::cv2eigen(tmp, noise_tmp);
      error_noise.middleRows(3 * i, 3) = noise_tmp;
    }

    // Add noise to true state and calculate nominal state
    V3D gra_noise;
    rng.fill(tmp, cv::RNG::NORMAL, 0., 0.1);
    cv::cv2eigen(tmp, gra_noise);
    V3D un_acc_0 = state_tmp.qbn_ * (imu_last_.acc - state_tmp.ba_ - error_noise.middleRows(6, 3)*dt - error_noise.middleRows(0, 3)) 
                   + state_tmp.gn_ + gra_noise;
    V3D un_gyr = 0.5 * (imu_last_.gyr + imu_cur_.gyr) - state_tmp.bw_ - error_noise.middleRows(9, 3)*dt - error_noise.middleRows(3, 3);
    Q4D dq = axis2Quat(un_gyr * dt);
    state_tmp.qbn_ = (state_tmp.qbn_ * dq).normalized();
    V3D un_acc_1 = state_tmp.qbn_ * (imu_cur_.acc - state_tmp.ba_ - error_noise.middleRows(6, 3)*dt - error_noise.middleRows(0, 3)) 
                   + state_tmp.gn_ + gra_noise;
    V3D un_acc = 0.5 * (un_acc_0 + un_acc_1);
    state_tmp.rn_ = state_tmp.rn_ + dt * state_tmp.vn_ + 0.5 * dt * dt * un_acc;
    state_tmp.vn_ = state_tmp.vn_ + dt * un_acc;
    trueState_ = state_tmp;
    trueState_.ba_ += error_noise.middleRows(6, 3)*dt;
    trueState_.bw_ += error_noise.middleRows(9, 3)*dt;

    // Calculate the derivative of the error state by analytical method
    Eigen::MatrixXd error_state = Eigen::MatrixXd::Zero(GlobalState::DIM_OF_STATE_, 1);
    Eigen::MatrixXd error_state_dot = Eigen::MatrixXd::Zero(GlobalState::DIM_OF_STATE_, 1);
    Sophus::SO3 delta_R(nominalState_.qbn_ * trueState_.qbn_.inverse());
    error_state.middleRows(0, 3) = delta_R.log();
    error_state.middleRows(3, 3) = nominalState_.vn_ - nominalState_.qbn_ * trueState_.qbn_.inverse() * trueState_.vn_;
    error_state.middleRows(6, 3) = nominalState_.rn_ - nominalState_.qbn_ * trueState_.qbn_.inverse() * trueState_.rn_;
    error_state.middleRows(9, 3) = nominalState_.bw_ - trueState_.bw_;
    error_state.middleRows(12, 3) = nominalState_.ba_ - trueState_.ba_;
    error_state.middleRows(15, 3) = nominalState_.gn_ - trueState_.gn_;
    error_state_dot = filter_->F_inekf * error_state + filter_->G_inkef * error_noise;
    
    // Calculate the derivative of the error state by numerical method
    //Eigen::MatrixXd error_state_dot_numerical = Eigen::MatrixXd::Zero(GlobalState::DIM_OF_STATE_, 1);
    //calcNumericalDerivative(error_state_dot_numerical, error_state);

    // Calculate the increment of the error state and compare with next frame
    if (cnt_ > 1) {
      std::cout << "Current error state is: " << std::endl;
      std::cout << error_state.transpose() << std::endl;
      Eigen::MatrixXd incre_error_state = Eigen::MatrixXd::Zero(GlobalState::DIM_OF_STATE_, 1);
      V3D epsilon_R_last(last_error_state_.middleRows(6, 3));
      V3D epsilon_R_incre(last_error_state_dot_.middleRows(6, 3));
      incre_error_state.middleRows(6, 3) = (Sophus::SO3::exp(epsilon_R_last) * Sophus::SO3::exp(epsilon_R_incre*dt)).log();
      incre_error_state.middleRows(0, 6) = last_error_state_.middleRows(0, 6) + dt * last_error_state_dot_.middleRows(0, 6);
      incre_error_state.middleRows(9, 9) = last_error_state_.middleRows(9, 9) + dt * last_error_state_dot_.middleRows(9, 9);
      std::cout << "Current error state with increment is: " << std::endl;
      std::cout << incre_error_state.transpose() << std::endl;
      std::cout << std::endl << std::endl;
    }

    last_error_state_ = error_state;
    last_error_state_dot_ = error_state_dot;
    state_tmp = trueState_;
    last_imu_time_ = imu_cur_.time;
    imu_last_ = imu_cur_;
    cnt_++;
  }
}

void stateTransitionTest::calcNumericalDerivative(Eigen::MatrixXd& error_state_dot_numerical, const Eigen::MatrixXd& cur_error_state) {
  // This is a complicated way and no use
  /* const double eps = 1e-6;
  Sophus::SO3 R_hat(nominalState_.qbn_), R(trueState_.qbn_);
  V3D v_hat_dot, v_dot, p_hat_dot, p_dot;
  M3D eta_R_dot, eta_R_dot_left, eta_R_dot_right;
  eta_R_dot_left.setZero();
  eta_R_dot_right.setZero();

  for (int k = 0; k < 3; k++) {
    V3D delta = Eigen::Vector3d(k == 0, k == 1, k == 2) * eps;
    V3D tmp_vec = ((R_hat * Sophus::SO3::exp(delta) * R.inverse()) * (R_hat * R.inverse()).inverse()).log() / eps;
    eta_R_dot_left += Sophus::SO3::hat(tmp_vec);
    tmp_vec = ((R_hat * (R * Sophus::SO3::exp(delta)).inverse()) * (R_hat * R.inverse()).inverse()).log() / eps;
    eta_R_dot_right += Sophus::SO3::hat(tmp_vec);
  }

  eta_R_dot = eta_R_dot_left + eta_R_dot_right;
  v_hat_dot = nominalState_.qbn_ * (imu_cur_.acc - nominalState_.ba_) + nominalState_.gn_;
  v_dot = trueState_.qbn_ * (imu_cur_.acc - trueState_.ba_ - error_noise.middleRows(0, 3)) + trueState_.gn_;
  p_hat_dot = nominalState_.vn_;
  p_dot = trueState_.vn_;

  error_state_dot_numerical.middleRows(0, 3) = p_hat_dot - eta_R_dot * trueState_.rn_ - R_hat * R.inverse() * p_dot;
  error_state_dot_numerical.middleRows(3, 3) = v_hat_dot - eta_R_dot * trueState_.vn_ - R_hat * R.inverse() * v_dot;
  error_state_dot_numerical.middleRows(6, 3) = Sophus::SO3(eta_R_dot).log();
  error_state_dot_numerical.middleRows(9, 3) = -error_noise.middleRows(6, 3);
  error_state_dot_numerical.middleRows(12, 3) = -error_noise.middleRows(9, 3); */
  V3D epsilon_R_cur(cur_error_state.middleRows(6, 3));
  V3D epsilon_R_last(last_error_state_.middleRows(6, 3));
  double dt = imu_cur_.time - imu_last_.time;
  error_state_dot_numerical.middleRows(6, 3) = (Sophus::SO3::exp(epsilon_R_cur) * Sophus::SO3::exp(epsilon_R_last).inverse()).log() / dt;
  error_state_dot_numerical.middleRows(0, 6) = (cur_error_state.middleRows(0, 6) - last_error_state_.middleRows(0, 6)) / dt;
  error_state_dot_numerical.middleRows(9, 9) = (cur_error_state.middleRows(9, 9) - last_error_state_.middleRows(9, 9)) / dt;
}


int main(int argc, char** argv) {
  ros::init(argc, argv, "state_transition_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  parameter::readParameters(pnh);

  stateTransitionTest stateTrans(nh, pnh);
  stateTrans.run();

  ros::spin();
  return 0;
}