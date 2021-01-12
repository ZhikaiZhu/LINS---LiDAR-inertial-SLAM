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

#include <unsupported/Eigen/MatrixFunctions>

inline M3D skewSymmetric(const V3D& w) {
  M3D w_hat;
  w_hat << 0., -w(2),  w(1),
         w(2),     0, -w(0), 
        -w(1),  w(0),     0;    
  return w_hat;
}

inline V3D vee(const Matrix3d& w_hat) {
  const double SMALL_EPS = 1e-10;
  assert(fabs(w_hat(2, 1) + w_hat(1, 2)) < SMALL_EPS);
  assert(fabs(w_hat(0, 2) + w_hat(2, 0)) < SMALL_EPS);
  assert(fabs(w_hat(1, 0) + w_hat(0, 1)) < SMALL_EPS);
  return Vector3d(w_hat(2, 1), w_hat(0, 2), w_hat(1, 0));
}

inline Eigen::VectorXd expSE_3(const V3D& xi_omega, const Eigen::VectorXd& xi_x) {
  const double theta = std::sqrt(xi_omega.squaredNorm());
  M3D Omega_1 = skewSymmetric(xi_omega);
  M3D Omega_2 = Omega_1 * Omega_1;
  if (theta < 1e-6) {
    return xi_x + 1 / 2 * Omega_1 * xi_x + 1 / 6 * Omega_2 * xi_x;
  }
  else {
    const double A = std::sin(theta) / theta;
    const double B = (1 - std::cos(theta)) / (theta * theta);
    const double C = (1 - A) / (theta * theta);
    const M3D  V = M3D::Identity() + B * Omega_1 + C * Omega_2;
    return V * xi_x;
  }
}

inline M3D expSO_3(const V3D& xi_omega) {
  const double theta = std::sqrt(xi_omega.squaredNorm());
  M3D Omega_1 = skewSymmetric(xi_omega);
  M3D Omega_2 = Omega_1 * Omega_1;
  if (theta < 1e-6) {
    return M3D::Identity() + 1 / 2 * Omega_1 + 1 / 6 * Omega_2;
  }
  else {
    const double A = std::sin(theta) / theta;
    const double B = (1 - std::cos(theta)) / (theta * theta);
    const M3D  dR = M3D::Identity() + A * Omega_1 + B * Omega_2;
    return dR;
  }
}


class stateTransitionTest{
  public:
    stateTransitionTest(ros::NodeHandle nh, ros::NodeHandle pnh): nh_(nh), pnh_(pnh), cnt_(0) {}
    ~stateTransitionTest() {}

    void run();
    void imuCallback(const sensor_msgs::Imu::ConstPtr& imuIn);

  public:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Subscriber subImu_;
    std::queue<Imu> imuBuf_;
    StatePredictor* nominalFilter_;
    StatePredictor* trueFilter_;
    GlobalState nominalState_;
    GlobalState trueState_;
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
  nominalFilter_ = new StatePredictor();
  trueFilter_ = new StatePredictor();
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
    nominalFilter_->initialization(imu_last_.time, V3D(0, 0, 0), V3D(0, 0, 0),
                            ba_init_, bw_init_, imu_last_.acc,
                            imu_last_.gyr);

    trueFilter_->initialization(imu_last_.time, V3D(0, 0, 0), V3D(0, 0, 0),
                            ba_init_, bw_init_, imu_last_.acc,
                            imu_last_.gyr);
    cnt_++;
  }
  else if (cnt_ <= 10) {   
    double dt = imu_cur_.time - last_imu_time_;
    nominalFilter_->predict(dt, imu_cur_.acc, imu_cur_.gyr, true);
    nominalState_ = nominalFilter_->state_;

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

    // Add noise to true state and calculate true state
    trueFilter_->state_.ba_ += error_noise.middleRows(6, 3) * dt;
    trueFilter_->state_.bw_ += error_noise.middleRows(9, 3) * dt;
    trueFilter_->predict(dt, imu_cur_.acc - error_noise.middleRows(0, 3), imu_cur_.gyr - error_noise.middleRows(3, 3), true);
    trueState_ = trueFilter_->state_;

    // Calculate the derivative of the error state by analytical method
    Eigen::MatrixXd X_nominal = Eigen::MatrixXd::Identity(5, 5), X_true = Eigen::MatrixXd::Identity(5, 5);
    X_nominal.block<3, 3>(0, 0) = nominalState_.qbn_.toRotationMatrix();
    X_nominal.block<3, 1>(0, 3) = nominalState_.vn_;
    X_nominal.block<3, 1>(0, 4) = nominalState_.rn_;
    X_true.block<3, 3>(0, 0) = trueState_.qbn_.toRotationMatrix();
    X_true.block<3, 1>(0, 3) = trueState_.vn_;
    X_true.block<3, 1>(0, 4) = trueState_.rn_;
    Eigen::MatrixXd omega_X = (X_nominal * X_true.inverse()).log();
    Eigen::MatrixXd error_state = Eigen::MatrixXd::Zero(GlobalState::DIM_OF_STATE_, 1);
    Eigen::MatrixXd error_state_dot = Eigen::MatrixXd::Zero(GlobalState::DIM_OF_STATE_, 1);
    error_state.middleRows(0, 3) = omega_X.block<3, 1>(0, 4);
    error_state.middleRows(3, 3) = omega_X.block<3, 1>(0, 3);
    error_state.middleRows(6, 3) = vee(omega_X.block<3, 3>(0, 0));
    error_state.middleRows(9, 3) = nominalState_.bw_ - trueState_.bw_;
    error_state.middleRows(12, 3) = nominalState_.ba_ - trueState_.ba_;
    error_state.middleRows(15, 3) = nominalState_.gn_ - trueState_.gn_;
    error_state_dot = nominalFilter_->F_inekf * error_state + nominalFilter_->G_inkef * error_noise;

    // Calculate the increment of the error state and compare with next frame
    if (cnt_ > 1) {
      std::cout << "Current error state is: " << std::endl;
      std::cout << error_state.transpose() << std::endl;
      Eigen::MatrixXd X_last = Eigen::MatrixXd::Identity(5, 5), X_incre = Eigen::MatrixXd::Identity(5, 5);
      X_last.block<3, 3>(0, 0) = expSO_3(last_error_state_.middleRows(6, 3));
      X_last.block<3, 1>(0, 3) = expSE_3(last_error_state_.middleRows(6, 3), last_error_state_.middleRows(3, 3));
      X_last.block<3, 1>(0, 4) = expSE_3(last_error_state_.middleRows(6, 3), last_error_state_.middleRows(0, 3));
      X_incre.block<3, 3>(0, 0) = expSO_3(last_error_state_dot_.middleRows(6, 3)*dt);
      X_incre.block<3, 1>(0, 3) = expSE_3(last_error_state_dot_.middleRows(6, 3)*dt, last_error_state_dot_.middleRows(3, 3)*dt);
      X_incre.block<3, 1>(0, 4) = expSE_3(last_error_state_dot_.middleRows(6, 3)*dt, last_error_state_dot_.middleRows(0, 3)*dt);
      Eigen::MatrixXd omega_cur = (X_incre *X_last).log();
      Eigen::MatrixXd incre_error_state = Eigen::MatrixXd::Zero(GlobalState::DIM_OF_STATE_, 1);
      incre_error_state.middleRows(0, 3) = omega_cur.block<3, 1>(0, 4);
      incre_error_state.middleRows(3, 3) = omega_cur.block<3, 1>(0, 3);
      incre_error_state.middleRows(6, 3) = vee(omega_cur.block<3, 3>(0, 0));
      incre_error_state.middleRows(9, 9) = last_error_state_.middleRows(9, 9) + dt * last_error_state_dot_.middleRows(9, 9);
      std::cout << "Current error state with increment is: " << std::endl;
      std::cout << incre_error_state.transpose() << std::endl;
      std::cout << std::endl << std::endl;
    }

    last_error_state_ = error_state;
    last_error_state_dot_ = error_state_dot;
    last_imu_time_ = imu_cur_.time;
    imu_last_ = imu_cur_;
    cnt_++;
  }
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