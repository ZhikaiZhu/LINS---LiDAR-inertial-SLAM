#include <ros/ros.h>

#include <parameters.h>
#include <MapRingBuffer.h>
#include <math_utils.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <tic_toc.h>

#include <StateEstimator.hpp>
#include <iostream>
#include <fstream>
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

// body frame to world frame
template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 3> euler2R(
  const Eigen::MatrixBase<Derived> &rpy) {
  typedef typename Derived::Scalar Scalar_t;

  Scalar_t r = rpy(0);
  Scalar_t p = rpy(1);
  Scalar_t y = rpy(2);

  Scalar_t cr = cos(r), sr = sin(r);
  Scalar_t cp = cos(p), sp = sin(p);
  Scalar_t cy = cos(y), sy = sin(y);

  Eigen::Matrix<Scalar_t, 3, 3> Rwb;
  Rwb << cy*cp, cy*sp*sr - sy*cr, sy*sr + cy*cr*sp,
         sy*cp, cy*cr + sy*sr*sp, sp*sy*cr - cy*sr,
           -sp,            cp*sr,            cp*cr;

  return Rwb;
}

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 3> eulerAngleRate2BodyAxisRate(
  const Eigen::MatrixBase<Derived> &rpy) {
  typedef typename Derived::Scalar Scalar_t;

  Scalar_t r = rpy(0);
  Scalar_t p = rpy(1);

  Scalar_t cr = cos(r), sr = sin(r);
  Scalar_t cp = cos(p), sp = sin(p);

  Eigen::Matrix<Scalar_t, 3, 3> L;
  L << 1,   0,   -sp,
       0,  cr, sr*cp,
       0, -sr, cr*cp;

  return L;
}

class imuMotionData {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    imuMotionData(V3D& acc_bias, V3D& gyr_bias): imu_acc_bias_(acc_bias), imu_gyr_bias_(gyr_bias) {}
    ~imuMotionData() {}

    void createImuData(double t);
    void addImuNoise();

  public:
    double timestamp_;
    M3D Rwb_;
    V3D imu_velocity_;
    V3D twb_;
    V3D imu_acc_;
    V3D imu_gyr_;

    V3D imu_acc_bias_;
    V3D imu_gyr_bias_;
    Eigen::MatrixXd noise_;
};

void imuMotionData::createImuData(double t) {
  double ellipse_x = 15.0;
  double ellipse_y = 20.0;
  double z = 1.0;         
  double k1 = M_PI / 10;
  double k2 = 10.0;
  double k = k1 * k1;

  // translation 
  V3D pos(ellipse_x * cos(k1 * t) + 5, ellipse_y * sin(k1 * t) + 5, z * sin(k2 * k1 * t) + 5);
  V3D pos_dot(-k1 * ellipse_x * sin(k1 * t), k1 * ellipse_y * cos(k1 * t), z * k2 * k1 * cos(k2 * k1 * t));
  V3D pos_dot2(-k * ellipse_x * cos(k1 * t), -k * ellipse_y * sin(k1 * t), -z * k2 * k2 * k * sin(k2 * k1 * t));
  twb_ = pos;
  imu_velocity_ = pos_dot;

  // Rotation
  double k_roll = 0.1;
  double k_pitch = 0.2;
  V3D eulerAngle(k_roll * cos(t), k_pitch * sin(t), k1 * t);
  V3D eulerAngleRate(-k_roll * sin(t), k_pitch * cos(t), k1);
  Rwb_ = euler2R(eulerAngle);
  imu_gyr_ = eulerAngleRate2BodyAxisRate(eulerAngle) * eulerAngleRate; // euler angle rate transform to body frame

  V3D gn(0, 0, -9.81);
  imu_acc_ = Rwb_.transpose() * (pos_dot2 - gn);
  timestamp_ = t;
}

void imuMotionData::addImuNoise() {
  // Create noise matrix n_t^a, n_t^w, n_t^{ba}, n_t^{bw}
  int imu_frequency = 100;
  const double dt = 1. / imu_frequency;
  double sqrt_cov[4] = {ACC_N * ug, GYR_N * dph, ACC_W * ugpsHz, GYR_W * dpsh};
  noise_ = Eigen::MatrixXd::Zero(GlobalState::DIM_OF_NOISE_, 1);
  V3D noise_tmp;
  cv::RNG rng( cv::getTickCount() );
  cv::Mat tmp = cv::Mat::zeros(3, 1, CV_64FC1);
  for (int i = 0; i < 4; i++) {
    rng.fill(tmp, cv::RNG::NORMAL, 0., sqrt_cov[i]);
    cv::cv2eigen(tmp, noise_tmp);
    noise_.middleRows(3 * i, 3) = noise_tmp;
  }

  imu_acc_ = imu_acc_ + sqrt_cov[0] * M3D::Identity() * noise_.middleRows(0, 3) / sqrt(dt) + imu_acc_bias_;
  imu_gyr_ = imu_gyr_ + sqrt_cov[1] * M3D::Identity() * noise_.middleRows(3, 3) / sqrt(dt) + imu_gyr_bias_;

  // Update bias
  imu_acc_bias_ += sqrt_cov[2] * noise_.middleRows(6, 3) * sqrt(dt);
  imu_gyr_bias_ += sqrt_cov[3] * noise_.middleRows(9, 3) * sqrt(dt);
}


class stateTransitionTest {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    stateTransitionTest(ros::NodeHandle nh, ros::NodeHandle pnh): nh_(nh), pnh_(pnh), cnt_(0) {}
    ~stateTransitionTest() {
      if (nominalFilter_ != nullptr)
        delete nominalFilter_;
      if (trueFilter_ != nullptr)
        delete trueFilter_;
      if (imuData_last_ != nullptr)
        delete imuData_last_;
    }

    void run();
    void imuCallback(const sensor_msgs::Imu::ConstPtr& imuIn);

    void savePose(std::string filename, const std::vector<imuMotionData*>& pose);
    void testInEKF();

  public:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Subscriber subImu_;
    //std::queue<Imu> imuBuf_;
    StatePredictor* nominalFilter_;
    StatePredictor* trueFilter_;
    GlobalState nominalState_;
    GlobalState trueState_;
    GlobalState initState_;
    imuMotionData* imuData_last_;
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
  //subImu_ = pnh_.subscribe<sensor_msgs::Imu>(IMU_TOPIC, 100, &stateTransitionTest::imuCallback, this);
  imuData_last_ = new imuMotionData(ba_init_, bw_init_);
}

void stateTransitionTest::savePose(std::string filename, const std::vector<imuMotionData*>& pose) {
  std::ofstream fout(filename, std::ios::out);
  fout.close();
  std::ofstream imu_data(filename, std::ios::app);

  if (imu_data.is_open()) {
    std::cout << filename << " is created." << std::endl;
  }

  for (size_t i = 0; i < pose.size(); i++) {
    imuMotionData* data_tmp = pose[i];
    double time = data_tmp->timestamp_;
    Q4D q(data_tmp->Rwb_);
    V3D vel = data_tmp->imu_velocity_;
    V3D p = data_tmp->twb_;
    V3D acc = data_tmp->imu_acc_;
    V3D gyr = data_tmp->imu_gyr_;

    imu_data << time << " "
             << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " "
             << vel(0) << " " << vel(1) << " " << vel(2) << " "
             << p(0) << " " << p(1) << " " << p(2) << " "
             << acc(0) << " " << acc(1) << " " << acc(2) << " "
             << gyr(0) << " " << gyr(1) << " " << gyr(2) << std::endl;
  }
  imu_data.close();
}

void stateTransitionTest::testInEKF() {
  double t_start = 0.0;
  double t_end = 20.0;
  double imu_frequency = 100.0;
  V3D acc_bias = ba_init_;
  V3D gyr_bias = bw_init_;
  std::vector<imuMotionData*> imudata;
  std::vector<imuMotionData*> imudata_noise;
  for (double t = t_start; t < t_end;) {
    imuMotionData* imuData_cur = new imuMotionData(acc_bias, gyr_bias);
    imuData_cur->createImuData(t);
    imudata.emplace_back(imuData_cur);
    if (cnt_ == 0) {
      imuData_last_ = imuData_cur;
      imuData_last_->addImuNoise();
      imudata_noise.emplace_back(imuData_last_);
      initState_.rn_ = imuData_last_->twb_;
      initState_.vn_ = imuData_last_->imu_velocity_;
      initState_.qbn_ = imuData_last_->Rwb_;
      initState_.ba_ = imuData_last_->imu_acc_bias_;
      initState_.bw_ = imuData_last_->imu_gyr_bias_;
      initState_.gn_ = V3D(0, 0, -9.81);
      nominalFilter_->initialization(t, V3D(0, 0, 0), V3D(0, 0, 0),
                            ba_init_, bw_init_, imuData_last_->imu_acc_,
                            imuData_last_->imu_gyr_);

      nominalFilter_->state_.vn_ = imuData_last_->Rwb_.transpose() * imuData_last_->imu_velocity_;
      nominalFilter_->state_.gn_ = imuData_last_->Rwb_.transpose() * V3D(0, 0, -9.81); 

      acc_bias = imuData_last_->imu_acc_bias_;
      gyr_bias = imuData_last_->imu_gyr_bias_;
      t += 1.0 / imu_frequency;
      cnt_++;
    }
    else {
      std::cout << "----------for the " << cnt_ << "th time----------" << std::endl;
      imuData_cur->addImuNoise();
      trueState_.qbn_ = initState_.qbn_.toRotationMatrix().transpose() * imuData_cur->Rwb_;         // R_{k}^{1}
      trueState_.vn_ = initState_.qbn_.inverse() * imuData_cur->imu_velocity_;                      // V_{k}^{1}
      trueState_.rn_ = initState_.qbn_.inverse() * (imuData_cur->twb_ - initState_.rn_);            // P_{k}^{1}
      trueState_.ba_ = imuData_cur->imu_acc_bias_;                                                  // ba
      trueState_.bw_ = imuData_cur->imu_gyr_bias_;                                                  // bw
      trueState_.gn_ = imuData_cur->Rwb_.transpose() * initState_.gn_;                              // g^{k}
      imudata_noise.emplace_back(imuData_cur);
      double dt = imuData_cur->timestamp_ - imuData_last_->timestamp_;
      nominalFilter_->predict(dt, imuData_cur->imu_acc_, imuData_cur->imu_gyr_, true);
      nominalState_ = nominalFilter_->state_;
      nominalState_.gn_ = nominalFilter_->state_.qbn_.inverse() * nominalFilter_->state_.gn_;
      nominalState_.gn_ = nominalState_.gn_ * 9.81 / nominalState_.gn_.norm();

      // Calculate the derivative of the error state by analytical method
      Eigen::MatrixXd X_nominal = Eigen::MatrixXd::Identity(5, 5), X_true = Eigen::MatrixXd::Identity(5, 5);
      X_nominal.block<3, 3>(0, 0) = nominalState_.qbn_.toRotationMatrix();
      X_nominal.block<3, 1>(0, 3) = nominalState_.vn_;
      X_nominal.block<3, 1>(0, 4) = nominalState_.rn_;
      X_true.block<3, 3>(0, 0) = trueState_.qbn_.toRotationMatrix();
      X_true.block<3, 1>(0, 3) = trueState_.vn_;
      X_true.block<3, 1>(0, 4) = trueState_.rn_;
      Eigen::MatrixXd omega_X = (X_true * X_nominal.inverse()).log();
      Eigen::MatrixXd error_state = Eigen::MatrixXd::Zero(GlobalState::DIM_OF_STATE_, 1);
      Eigen::MatrixXd error_state_dot = Eigen::MatrixXd::Zero(GlobalState::DIM_OF_STATE_, 1);
      error_state.middleRows(0, 3) = omega_X.block<3, 1>(0, 4);
      error_state.middleRows(3, 3) = omega_X.block<3, 1>(0, 3);
      error_state.middleRows(6, 3) = vee(omega_X.block<3, 3>(0, 0));
      error_state.middleRows(9, 3) = trueState_.ba_ - nominalState_.ba_;
      error_state.middleRows(12, 3) = trueState_.bw_ - nominalState_.bw_;
      error_state.middleRows(15, 3) = trueState_.gn_ - nominalState_.gn_;
      error_state_dot = nominalFilter_->F_inekf * error_state + nominalFilter_->G_inekf * imuData_cur->noise_;

      // Calculate the increment of the error state and compare with next frame
      if (cnt_ > 1) {
        std::cout << "Current error state is: " << std::endl;
        std::cout << error_state.transpose() << std::endl;
        Eigen::MatrixXd incre_error_state = Eigen::MatrixXd::Zero(GlobalState::DIM_OF_STATE_, 1);
        incre_error_state = last_error_state_ + last_error_state_dot_ * dt;
        std::cout << "Current error state with increment is: " << std::endl;
        std::cout << incre_error_state.transpose() << std::endl;
        std::cout << std::endl << std::endl;
      }

      acc_bias = imuData_cur->imu_acc_bias_;
      gyr_bias = imuData_cur->imu_gyr_bias_;
      last_error_state_ = error_state;
      last_error_state_dot_ = error_state_dot;
      imuData_last_ = imuData_cur;
      t += 1.0 / imu_frequency;
      cnt_++;
    }

  }

  // Save simulation data 
  //savePose("/home/spc/output/imu_simulation_data/imu_pose.txt", imudata);
  //savePose("/home/spc/output/imu_simulation_data/imu_pose_noise.txt", imudata_noise);
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
    error_state_dot = nominalFilter_->F_inekf * error_state + nominalFilter_->G_inekf * error_noise;

    // Calculate the increment of the error state and compare with next frame
    if (cnt_ > 1) {
      std::cout << "Current error state is: " << std::endl;
      std::cout << error_state.transpose() << std::endl;
      /*Eigen::MatrixXd X_last = Eigen::MatrixXd::Identity(5, 5), X_incre = Eigen::MatrixXd::Identity(5, 5);
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
      incre_error_state.middleRows(9, 9) = last_error_state_.middleRows(9, 9) + dt * last_error_state_dot_.middleRows(9, 9); */
      Eigen::MatrixXd incre_error_state = Eigen::MatrixXd::Zero(GlobalState::DIM_OF_STATE_, 1);
      incre_error_state = last_error_state_ + last_error_state_dot_ * dt;
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
  stateTrans.testInEKF();

  //ros::spin();
  ros::spinOnce();
  return 0;
}