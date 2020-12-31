#include <Estimator.h>
#include <iostream>

std::ostream& operator<<(std::ostream& out, Eigen::Quaterniond q) {
    out << q.w() << std::endl << q.vec() << std::endl;
    return out;
}

int main(int argc, char** argv) {
    double s = 0.;
    std::cout << isnan(s) << std::endl;
    std::cout << s << std::endl;
    for (size_t i = 0; i < 10; ++i) {
        Eigen::Vector3d d_theta = Eigen::Vector3d::Random();
        Eigen::AngleAxisd axis_dq(d_theta.norm(), d_theta / d_theta.norm());
        Eigen::Quaterniond dq;
        dq = axis_dq;
        Eigen::Quaterniond q_identity{1, 0, 0, 0};
        double s = 1 / (i * 0.1 + 1.2);
        Eigen::Quaterniond q_slerp = q_identity.slerp(s, dq);
        std::cout << "test" << std::endl;
        std::cout << q_slerp;
        Eigen::Matrix<double, 3, 1> phi = math_utils::Quat2axis(dq);
        std::cout << axis2Quat(s * phi) << std::endl;
    }
    return 0;
}