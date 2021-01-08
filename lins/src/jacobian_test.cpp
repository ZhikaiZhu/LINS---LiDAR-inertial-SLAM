#include <Estimator.h>
#include <iostream>

int main(int argc, char** argv) {

    PointType pointOri;
    pointOri.x = 1.1;
    pointOri.y = 1.2;
    pointOri.z = 1.3;
    PointType pointSel;
    PointType coeff, tripod1, tripod2;

    Q4D qbn_ = math_utils::axis2Quat(V3D(0.1, 0.35, -0.81));
    //std::cin >> qbn_.w() >> qbn_.x() >> qbn_.y() >> qbn_.z();
    qbn_.normalize();
    V3D rn_;
    rn_.setZero();

    V3D P0xyz(pointOri.x, pointOri.y, pointOri.z);
    Q4D R21xyz = qbn_;
    V3D T112xyz = rn_;
    P0xyz = R21xyz * P0xyz + T112xyz;
    V3D P1xyz(1.0, 2.0, 3.0);
    V3D P2xyz(6.0, 4.0, 17.0);

    V3D P = math_utils::skew(P0xyz - P1xyz) * (P0xyz - P2xyz);
    float r = P.norm();
    float d12 = (P1xyz - P2xyz).norm();
    float res = r / d12;

    V3D jacxyz =
        P.transpose() * math_utils::skew(P2xyz - P1xyz) / (d12 * r);

    //std::cout << "Analytic value: " << std::endl << jacxyz << std::endl;

    V3D jac_num;
    double epsilon = 1e-3;
    for (size_t i{0}; i < 3; ++i) {
        P0xyz(i) += epsilon;
        //std::cout << std::setprecision(10) << "current: " << std::endl << P0xyz << std::endl;
        P = math_utils::skew(P0xyz - P1xyz) * (P0xyz - P2xyz);
        r = P.norm();
        float res_ep = r / d12;
        jac_num(i) = (res_ep - res) / epsilon;
        P0xyz(i) -= epsilon;
    }

    //std::cout << "Numeric value: " << std::endl << jac_num << std::endl;

    V3D axis = math_utils::Quat2axis(qbn_);
    V3D P3xyz(pointOri.x, pointOri.y, pointOri.z);
    V3D jac_angle = jacxyz.transpose() * (-qbn_.toRotationMatrix() * math_utils::skew(P3xyz)); // * math_utils::Rinvleft(-axis);
    
    std::cout << "Analytic value: " << std::endl << jac_angle << std::endl;

    V3D jac_angle_num;
    //cin >> epsilon;
    for (size_t i{0}; i < 3; ++i) {
        V3D tmp;
        tmp.setZero();
        tmp(i) = epsilon;
        Q4D dq = axis2Quat(tmp);
        Q4D new_qbn_ = (qbn_ * dq).normalized();
        V3D new_P0xyz(pointOri.x, pointOri.y, pointOri.z);
        new_P0xyz = new_qbn_ * new_P0xyz + T112xyz;
        P = math_utils::skew(new_P0xyz - P1xyz) * (new_P0xyz - P2xyz);
        r = P.norm();
        float new_res = r / d12;
        jac_angle_num(i) = (new_res - res) / epsilon;
    }
    std::cout << "Numeric value: " << std::endl << jac_angle_num << std::endl;

    M3D jac_box_minus_num;
    Q4D qin(1, 2, 3, 4);
    qin.normalize();
    V3D res_box_minus = math_utils::Quat2axis(qin.inverse() * qbn_);

    for (size_t i{0}; i < 3; ++i) {
        V3D tmp;
        tmp.setZero();
        tmp(i) = epsilon;
        //Q4D dq = axis2Quat(tmp);
        Q4D dq(1, (i == 0) * epsilon, (i == 1) * epsilon, (i == 2) * epsilon);
        Q4D new_qbn_ = (qin * dq).normalized();
        V3D res_tmp = math_utils::Quat2axis(new_qbn_.inverse() * qbn_);
        jac_box_minus_num.col(i) = (res_tmp - res_box_minus) / epsilon;
    }

    std::cout <<  math_utils::Rinvleft(-math_utils::Quat2axis(qin)) << std::endl;
    std::cout <<  jac_box_minus_num << std::endl;
    std::cout <<  jac_box_minus_num.inverse() << std::endl;
}