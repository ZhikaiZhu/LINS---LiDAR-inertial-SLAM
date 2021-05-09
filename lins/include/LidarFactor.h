#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <parameters.h>

struct LidarEdgeFactor
{
	LidarEdgeFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
					Eigen::Vector3d last_point_b_, double s_, Eigen::Vector3d rn_, Eigen::Quaterniond qbn_, double cov_)
		: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_), rn(rn_), qbn(qbn_), cov(cov_) {}

	template <typename T>
	bool operator()(const T *dx, T *residual) const
	{
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
		Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

        Eigen::Matrix<T, 3, 1> Tbl{T(INIT_TBL.x()), T(INIT_TBL.y()), T(INIT_TBL.z())};
        Eigen::Quaternion<T> Rbl = INIT_RBL.cast<T>();

        Eigen::Matrix<T, 18, 1> delta_x;
        for (size_t i = 0; i < 18; ++i) {
            delta_x(i, 0) = dx[i];
        }

        Eigen::Matrix<T, 3, 1> nominal_rn{T(rn[0]), T(rn[1]), T(rn[2])};
        Eigen::Matrix<T, 3, 1> delta_rn{T(dx[0]), T(dx[1]), T(dx[2])};
        Eigen::Quaternion<T> nominal_qbn{T(qbn.w()), T(qbn.x()), T(qbn.y()), T(qbn.z())};
        Eigen::Matrix<T, 3, 1> d_theta{T(dx[6]), T(dx[7]), T(dx[8])};

        Eigen::Quaternion<T> dq{T(1), T(0), T(0), T(0)};
        if (d_theta.norm() > 1e-10) {
            Eigen::AngleAxis<T> axis_dq(d_theta.norm(), d_theta / d_theta.norm());
            dq = axis_dq;
        }
        
        //Eigen::Quaternion<T> dq = axis2Quat(delta_x.template segment<3>(6));
        Eigen::Quaternion<T> q_last_curr = (dq * nominal_qbn).normalized();
        Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);

        nominal_rn = dq * nominal_rn + delta_rn;
        /*
        Eigen::Matrix<T, 3, 1> phi = Quat2axis(nominal_qbn);
        Eigen::Quaternion<T> q_last_curr = axis2Quat(s * phi);
        q_last_curr.normalized();
        */

        Eigen::Matrix<T, 3, 1> t_last_curr = T(s) * nominal_rn;
        Eigen::Matrix<T, 3, 1> lp = Rbl.inverse() * (q_last_curr * (Rbl * cp + Tbl) + t_last_curr - Tbl);

		Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
		Eigen::Matrix<T, 3, 1> de = lpa - lpb;

        residual[0] = nu.norm() / (de.norm() * T(cov));

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_,
									   const Eigen::Vector3d last_point_b_, const double s_, 
                                       const Eigen::Vector3d rn_, const Eigen::Quaterniond qbn_, const double cov_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarEdgeFactor, 1, 18>(
			new LidarEdgeFactor(curr_point_, last_point_a_, last_point_b_, s_, rn_, qbn_, cov_)));
	}

	Eigen::Vector3d curr_point, last_point_a, last_point_b;
	double s;
    Eigen::Vector3d rn;
    Eigen::Quaterniond qbn;
    double cov;
};

struct LidarPlaneFactor
{
	LidarPlaneFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_, 
                    Eigen::Vector3d last_point_b_, Eigen::Vector3d last_point_c_, 
                    double s_, Eigen::Vector3d rn_, Eigen::Quaterniond qbn_, double cov_)
		: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), 
        last_point_c(last_point_c_), s(s_), rn(rn_), qbn(qbn_), cov(cov_) {}

	template <typename T>
	bool operator()(const T *dx, T *residual) const
	{
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
		Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};
		Eigen::Matrix<T, 3, 1> lpc{T(last_point_c.x()), T(last_point_c.y()), T(last_point_c.z())};

        Eigen::Matrix<T, 3, 1> Tbl{T(INIT_TBL.x()), T(INIT_TBL.y()), T(INIT_TBL.z())};
        Eigen::Quaternion<T> Rbl = INIT_RBL.cast<T>();

        Eigen::Matrix<T, 18, 1> delta_x;
        for (size_t i = 0; i < 18; ++i) {
            delta_x(i, 0) = dx[i];
        }

        Eigen::Matrix<T, 3, 1> nominal_rn{T(rn[0]), T(rn[1]), T(rn[2])};
        Eigen::Matrix<T, 3, 1> delta_rn{T(dx[0]), T(dx[1]), T(dx[2])};
        Eigen::Quaternion<T> nominal_qbn{T(qbn.w()), T(qbn.x()), T(qbn.y()), T(qbn.z())};
        Eigen::Matrix<T, 3, 1> d_theta{T(dx[6]), T(dx[7]), T(dx[8])};

        Eigen::Quaternion<T> dq{T(1), T(0), T(0), T(0)};
        if (d_theta.norm() > 1e-10) {
            Eigen::AngleAxis<T> axis_dq(d_theta.norm(), d_theta / d_theta.norm());
            dq = axis_dq;
        }

        //Eigen::Quaternion<T> dq = axis2Quat(delta_x.template segment<3>(6));
        Eigen::Quaternion<T> q_last_curr = (dq * nominal_qbn).normalized();
        Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);

        nominal_rn = dq * nominal_rn + delta_rn;
        /*
        Eigen::Matrix<T, 3, 1> phi = Quat2axis(nominal_qbn);
        Eigen::Quaternion<T> q_last_curr = axis2Quat(s * phi);
        q_last_curr.normalized();
        */




















































































































































































        Eigen::Matrix<T, 3, 1> t_last_curr = T(s) * nominal_rn;
        Eigen::Matrix<T, 3, 1> lp = Rbl.inverse() * (q_last_curr * (Rbl * cp + Tbl) + t_last_curr - Tbl);
        Eigen::Matrix<T, 3, 1> M = (lpa - lpb).cross(lpa - lpc);
        residual[0] = ((lp - lpa).transpose() * M).norm() / (M.norm() * T(cov));
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_,
									   const Eigen::Vector3d last_point_b_, const Eigen::Vector3d last_point_c_,
                                       const double s_, const Eigen::Vector3d rn_, const Eigen::Quaterniond qbn_, const double cov_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneFactor, 1, 18>(
			new LidarPlaneFactor(curr_point_, last_point_a_, last_point_b_, last_point_c_, s_, rn_, qbn_, cov_)));
	}

	Eigen::Vector3d curr_point, last_point_a, last_point_b, last_point_c;
	double s;
    Eigen::Vector3d rn;
    Eigen::Quaterniond qbn;
    double cov;
};
