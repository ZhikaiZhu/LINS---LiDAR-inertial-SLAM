#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include<eigen3/Eigen/Cholesky>

struct PriorFactor
{
    PriorFactor(Eigen::Matrix<double, 18, 18> covariance) : covariance_(covariance) {}

	template <typename T>
	bool operator()(const T *dx, T *residual) const
	{
        Eigen::Matrix<T, 18, 1> delta_x;
        for (size_t i = 0; i < 18; ++i) {
            delta_x(i, 0) = dx[i];
        }
		Eigen::Matrix<double, 18, 18> tmp = Eigen::LLT<Eigen::Matrix<double, 18, 18>>(covariance_.inverse()).matrixL().transpose();
		Eigen::Matrix<T, 18, 18> sqrt_info = tmp.cast<T>();
        Eigen::Matrix<T, 18, 1> r = sqrt_info * delta_x;
		
		for (size_t i = 0; i < 18; ++i) {
			residual[i] = r(i, 0);
		}

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Matrix<double, 18, 18> covariance)
	{
		return (new ceres::AutoDiffCostFunction<
				PriorFactor, 18, 18>(new PriorFactor(covariance)));
	}

    Eigen::Matrix<double, 18, 18> covariance_;
    
};