#ifndef __DEEP_PHYSICS_FUNCTIONS_H__
#define __DEEP_PHYSICS_FUNCTIONS_H__
#include "dart/dart.hpp"

namespace DPhy
{

// Utilities
std::vector<double> split_to_double(const std::string& input, int num);
    std::vector<double> split_to_double(const std::string& input);
Eigen::Vector3d string_to_vector3d(const std::string& input);
Eigen::VectorXd string_to_vectorXd(const std::string& input, int n);
    Eigen::VectorXd string_to_vectorXd(const std::string& input);
Eigen::Matrix3d string_to_matrix3d(const std::string& input);

double exp_of_squared(const Eigen::VectorXd& vec,double sigma = 1.0);
double exp_of_squared(const Eigen::Vector3d& vec,double sigma = 1.0);
double exp_of_squared(const Eigen::MatrixXd& mat,double sigma = 1.0);
std::pair<int, double> maxCoeff(const Eigen::VectorXd& in);

double RadianClamp(double input);

Eigen::Quaterniond DARTPositionToQuaternion(Eigen::Vector3d in);
Eigen::Vector3d QuaternionToDARTPosition(const Eigen::Quaterniond& in);
void QuaternionNormalize(Eigen::Quaterniond& in);

void SetBodyNodeColors(dart::dynamics::BodyNode* bn, const Eigen::Vector3d& color);
void SetSkeletonColor(const dart::dynamics::SkeletonPtr& object, const Eigen::Vector3d& color);
void SetSkeletonColor(const dart::dynamics::SkeletonPtr& object, const Eigen::Vector4d& color);

void EditBVH(std::string& path);
Eigen::Quaterniond GetYRotation(Eigen::Quaterniond q);

Eigen::Vector3d changeToRNNPos(Eigen::Vector3d pos);
Eigen::Isometry3d getJointTransform(dart::dynamics::SkeletonPtr skel, std::string bodyname);
Eigen::Vector4d rootDecomposition(dart::dynamics::SkeletonPtr skel, Eigen::VectorXd positions);
Eigen::VectorXd solveIK(dart::dynamics::SkeletonPtr skel, const std::string& bodyname, const Eigen::Vector3d& delta,  const Eigen::Vector3d& offset);
Eigen::VectorXd solveMCIK(dart::dynamics::SkeletonPtr skel, const std::vector<std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>>& constraints);

}

#endif
