#include "Functions.h"
#include "CharacterConfigurations.h"
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm> 
#include <cctype>
#include <locale>

namespace DPhy
{

// trim from start (in place)
static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

// trim from start (copying)
static inline std::string ltrim_copy(std::string s) {
    ltrim(s);
    return s;
}

// trim from end (copying)
static inline std::string rtrim_copy(std::string s) {
    rtrim(s);
    return s;
}

// trim from both ends (copying)
static inline std::string trim_copy(std::string s) {
    trim(s);
    return s;
}

template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
       	*(result++) = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim=' ') {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

std::string join(const std::vector<std::string> &v, char delim=' '){
	std::stringstream ss;
	for(size_t i = 0; i < v.size(); ++i)
	{
		if(i != 0)
			ss << delim;
		ss << v[i];
	}

	return ss.str();
}

std::vector<double> split_to_double(const std::string& input, int num)
{
    std::vector<double> result;
    std::string::size_type sz = 0, nsz = 0;
    for(int i = 0; i < num; i++){
        result.push_back(std::stold(input.substr(sz), &nsz));
        sz += nsz;
    }
    return result;
}

std::vector<double> split_to_double(const std::string& input)
{
    std::vector<double> result;
    std::string::size_type sz = 0, nsz = 0;
    while(sz< input.length()){
        result.push_back(std::stold(input.substr(sz), &nsz));
        sz += nsz;
    }
    return result;
}

Eigen::Vector3d string_to_vector3d(const std::string& input){
	std::vector<double> v = split_to_double(input, 3);
	Eigen::Vector3d res;
	res << v[0], v[1], v[2];

	return res;
}

Eigen::VectorXd string_to_vectorXd(const std::string& input, int n){
	std::vector<double> v = split_to_double(input, n);
	Eigen::VectorXd res(n);
	for(int i = 0; i < n; i++){
		res[i] = v[i];
	}
	return res;
}

Eigen::VectorXd string_to_vectorXd(const std::string& input){
    std::vector<double> v = split_to_double(input);
    Eigen::VectorXd res(v.size());
    for(int i = 0; i < v.size(); i++){
        res[i] = v[i];
    }
    return res;
}

    Eigen::Matrix3d string_to_matrix3d(const std::string& input){
	std::vector<double> v = split_to_double(input, 9);
	Eigen::Matrix3d res;
	res << v[0], v[1], v[2],
			v[3], v[4], v[5],
			v[6], v[7], v[8];

	return res;
}

double RadianClamp(double input){
	return std::fmod(input+M_PI, 2*M_PI)-M_PI;
}

double exp_of_squared(const Eigen::VectorXd& vec,double sigma)
{
	return exp(-1.0*vec.dot(vec)/(sigma*sigma)/vec.rows());
}
double exp_of_squared(const Eigen::Vector3d& vec,double sigma)
{
	return exp(-1.0*vec.dot(vec)/(sigma*sigma)/vec.rows());
}
double exp_of_squared(const Eigen::MatrixXd& mat,double sigma)
{
	return exp(-1.0*mat.squaredNorm()/(sigma*sigma)/mat.size());
}


std::pair<int, double> maxCoeff(const Eigen::VectorXd& in){
	double m = 0;
	int idx = 0;
	for(int i = 0; i < in.rows(); i++){
		if( m < in[i]){
			m = in[i];
			idx = i;
		}
	}
	return std::make_pair(idx, m);
}

void SetBodyNodeColors(dart::dynamics::BodyNode* bn, const Eigen::Vector3d& color)
{
	auto visualShapeNodes = bn->getShapeNodesWith<dart::dynamics::VisualAspect>();
	for(auto visualShapeNode : visualShapeNodes)
		visualShapeNode->getVisualAspect()->setColor(color);
}

void SetSkeletonColor(const dart::dynamics::SkeletonPtr& object, const Eigen::Vector3d& color)
{
	// Set the color of all the shapes in the object
	for(std::size_t i=0; i < object->getNumBodyNodes(); ++i)
	{
		Eigen::Vector3d c = color;
		dart::dynamics::BodyNode* bn = object->getBodyNode(i);
		if(bn->getName() == "Neck")
			c.head<3>() *= 0.5;
		auto visualShapeNodes = bn->getShapeNodesWith<dart::dynamics::VisualAspect>();
		for(auto visualShapeNode : visualShapeNodes)
			visualShapeNode->getVisualAspect()->setColor(c);
	}
}

void SetSkeletonColor(const dart::dynamics::SkeletonPtr& object, const Eigen::Vector4d& color)
{
	// Set the color of all the shapes in the object
	for(std::size_t i=0; i < object->getNumBodyNodes(); ++i)
	{
		Eigen::Vector4d c = color;
		dart::dynamics::BodyNode* bn = object->getBodyNode(i);
		if(bn->getName() == "Neck")
			c.head<3>() *= 0.5;
		auto visualShapeNodes = bn->getShapeNodesWith<dart::dynamics::VisualAspect>();
		for(auto visualShapeNode : visualShapeNodes)
			visualShapeNode->getVisualAspect()->setRGBA(c);
	}
}


Eigen::Quaterniond DARTPositionToQuaternion(Eigen::Vector3d in){
	if( in.norm() < 1e-8 ){
		return Eigen::Quaterniond::Identity();
	}
	Eigen::AngleAxisd aa(in.norm(), in.normalized());
	Eigen::Quaterniond q(aa);
	QuaternionNormalize(q);
	return q;
}

Eigen::Vector3d QuaternionToDARTPosition(const Eigen::Quaterniond& in){
	Eigen::AngleAxisd aa(in);
	double angle = aa.angle();
	angle = std::fmod(angle+M_PI, 2*M_PI)-M_PI;
	return angle*aa.axis();
}

void QuaternionNormalize(Eigen::Quaterniond& in){
	if(in.w() < 0){
		in.coeffs() *= -1;
	}
}

void EditBVH(std::string& path){
	double scale = 100;
	std::ifstream ifs(path);
	std::vector<std::string> out;
	std::string line;

	while(true){
		if(!std::getline(ifs, line))
			break;

		if(line == "MOTION"){
			out.push_back(line);
			break;
		}

		int space_count = line.length();
		for(int i = 0; i < line.length(); i++){
			if(line[i] != ' '){
				space_count = i;
				break;
			}
		}
		if(space_count == line.length()){
			out.push_back(line);
			continue;
		}

		std::vector<std::string> sp = split(line, ' ');
		if(sp[space_count] == "OFFSET"){
			sp[space_count+1] = std::to_string(std::stold(sp[space_count+1])*scale);
			sp[space_count+2] = std::to_string(std::stold(sp[space_count+2])*scale);
			sp[space_count+3] = std::to_string(std::stold(sp[space_count+3])*scale);

			std::string new_line = join(sp);
			out.push_back(new_line);
		}
		else{
			out.push_back(line);
			continue;			
		}
	}
	std::getline(ifs, line);
	out.push_back(line);
	std::getline(ifs, line);
	out.push_back(line);

	while(std::getline(ifs, line)){
		std::vector<std::string> sp = split(line, ' ');
		Eigen::Vector3d pos, rot;
		pos << std::stold(sp[0]), std::stold(sp[1]), std::stold(sp[2]);
		rot << std::stold(sp[3]), std::stold(sp[4]), std::stold(sp[5]);
		rot = rot*M_PI/180.;

		pos = pos * scale;
		double tmp = pos[0];
		pos[0] = pos[2];
		pos[2] = -tmp;

		Eigen::AngleAxisd rotaa;
		rotaa = Eigen::AngleAxisd(rot[0], Eigen::Vector3d::UnitZ())
			* Eigen::AngleAxisd(rot[1], Eigen::Vector3d::UnitX())
			* Eigen::AngleAxisd(rot[2], Eigen::Vector3d::UnitY());

		rotaa = Eigen::AngleAxisd(M_PI/2.0, Eigen::Vector3d::UnitY())*rotaa;
		Eigen::Matrix3d m(rotaa);
		Eigen::Vector3d fixed_rot = m.eulerAngles(2,0,1);
		fixed_rot = fixed_rot * 180./M_PI;

		sp[0] = std::to_string(pos[0]);
		sp[1] = std::to_string(pos[1]);
		sp[2] = std::to_string(pos[2]);
		sp[3] = std::to_string(fixed_rot[0]);
		sp[4] = std::to_string(fixed_rot[1]);
		sp[5] = std::to_string(fixed_rot[2]);

		std::string new_line = join(sp);
		out.push_back(new_line);
	}	
	ifs.close();

	std::ofstream outputfile(path.substr(0,path.length()-4) + std::string("_fixed_c.bvh"));
	for(auto& s : out){
		outputfile << s << std::endl;
	}
	outputfile.close();
}

Eigen::Quaterniond GetYRotation(Eigen::Quaterniond q){
	// from body joint vector
	Eigen::Vector3d rotated = q._transformVector(Eigen::Vector3d::UnitZ());
	double angle = atan2(rotated[0], rotated[2]);
	Eigen::Quaterniond ret(Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY()));

	return ret;
}


Eigen::Vector3d changeToRNNPos(Eigen::Vector3d pos){
	Eigen::Vector3d ret;
	ret[0] = pos[2]*100;
	ret[1] = (pos[1]-ROOT_HEIGHT_OFFSET)*100;
	ret[2] = -pos[0]*100;
	return ret;
}

Eigen::Isometry3d getJointTransform(dart::dynamics::SkeletonPtr skel, std::string bodyname){
	return skel->getBodyNode(bodyname)->getParentBodyNode()->getWorldTransform()
		*skel->getBodyNode(bodyname)->getParentJoint()->getTransformFromParentBodyNode();
}

Eigen::Vector4d rootDecomposition(dart::dynamics::SkeletonPtr skel, Eigen::VectorXd positions){
	// DEBUG : decomposition
	Eigen::VectorXd p_save = skel->getPositions();
	skel->setPositions(positions);
	int femur_l_idx = skel->getBodyNode("FemurL")->getParentJoint()->getIndexInSkeleton(0);
	int femur_r_idx = skel->getBodyNode("FemurR")->getParentJoint()->getIndexInSkeleton(0);

	Eigen::Isometry3d femur_l_transform = getJointTransform(skel, "FemurL");
	Eigen::Isometry3d femur_r_transform = getJointTransform(skel, "FemurR");

	Eigen::Vector3d up_vec = Eigen::Vector3d::UnitY();
	Eigen::Vector3d x_vec = femur_l_transform.translation() - femur_r_transform.translation();
	x_vec.normalize();
	Eigen::Vector3d z_vec = x_vec.cross(up_vec);
	z_vec[1] = 0;
	z_vec.normalize();
	double angle = std::atan2(z_vec[0], z_vec[2]);

	skel->setPositions(p_save);

	Eigen::AngleAxisd aa_root(angle, Eigen::Vector3d::UnitY());
	Eigen::AngleAxisd aa_hip(positions.segment<3>(0).norm(), positions.segment<3>(0).normalized());

	Eigen::Vector3d hip_dart = QuaternionToDARTPosition(Eigen::Quaterniond(aa_root).inverse()*Eigen::Quaterniond(aa_hip));
	
	Eigen::Vector4d ret;
	ret << angle, hip_dart;

	return ret;
}

Eigen::VectorXd solveIK(dart::dynamics::SkeletonPtr skel, const std::string& bodyname, const Eigen::Vector3d& delta, const Eigen::Vector3d& offset)
{
	auto bn = skel->getBodyNode(bodyname);
	int foot_l_idx = skel->getBodyNode("FootL")->getParentJoint()->getIndexInSkeleton(0);
	int foot_r_idx = skel->getBodyNode("FootR")->getParentJoint()->getIndexInSkeleton(0);
	int footend_l_idx = skel->getBodyNode("FootEndL")->getParentJoint()->getIndexInSkeleton(0);
	int footend_r_idx = skel->getBodyNode("FootEndR")->getParentJoint()->getIndexInSkeleton(0);
	int femur_l_idx = skel->getBodyNode("FemurL")->getParentJoint()->getIndexInSkeleton(0);
	int femur_r_idx = skel->getBodyNode("FemurR")->getParentJoint()->getIndexInSkeleton(0);
	int tibia_l_idx = skel->getBodyNode("TibiaL")->getParentJoint()->getIndexInSkeleton(0);
	int tibia_r_idx = skel->getBodyNode("TibiaR")->getParentJoint()->getIndexInSkeleton(0);
	Eigen::VectorXd newPose = skel->getPositions();
	Eigen::Vector3d tp = delta;
	for(std::size_t i = 0; i < 1000; ++i)
	{
		Eigen::Vector3d deviation = tp - bn->getTransform()*offset;
		if(deviation.norm() < 0.001)
			break;
		// Eigen::Vector3d localCOM = bn->getCOM(bn);
		dart::math::LinearJacobian jacobian = skel->getLinearJacobian(bn, offset);
		jacobian.block<3,6>(0,0).setZero();
		// jacobian.block<3,3>(0,foot_l_idx).setZero();
		// jacobian.block<3,3>(0,foot_r_idx).setZero();
		jacobian.block<3,3>(0,footend_l_idx).setZero();
		jacobian.block<3,3>(0,footend_r_idx).setZero();
		// jacobian.block<3,2>(0,femur_l_idx+1).setZero();
		// jacobian.block<3,2>(0,femur_r_idx+1).setZero();
		// jacobian.block<3,2>(0,tibia_l_idx+1).setZero();
		// jacobian.block<3,2>(0,tibia_r_idx+1).setZero();

		Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::Matrix3d inv_singular_value;
		
		inv_singular_value.setZero();
		for(int k=0;k<3;k++)
		{
			if(svd.singularValues()[k]==0)
				inv_singular_value(k,k) = 0.0;
			else
				inv_singular_value(k,k) = 1.0/svd.singularValues()[k];
		}


		Eigen::MatrixXd jacobian_inv = svd.matrixV()*inv_singular_value*svd.matrixU().transpose();

		// Eigen::VectorXd gradient = jacobian.colPivHouseholderQr().solve(deviation);
		Eigen::VectorXd gradient = jacobian_inv * deviation;
		double prev_norm = deviation.norm();
		double gamma = 0.5;
		for(int j = 0; j < 24; j++){
			Eigen::VectorXd newDirection = gamma * gradient;
			Eigen::VectorXd np = newPose + newDirection;
			skel->setPositions(np);
			skel->computeForwardKinematics(true, false, false);
			double new_norm = (tp - bn->getTransform()*offset).norm();
			if(new_norm < prev_norm){
				newPose = np;
				break;
			}
			gamma *= 0.5;
		}
	}
	return newPose;
}

Eigen::VectorXd solveMCIK(dart::dynamics::SkeletonPtr skel, const std::vector<std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>>& constraints)
{
	int foot_l_idx = skel->getBodyNode("FootL")->getParentJoint()->getIndexInSkeleton(0);
	int foot_r_idx = skel->getBodyNode("FootR")->getParentJoint()->getIndexInSkeleton(0);
	int footend_l_idx = skel->getBodyNode("FootEndL")->getParentJoint()->getIndexInSkeleton(0);
	int footend_r_idx = skel->getBodyNode("FootEndR")->getParentJoint()->getIndexInSkeleton(0);
	int femur_l_idx = skel->getBodyNode("FemurL")->getParentJoint()->getIndexInSkeleton(0);
	int femur_r_idx = skel->getBodyNode("FemurR")->getParentJoint()->getIndexInSkeleton(0);
	int tibia_l_idx = skel->getBodyNode("TibiaL")->getParentJoint()->getIndexInSkeleton(0);
	int tibia_r_idx = skel->getBodyNode("TibiaR")->getParentJoint()->getIndexInSkeleton(0);

	Eigen::VectorXd newPose = skel->getPositions();
	int num_constraints = constraints.size();

	std::vector<dart::dynamics::BodyNode*> bodynodes(num_constraints);
	std::vector<Eigen::Vector3d> targetposes(num_constraints);
	std::vector<Eigen::Vector3d> offsets(num_constraints);

	for(int i = 0; i < num_constraints; i++){
		bodynodes[i] = skel->getBodyNode(std::get<0>(constraints[i]));
		targetposes[i] = std::get<1>(constraints[i]);
		offsets[i] = std::get<2>(constraints[i]);
	}

	int not_improved = 0;
	for(std::size_t i = 0; i < 100; i++)
	{

		// make deviation vector and jacobian matrix
		Eigen::VectorXd deviation(num_constraints*3);
		for(int j = 0; j < num_constraints; j++){
			deviation.segment<3>(j*3) = targetposes[j] - bodynodes[j]->getTransform()*offsets[j];
		}
		if(deviation.norm() < 0.001)
			break;

		int nDofs = skel->getNumDofs();
		Eigen::MatrixXd jacobian_concatenated(3*num_constraints, nDofs);
		for(int j = 0; j < num_constraints; j++){
			dart::math::LinearJacobian jacobian = skel->getLinearJacobian(bodynodes[j], offsets[j]);
			jacobian.block<3,6>(0,0).setZero();
			// jacobian.block<3,3>(0,foot_l_idx).setZero();
			// jacobian.block<3,3>(0,foot_r_idx).setZero();
			jacobian.block<3,3>(0,footend_l_idx).setZero();
			jacobian.block<3,3>(0,footend_r_idx).setZero();
			// jacobian.block<3,2>(0,femur_l_idx+1).setZero();
			// jacobian.block<3,2>(0,femur_r_idx+1).setZero();
			jacobian.block<3,2>(0,tibia_l_idx+1).setZero();
			jacobian.block<3,2>(0,tibia_r_idx+1).setZero();

			jacobian_concatenated.block(3*j, 0, 3, nDofs) = jacobian;
		}
		// std::cout << jacobian_concatenated << std::endl;

		Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian_concatenated, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::MatrixXd inv_singular_value(3*num_constraints, 3*num_constraints);
		
		inv_singular_value.setZero();
		for(int k=0;k<3*num_constraints;k++)
		{
			if(svd.singularValues()[k]<1e-8)
				inv_singular_value(k,k) = 0.0;
			else
				inv_singular_value(k,k) = 1.0/svd.singularValues()[k];
		}


		Eigen::MatrixXd jacobian_inv = svd.matrixV()*inv_singular_value*svd.matrixU().transpose();
		// std::cout << svd.singularValues().transpose() << std::endl;
		// std::cout << svd.matrixV().size() << std::endl;

		// std::cout << jacobian_inv << std::endl;
		// exit(0);
		// Eigen::VectorXd gradient = jacobian.colPivHouseholderQr().solve(deviation);
		Eigen::VectorXd gradient = jacobian_inv * deviation;
		double prev_norm = deviation.norm();
		double gamma = 0.5;
		not_improved++;
		for(int j = 0; j < 24; j++){
			Eigen::VectorXd newDirection = gamma * gradient;
			Eigen::VectorXd np = newPose + newDirection;
			skel->setPositions(np);
			skel->computeForwardKinematics(true, false, false);

			Eigen::VectorXd new_deviation(num_constraints*3);
			for(int j = 0; j < num_constraints; j++){
				new_deviation.segment<3>(j*3) = targetposes[j] - bodynodes[j]->getTransform()*offsets[j];
			}
			double new_norm = new_deviation.norm();
			if(new_norm < prev_norm){
				newPose = np;
				not_improved = 0;
				break;
			}
			gamma *= 0.5;
		}
		if(not_improved > 1){
			break;
		}
	}
	return newPose;
}

}
