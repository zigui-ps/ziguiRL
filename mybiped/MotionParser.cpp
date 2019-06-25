#include<vector>
#include<iostream>
#include <dart/dart.hpp>
#include <dart/gui/gui.hpp>
#include <dart/utils/utils.hpp>
#include<string>
#include "MotionParser.h"
#include "Functions.h"

const double PI = acos(-1);

myBVH::myBVH(std::string filename, SkeletonPtr skel_ori){
	BVHParser(filename);
	skel = skel_ori->cloneSkeleton();

	assert(skel_ori->getName() == "Humanoid");
	assert(motion.size() != 0);
}

Eigen::VectorXd myBVH::getPositions(int count){
	if(count >= frames){
		std::cout << "ReferenceManager.cpp : count exceeds frame limit" << std::endl;
		std::cout << "count : " << count << ", " << "tf : " << frames << std::endl;
		count = motion.size()-1;
	}
	return motion[count];
}

double myBVH::closest_pose(Eigen::VectorXd &position){
	int nDofs = skel->getNumDofs();
	double mn = 1e18, idx = 0;
	
	for(int i = 0; i+2 < motion.size(); i++){
		auto prev_motion = motion[i];
		auto current_motion = motion[i+1];
		// Region Effect
		double t2 = 0, t1 = 0, t0 = 0;
		for(int i = 6; i < nDofs; i++){
			t2 += (prev_motion[i] - current_motion[i]) * (prev_motion[i] - current_motion[i]);
			t1 += (prev_motion[i] - current_motion[i]) * (current_motion[i] - position[i]);
			t0 += (current_motion[i] - position[i]) * (current_motion[i] - position[i]);
		}
		double t = -t1/t2, dist = 0, tmp = 0;
		if(t < 0) dist = t0, tmp = 0;
		else if(t > 1) dist = t2 + 2*t1 + t0, tmp = 1;
		else dist = t2 * t*t2 + 2*t1*t + t0, tmp = t;
		if(dist < mn) mn = dist, idx = tmp;
	}
	return idx;
}

//*
Eigen::VectorXd myBVH::getPositions(double count){
	int k = (int)std::floor(count);
	int k1 = k+1;

	Eigen::VectorXd motion_k = getPositions(k);
	Eigen::VectorXd motion_k1 = getPositions(k1);

	Eigen::VectorXd motion_t = Eigen::VectorXd::Zero(motion_k.rows());

	double dt = count - k;

	for(int i = 0; i < skel->getNumJoints(); i++){
		dart::dynamics::Joint* jn = skel->getJoint(i);
		if(dynamic_cast<dart::dynamics::BallJoint*>(jn)!=nullptr){
			int index = jn->getIndexInSkeleton(0);
			int dof = jn->getNumDofs();
			Eigen::Quaterniond pos_k = DPhy::DARTPositionToQuaternion(motion_k.segment<3>(index));
			Eigen::Quaterniond pos_k1 = DPhy::DARTPositionToQuaternion(motion_k1.segment<3>(index));

			motion_t.segment<3>(index) = DPhy::QuaternionToDARTPosition(pos_k.slerp(dt, pos_k1));
		}
		else if(dynamic_cast<dart::dynamics::FreeJoint*>(jn)!=nullptr){
			int index = jn->getIndexInSkeleton(0);
			int dof = jn->getNumDofs();
			Eigen::Quaterniond pos_k = DPhy::DARTPositionToQuaternion(motion_k.segment<3>(index));
			Eigen::Quaterniond pos_k1 = DPhy::DARTPositionToQuaternion(motion_k1.segment<3>(index));

			motion_t.segment<3>(index) = DPhy::QuaternionToDARTPosition(pos_k.slerp(dt, pos_k1));

			motion_t.segment<3>(index+3) = motion_k.segment<3>(index+3)*(1-dt) + motion_k1.segment<3>(index+3)*dt;
		}
		else if(dynamic_cast<dart::dynamics::RevoluteJoint*>(jn)!=nullptr){
			int index = jn->getIndexInSkeleton(0);
			int dof = jn->getNumDofs();
			double delta = DPhy::RadianClamp(motion_k1[index]-motion_k[index]);
			motion_t[index] = motion_k[index] + delta*dt;
		}
	}

	return motion_t;
}
// */

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> myBVH::getMotion(int count){
	Eigen::VectorXd cur_pos, cur_vel, next_pos;
	if( count >= frames-1){
		std::cout << "Exceed reference max time" << std::endl;
		cur_pos = getPositions(count);
		next_pos = getPositions(count);
		cur_vel = Eigen::VectorXd::Zero(cur_pos.rows());
	}
	else{
		int next_count = count + 1;
		cur_pos = getPositions(count);
		next_pos = getPositions(next_count);
		cur_vel = skel->getPositionDifferences(next_pos, cur_pos) * fps;
	}
	return std::make_tuple(cur_pos, cur_vel, next_pos);
}

//*
std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> myBVH::getMotion(double count)
{
	Eigen::VectorXd cur_pos, cur_vel, next_pos;
	if(count >= frames){
		std::cout << "Exceed reference max time" << std::endl;
		Eigen::VectorXd cur_pos = getPositions(count);
		Eigen::VectorXd next_pos = getPositions(count);
		Eigen::VectorXd cur_vel = Eigen::VectorXd::Zero(cur_pos.rows());
	}

	else{
		Eigen::VectorXd cur_pos = getPositions(count);
		Eigen::VectorXd next_pos = getPositions(count + 1);
		Eigen::VectorXd cur_vel = skel->getPositionDifferences(next_pos, cur_pos) * fps;

		return std::make_tuple(cur_pos, cur_vel, next_pos);
	}
}
// */

void myBVH::BVHParser(std::string filename)
{
	std::ifstream file(filename);
	std::string command, tmp;

	motion.clear();
	channels = frames = 0;

	while(1){
		command = ""; file >> command;
		if(command == "HIERARCHY"){
			file >> tmp; assert(tmp == "ROOT");
			root = nodeParser(file);
		}
		else if(command == "MOTION") motionParser(file);
		else break;
	}
	file.close();
}

BVHNode* myBVH::endParser(std::ifstream &file){
	BVHNode* current_node = new BVHNode();
	std::string tmp;
	file >> current_node->name;
	file >> tmp; assert(tmp == "{");
	file >> tmp; assert(tmp == "OFFSET");
	for(int i = 0; i < 3; i++) file >> current_node->offset[i];
	file >> tmp; assert(tmp == "}");

	return current_node;
}

BVHNode* myBVH::nodeParser(std::ifstream &file)
{
	std::string command, tmp;
	int sz;
	BVHNode* current_node = new BVHNode();

	file >> current_node->name;
	file >> tmp; assert(tmp == "{");

	while(1){
		file >> command;
		if(command == "OFFSET"){
			for(int i = 0; i < 3; i++) file >> current_node->offset[i];
		}
		else if(command == "CHANNELS"){
			file >> sz;
			channels += sz;
			for(int i = 0; i < sz; i++){
				file >> tmp;
				current_node->channelList.push_back(tmp);
			}
		}
		else if(command == "JOINT"){
			current_node->child.push_back(nodeParser(file));
		}
		else if(command == "End"){
			current_node->child.push_back(endParser(file));
		}
		else if(command == "}") break;
	}
	return current_node;
}

void myBVH::motionParser(std::ifstream &file)
{
	double time_interval;
	std::string command;

	file >> command; assert(command == "Frames:");
	file >> frames;
	file >> command; assert(command == "Frame");
	file >> command; assert(command == "Time:");
	file >> time_interval; fps = 1. / time_interval;
	printf("%lf\n", fps);

	for(int t = 0; t < frames; t++){
		// Position
		Eigen::VectorXd vec = Eigen::VectorXd::Zero(channels);
		for(int i = 0; i < 3; i++){
			file >> vec[i+3];
			vec[i+3] /= 100;
		}
		// EulerZXY
		for(int i = 3; i < channels; i += 3){
			Eigen::Vector3d angle, joint;
			for(int j = 0; j < 3; j++){
				file >> angle[j];
				angle[j] *= PI / 180;
			}
			Eigen::Matrix3d mat = dart::math::eulerZXYToMatrix(angle);
			joint = dart::dynamics::BallJoint::convertToPositions(mat);
			for(int j = 0, k = i == 3? 0 : i; j < 3; j++, k++) vec[k] = joint[j];
		}
		motion.push_back(vec);
	}
}
