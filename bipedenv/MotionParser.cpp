#include<vector>
#include<iostream>
#include <dart/dart.hpp>
#include <dart/gui/gui.hpp>
#include <dart/utils/utils.hpp>
#include<string>
#include "MotionParser.h"

namespace myBVH{

static myBVH::BVHNode* get_end(std::ifstream &file){
	BVHNode* current_node = new BVHNode();
	std::string tmp;
	file >> current_node->name;
	file >> tmp; assert(tmp == "{");
	file >> tmp; assert(tmp == "OFFSET");
	for(int i = 0; i < 3; i++) file >> current_node->offset[i];
	file >> tmp; assert(tmp == "}");

	return current_node;
}

static int total_channels(BVHNode* current){
	int cnt = current->channels.size();
	for(BVHNode* next : current->child){
		cnt += total_channels(next);
	}
	return cnt;
}

static BVHNode* get_node(std::ifstream &file)
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
			for(int i = 0; i < sz; i++){
				file >> tmp;
				current_node->channels.push_back(tmp);
			}
		}
		else if(command == "JOINT"){
			current_node->child.push_back(get_node(file));
		}
		else if(command == "End"){
			current_node->child.push_back(get_end(file));
		}
		else if(command == "}") break;
	}
	return current_node;
}

static void get_motion(std::vector<Eigen::VectorXd> &motion, std::ifstream &file, double &fps, BVHNode* root)
{
	int size = 0, channels = total_channels(root);
	double time_interval;
	std::string command;

	file >> command; assert(command == "Frames:");
	file >> size;
	file >> command; assert(command == "Frame");
	file >> command; assert(command == "Time:");
	file >> time_interval; fps = 1. / time_interval;
	const double PI = acos(-1);

	for(int t = 0; t < size; t++){
		// Position, rotation
		Eigen::VectorXd vec = Eigen::VectorXd::Zero(channels);
		for(int i = 0; i < 6; i++) file >> vec[i];
		// EulerZXY
		for(int i = 6; i < channels; i += 3){
			Eigen::Vector3d angle, joint;
			for(int j = 0; j < 3; j++){
				file >> angle[j];
				angle[j] *= PI / 180;
			}
			Eigen::Matrix3d mat = dart::math::eulerZXYToMatrix(angle);
			joint = dart::dynamics::BallJoint::convertToPositions(mat);
			for(int j = 0, k = i; j < 3; j++, k++) vec[k] = joint[j];
		}
		motion.push_back(vec);
	}
}

void MotionParser(std::vector<Eigen::VectorXd> &motion, std::string filename, double &fps, BVHNode* &root)
{
	std::ifstream file(filename);
	motion.clear();
	
	while(1){
		std::string command, tmp;
		file >> command;
		if(command == "HIERARCHY"){
			file >> tmp;
			assert(tmp == "ROOT");
			root = get_node(file);
		}
		else if(command == "MOTION") get_motion(motion, file, fps, root);
		else break;
	}
	file.close();
}
}
