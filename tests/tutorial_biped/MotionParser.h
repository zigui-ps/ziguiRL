#ifndef __MYBVH_MOTION_SKELETON_BUILDER_H__
#define __MYBVH_MOTION_SKELETON_BUILDER_H__
#include "dart/dart.hpp"
#include<vector>
#include<iostream>
#include<string>

namespace myBVH{
struct BVHNode{
	std::string name;
	double offset[3];
	std::vector<std::string> channels;
	std::vector<BVHNode*> child;
};

void MotionParser(std::vector<Eigen::VectorXd> &motion, std::string filename, double &fps, BVHNode* &root);
}

#endif
