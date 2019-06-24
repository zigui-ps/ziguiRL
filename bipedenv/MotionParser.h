#ifndef __MYBVH_MOTION_SKELETON_BUILDER_H__
#define __MYBVH_MOTION_SKELETON_BUILDER_H__
#include "dart/dart.hpp"
#include<vector>
#include<iostream>
#include<string>

using namespace dart::dynamics;

struct BVHNode{
	std::string name;
	double offset[3];
	std::vector<std::string> channelList;
	std::vector<BVHNode*> child;
};

class myBVH{
	public:
		myBVH(std::string filename, SkeletonPtr skel_ori);
		Eigen::VectorXd getPositions(int count);
//		Eigen::VectorXd getPositions(double count);
		std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> getMotion(int count);
//		std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> getMotion(double count);
		
		double fps;
		int channels, frames;

	protected:
		void BVHParser(std::string filename);
		BVHNode* endParser(std::ifstream &file);
		BVHNode* nodeParser(std::ifstream &file);
		void motionParser(std::ifstream &file);

		std::vector<Eigen::VectorXd> motion;
		BVHNode *root;
		SkeletonPtr skel;
};

#endif
