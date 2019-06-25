#include<iostream>
#include <dart/dart.hpp>
#include <dart/gui/gui.hpp>
#include <GL/freeglut.h>
#include<vector>
#include "CharacterConfiguration.h"
#include "SkeletonBuilder.h"
#include "MotionParser.h"
#include "Functions.h"

using namespace dart::dynamics;
using namespace dart::simulation;
using namespace dart::math;

const int TIME_LIMIT = 1024;

const int DOFS_SIZE = 200;

struct State{
	double array[DOFS_SIZE * 2 + 1];
};

struct Action{
	double desired[DOFS_SIZE * 2 + 1];
};

struct Result{
	Result(State s, double r, bool done, bool tl):
		state(s), reward(r), done(done), tl(tl){}
	State state;
	double reward;
	bool done, tl;
};

class Controller
{
public:
  /// Constructor
  Controller(const SkeletonPtr& biped)
    : mBiped(biped)
  {
    int nDofs = mBiped->getNumDofs();

    mForces = Eigen::VectorXd::Zero(nDofs);

    mKp = Eigen::VectorXd::Zero(nDofs);
    mKv = Eigen::VectorXd::Zero(nDofs);

    for (std::size_t i = 0; i < 6; ++i) mKp[i] = mKv[i] = 0;
    for (std::size_t i = 6; i < biped->getNumDofs(); ++i) mKp[i] = 500, mKv[i] = 50;

    setTargetPositions(mBiped->getPositions(), mBiped->getVelocities());
  }

  /// Reset the desired dof position to the current position
  void setTargetPositions(const Eigen::VectorXd& pose, const Eigen::VectorXd& vel)
  {
    p_desired = pose;
		v_desired = vel;
  }

  /// Clear commanding forces
  void clearForces()
  {
    mForces.setZero();
  }

  /// Add commanind forces from Stable-PD controllers
  void addSPDForces()
	{
		auto& skel = mBiped;

		Eigen::VectorXd q = skel->getPositions();
		Eigen::VectorXd dq = skel->getVelocities();
		double dt = skel->getTimeStep();
		Eigen::MatrixXd M_inv = (skel->getMassMatrix() + Eigen::MatrixXd(dt*mKv.asDiagonal())).inverse();

		// Eigen::VectorXd p_d = q + dq*dt - p_desired;
		Eigen::VectorXd p_d(q.rows());
		// clamping radians to [-pi, pi], only for ball joints
		// TODO : make it for all type joints
		p_d.segment<6>(0) = Eigen::VectorXd::Zero(6);
		for(int i = 6; i < skel->getNumDofs(); i+=3){
			Eigen::Quaterniond q_s = DPhy::DARTPositionToQuaternion(q.segment<3>(i));
			Eigen::Quaterniond dq_s = DPhy::DARTPositionToQuaternion(dt*(dq.segment<3>(i)));
			Eigen::Quaterniond q_d_s = DPhy::DARTPositionToQuaternion(p_desired.segment<3>(i));

			Eigen::Quaterniond p_d_s = q_d_s.inverse()*q_s*dq_s;

			Eigen::Vector3d v = DPhy::QuaternionToDARTPosition(p_d_s);
			double angle = v.norm();
			if(angle > 1e-8){
				Eigen::Vector3d axis = v.normalized();

				angle = DPhy::RadianClamp(angle);	
				p_d.segment<3>(i) = angle * axis;
			}
			else
				p_d.segment<3>(i) = v;
		}
		Eigen::VectorXd p_diff = -mKp.cwiseProduct(p_d);
		Eigen::VectorXd v_diff = -mKv.cwiseProduct(dq-v_desired);
		Eigen::VectorXd qddot = M_inv*(-skel->getCoriolisAndGravityForces()+
				p_diff+v_diff+skel->getConstraintForces());

		Eigen::VectorXd tau = p_diff + v_diff - dt*mKv.cwiseProduct(qddot);
		tau.segment<6>(0) = Eigen::VectorXd::Zero(6);

		mForces += tau;

		mBiped->setForces(mForces);
	}

protected:
	/// The biped Skeleton that we will be controlling
	SkeletonPtr mBiped;

	/// Joint forces for the biped (output of the Controller)
	Eigen::VectorXd mForces;

	Eigen::VectorXd mKp, mKv;

	/// Target positions for the PD controllers
	Eigen::VectorXd p_desired;
	Eigen::VectorXd v_desired;
};

namespace BipedEnv{
	/* TODO
		 int closest_pose(const Eigen::VectorXd &prev_state, const Eigen::VectorXd &current_state, int nDofs){
		 double mn = 1e18;
		 int idx = 0;
		 for(int i = 0; i+1 < motion.size(); i++){
		 double dist = pose_distance(prev_state, current_state, motion[i], motion[i+1], nDofs);
		 if(dist < mn) mn = dist, idx = i;
		 }
		 return idx;
		 }

		 double flow_effect(const Eigen::VectorXd &prev_state, const Eigen::VectorXd &current_state, const Eigen::VectorXd &prev_motion, const Eigen::VectorXd &current_motion, int nDofs){
		 double vec = 0, s1 = 0, s2 = 0;
		 for(int i = 6; i < nDofs; i++){
		 double ds = current_state[i] - prev_state[i];
		 double dm = current_motion[i] - prev_motion[i];
		 vec += (ds-dm) * (ds-dm);
		 }
		 return vec;
		 }
	// */

	struct State{
		Eigen::VectorXd positions;
		Eigen::VectorXd velocities;
		State(Eigen::VectorXd p, Eigen::VectorXd v):
			positions(p), velocities(v){}
	};

	std::vector<std::string> mInterestBodies = {
		"Spine", "Neck", "Head", 
		"ForeArmL", "ArmL", "HandL",
		"ForeArmR", "ArmR", "HandR",
		"FemurL", "TibiaL", "FootL",
		"FemurR", "TibiaR", "FootR"
	};

	std::vector<std::string> mRewardBodies = {
		"Spine", "Neck", "Head", "Torso",
		"ForeArmL", "ArmL", "HandL",
		"ForeArmR", "ArmR", "HandR",
		"FemurL", "TibiaL", "FootL",
		"FemurR", "TibiaR", "FootR"
	};
	
	std::vector<std::string> mEndEffectors = {
		"HandL", "HandR", "FootL", "FootR"
	};

	std::shared_ptr<myBVH> targetData;
	SkeletonPtr clone;
	int torso_idx;

	void init(const char* filename, SkeletonPtr skel){
		targetData = std::shared_ptr<myBVH>(new myBVH(filename, skel));
		torso_idx = skel->getBodyNode("Torso")->getParentJoint()->getIndexInSkeleton(0);
		clone = skel->cloneSkeleton();
	}

	bool isTerminateState(SkeletonPtr skel, SkeletonPtr target)
	{
		bool mIsNanAtTerminal = false, mIsTerminal = false;
		int terminationReason = 0;

		Eigen::VectorXd p = skel->getPositions();
		Eigen::VectorXd v = skel->getVelocities();
		Eigen::Vector3d root_pos = p.segment<3>(3);
		Eigen::Isometry3d cur_root_inv = skel->getRootBodyNode()->getWorldTransform().inverse();

		double root_y = skel->getBodyNode(0)->getTransform().translation()[1];
		Eigen::Vector3d root_v = skel->getBodyNode(0)->getCOMLinearVelocity();
		double root_v_norm = root_v.norm();

		Eigen::Vector3d root_pos_diff = target->getPositions().segment<3>(3) - root_pos;
		Eigen::Isometry3d root_diff = cur_root_inv * target->getRootBodyNode()->getWorldTransform();

		Eigen::AngleAxisd root_diff_aa(root_diff.linear());
		double angle = DPhy::RadianClamp(root_diff_aa.angle());

		// check nan
		if(dart::math::isNan(p)){
			mIsNanAtTerminal = true;
			mIsTerminal = true;
			terminationReason = 3;
			return mIsTerminal;
		}
		if(dart::math::isNan(v)){
			mIsNanAtTerminal = true;
			mIsTerminal = true;
			terminationReason = 4;
			return mIsTerminal;
		}
		//ET
		if(root_y<TERMINAL_ROOT_HEIGHT_LOWER_LIMIT || root_y > TERMINAL_ROOT_HEIGHT_UPPER_LIMIT){
			mIsNanAtTerminal = false;
			mIsTerminal = true;
			terminationReason = 1;
		}
		if(std::abs(root_pos[0]) > 4990){
			mIsNanAtTerminal = false;
			mIsTerminal = true;
			terminationReason = 9;
		}
		if(std::abs(root_pos[2]) > 4990){
			mIsNanAtTerminal = false;
			mIsTerminal = true;
			terminationReason = 9;
		}
		if(root_pos_diff.norm() > TERMINAL_ROOT_DIFF_THRESHOLD){
			mIsNanAtTerminal = false;
			mIsTerminal = true;
			terminationReason = 2;
		}
		if(std::abs(angle) > TERMINAL_ROOT_DIFF_ANGLE_THRESHOLD){
			mIsNanAtTerminal = false;
			mIsTerminal = true;
			terminationReason = 5;
		}
		return mIsTerminal;
	}

	double flow_effect(const Eigen::VectorXd &prev_state, const Eigen::VectorXd &current_state, const Eigen::VectorXd &prev_motion, const Eigen::VectorXd &current_motion, int nDofs){
		double vec = 0, s1 = 0, s2 = 0;
		for(int i = 6; i < nDofs; i++){
			double ds = current_state[i] - prev_state[i];
			double dm = current_motion[i] - prev_motion[i];
			vec += (ds-dm) * (ds-dm);
		}
		return vec;
	}

	double pose_reward(const SkeletonPtr skel, const State &target){
		assert(skel->getName() == "Humanoid");

		clone->setPositions(target.positions);
		clone->setVelocities(target.velocities);
		clone->computeForwardKinematics(true,true,false);

		if(isTerminateState(skel, clone)) return -1.0;

		//Position, Velocity Differences
		Eigen::VectorXd p_diff = skel->getPositionDifferences(skel->getPositions(), target.positions);
		Eigen::VectorXd v_diff = skel->getVelocityDifferences(skel->getVelocities(), target.velocities);

		Eigen::VectorXd p_diff_lower, v_diff_lower;
		p_diff_lower.resize(mRewardBodies.size()*3);
		v_diff_lower.resize(mRewardBodies.size()*3);

		for(int i = 0; i < mRewardBodies.size(); i++){
			int idx = skel->getBodyNode(mRewardBodies[i])->getParentJoint()->getIndexInSkeleton(0);
			p_diff_lower.segment<3>(3*i) = p_diff.segment<3>(idx);
			v_diff_lower.segment<3>(3*i) = v_diff.segment<3>(idx);
		}

		//End-effector position and COM Differences
		Eigen::VectorXd ee_diff(mEndEffectors.size()*3);
		Eigen::Vector3d com_diff;

		com_diff = skel->getCOM() - clone->getCOM();

		for(int i = 0; i < mEndEffectors.size(); i++){
			Eigen::Isometry3d diff = 
				skel->getBodyNode(mEndEffectors[i])->getWorldTransform().inverse() * \
				clone->getBodyNode(mEndEffectors[i])->getWorldTransform();
			ee_diff.segment<3>(3*i) = diff.translation();
		}

		double scale = 1.0;

		double sig_p = 0.1 * scale; 		// 2
		double sig_v = 1.0 * scale;		// 3
		double sig_com = 0.3 * scale;		// 4
		double sig_ee = 0.3 * scale;		// 8

		double r_p = DPhy::exp_of_squared(p_diff_lower,sig_p);
		double r_v = DPhy::exp_of_squared(v_diff_lower,sig_v);
		double r_com = DPhy::exp_of_squared(com_diff,sig_com);
		double r_ee = DPhy::exp_of_squared(ee_diff,sig_ee);

		//		printf("%g %g %g %g\n", r_p, r_v, r_com, r_ee);

		double r_tot = r_p*r_v*r_com*r_ee;

		return r_tot;
	}

	State getTargetState(const int target_frame){
		auto tmp = targetData->getMotion(target_frame);
		return State(std::get<0>(tmp), std::get<1>(tmp));
	}

	double get_reward(SkeletonPtr skel, Eigen::VectorXd prev_position){
		/*
			 Eigen::VectorXd prev_motion = motion[idx], current_motion = motion[idx+1];
			 double dist = pose_distance(prev_state, current_state, prev_motion, current_motion, nDofs);
			 double flow = flow_effect(prev_state, current_state, prev_motion, current_motion, nDofs);

			 double reward = 0;
			 double r1 = 1. / (pow(dist / 10, 4) + 0.1);
			 double r2 = exp(-flow);	

			 reward += 0.3 * r1 + 0.7 * r2;

			 if(dist > 15) reward = -1;
		// TODO : End point control
		// */

		double idx = targetData->closest_pose(prev_position);
		State target = getTargetState(idx+1);

		double reward = 0;
		reward += pose_reward(skel, target);

		return reward;
	}

	std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> getMotion(int step){
		return targetData->getMotion(step);
	}
	
	Eigen::VectorXd getVelocities(int step){
		return std::get<1>(targetData->getMotion(step));
	}
}

class Agent{
	public:
		Agent(WorldPtr world){
			initial_world = world;
			mWorld = initial_world->clone();

			// Find the Skeleton named "pendulum" within the World
			biped = mWorld->getSkeleton("Humanoid"); // TODO name

			// Make sure that the pendulum was found in the World
			assert(biped != nullptr);

			mController = std::unique_ptr<Controller>(new Controller(biped));

			state_size = (biped->getNumDofs()) * 2;
			action_size = biped->getNumDofs() - 6;

			step = 0;
			done = false;
		}

		void setState(int step, Eigen::VectorXd &position, Eigen::VectorXd &velocity)
		{
			for(size_t i = 0, j = 0; i < biped->getNumDofs(); i++, j++){
				state.array[j*2] = i == 3 || i == 5 ? 0 :  position[i];
				state.array[j*2+1] = velocity[i]; //velocity 
			}
		}

		double applyAction(Action action)
		{
			double reward = 0;
			int nDofs = biped->getNumDofs();
			Eigen::VectorXd act = biped->getPositions();
			Eigen::VectorXd vel = BipedEnv::getVelocities(step+1);
			Eigen::VectorXd prev_position = biped->getPositions();

			for(int i = 0; i < 6; i++) act[i] = vel[i] = 0;
			for (std::size_t i = 6, j = 0; i < nDofs; i++, j++){
				act[i] += action.desired[j];
			}

			mController->setTargetPositions(act, vel);

			step += 1;
			for(int i = 0; i < 20; i++){
				mController->clearForces();
				mController->addSPDForces();
				mWorld->step();
			}

			Eigen::VectorXd curr_position = biped->getPositions();
			Eigen::VectorXd curr_velocity = biped->getVelocities();

			setState(step, curr_position, curr_velocity);

			reward = BipedEnv::get_reward(biped, prev_position);
//			printf("reward: %lf\n", reward);

			if(reward < 0) done = true, reward = 0;
			if(step >= BipedEnv::targetData->frames-2) done = true; //tl = true;

			return reward;
		}

		State getState(){ 
			return state;
		}

		void resetWorld(){
			step = rand() % (BipedEnv::targetData->frames - 2); // ??
			done = tl = false;
			mWorld = initial_world->clone();

			biped = mWorld->getSkeleton("Humanoid");
			mController = std::unique_ptr<Controller>(new Controller(biped));

			auto tmp = BipedEnv::getMotion(step);
			Eigen::VectorXd position = std::get<0>(tmp);
			Eigen::VectorXd velocity = std::get<1>(tmp);

			biped->setPositions(position);
			biped->setVelocities(velocity);

			biped->computeForwardKinematics(true,true,false);

			setState(step, position, velocity);
		}
		bool done, tl;
		int action_size, state_size;
		WorldPtr mWorld, initial_world;

	protected:
		int step;
		State state;
		SkeletonPtr biped;
		std::unique_ptr<Controller> mController;
};

class MyWindow : public dart::gui::glut::SimWindow
{
	public:
		/// Constructor
		MyWindow(std::shared_ptr<Agent> _agent)
		{
			agent = _agent;
			setWorld(agent->mWorld);
		}

		/// Handle keyboard input
		void keyboard(unsigned char key, int x, int y) override
		{
			switch (key)
			{
				default:
					SimWindow::keyboard(key, x, y);
			}
		}

	protected:
		std::shared_ptr<Agent> agent;
};

std::shared_ptr<MyWindow> window;
std::shared_ptr<Agent> agent;
int isRender;

void init(int argc, char* argv[])
{
	const char* humanoidFile = "./character/humanoid_new.xml";
	const char* groundFile = "./character/ground.xml";
	const char* motionFile = "./motion_data/run.bvh";

	printf("Skeleton File: %s\n", humanoidFile);
	SkeletonPtr biped = DPhy::SkeletonBuilder::BuildFromFile(humanoidFile);

	printf("Ground File: %s\n", groundFile);
	SkeletonPtr ground = DPhy::SkeletonBuilder::BuildFromFile(groundFile);

	printf("Motion File: %s\n", motionFile);
	BipedEnv::init(motionFile, biped);

	// Enable self collision check but ignore adjacent bodies
	//	biped->enableSelfCollisionCheck();
	//	biped->disableAdjacentBodyCheck();
	
	for(int i = 0; i < argc; i++) if(strcmp(argv[i], "--render") == 0) isRender = 1;

	// Create a world and add the pendulum to the world
	WorldPtr world = World::create();
	world->setGravity(Eigen::Vector3d(0.0, -9.81, 0.0));
	world->addSkeleton(biped);
	world->addSkeleton(ground);
	world->setTimeStep(1.0/600);

	agent = std::shared_ptr<Agent>(new Agent(world));

	// Create a window for rendering the world and handling user input
	if(isRender) window = std::shared_ptr<MyWindow>(new MyWindow(agent));

	// Print instructions
	std::cout << "init Called!!" << std::endl;

	// Initialize glut, initialize the window, and begin the glut event loop
	if(isRender){
		glutInit(&argc, argv);
		window->initWindow(640, 480, "Biped Environment");
	}
}

void render(){
	if(isRender) glutMainLoopEvent();
}

State reset()
{
	agent->resetWorld();
	if(isRender) window->setWorld(agent->mWorld);
	return agent->getState();
}

Result step(Action action)
{
	double reward = agent->applyAction(action);
	return Result(agent->getState(), reward, agent->done, agent->tl);
}

int observation_size()
{
	return agent->state_size;
}

int action_size()
{
	return agent->action_size;
}
