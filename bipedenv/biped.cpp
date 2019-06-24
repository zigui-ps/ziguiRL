#include<iostream>
#include <dart/dart.hpp>
#include <dart/gui/gui.hpp>
#include <GL/freeglut.h>
#include<vector>
#include "SkeletonBuilder.h"
#include "MotionParser.h"

using namespace dart::dynamics;
using namespace dart::simulation;
using namespace dart::math;

const int TIME_LIMIT = 1024;

const int DOFS_SIZE = 200;
const int FORCES_SIZE = 200;

struct State{
	double dofs[DOFS_SIZE * 2];
};

struct Action{
	double angle[FORCES_SIZE];
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
		mAvgForce = Eigen::VectorXd::Zero(nDofs);
		count = 0;

    mKp = Eigen::MatrixXd::Identity(nDofs, nDofs);
    mKd = Eigen::MatrixXd::Identity(nDofs, nDofs);

    for (std::size_t i = 0; i < 6; ++i)
    {
      mKp(i, i) = 0.0;
      mKd(i, i) = 0.0;
    }

    for (std::size_t i = 6; i < biped->getNumDofs(); ++i)
    {
      mKp(i, i) = 1000;
      mKd(i, i) = 50;
    }

    setTargetPositions(mBiped->getPositions());
  }

  /// Reset the desired dof position to the current position
  void setTargetPositions(const Eigen::VectorXd& pose)
  {
    mTargetPositions = pose;
  }

  /// Clear commanding forces
  void clearForces()
  {
    mForces.setZero();
  }

  /// Add commanind forces from Stable-PD controllers
  void addSPDForces()
  {
    // Lesson 3
    Eigen::VectorXd q = mBiped->getPositions();
    Eigen::VectorXd dq = mBiped->getVelocities();

    Eigen::MatrixXd invM
        = (mBiped->getMassMatrix() + mKd * mBiped->getTimeStep()).inverse();
    Eigen::VectorXd p
        = -mKp * (q + dq * mBiped->getTimeStep() - mTargetPositions);
    Eigen::VectorXd d = -mKd * dq;
    Eigen::VectorXd qddot = invM
                            * (-mBiped->getCoriolisAndGravityForces() + p + d
                               + mBiped->getConstraintForces());

    mForces += p + d - mKd * qddot * mBiped->getTimeStep();
    mBiped->setForces(mForces);
		mAvgForce = (mAvgForce * count + mForces) / (count+1);
  }

protected:
  /// The biped Skeleton that we will be controlling
  SkeletonPtr mBiped;

  /// Joint forces for the biped (output of the Controller)
  Eigen::VectorXd mForces;
	Eigen::VectorXd mAvgForce;
	int count;

  /// Control gains for the proportional error terms in the PD controller
  Eigen::MatrixXd mKp;

  /// Control gains for the derivative error terms in the PD controller
  Eigen::MatrixXd mKd;

  /// Target positions for the PD controllers
  Eigen::VectorXd mTargetPositions;
};

namespace BipedEnv{
	std::vector<Eigen::VectorXd> motion;

	double pose_distance(const Eigen::VectorXd &prev_state, const Eigen::VectorXd &current_state, const Eigen::VectorXd &prev_motion, const Eigen::VectorXd &current_motion, int nDofs){
		double reward = 0;

		// Region Effect
		double t2 = 0, t1 = 0, t0 = 0;
		for(int i = 6; i < nDofs; i++){
			t2 += (prev_motion[i] - current_motion[i]) * (prev_motion[i] - current_motion[i]);
			t1 += (prev_motion[i] - current_motion[i]) * (current_motion[i] - current_state[i]);
			t0 += (current_motion[i] - current_state[i]) * (current_motion[i] - current_state[i]);
		}
		double t = -t1/t2, dist = 0, r1 = 0;
		if(t < 0) dist = t0;
		else if(t > 1) dist = t2 + 2*t1 + t0;
		else dist = t2 * t*t2 + 2*t1*t + t0;

		return dist;
	}

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

	double get_reward(const Eigen::VectorXd &prev_state, const Eigen::VectorXd &current_state, int nDofs){
		int idx = closest_pose(prev_state, current_state, nDofs);
		Eigen::VectorXd prev_motion = motion[idx], current_motion = motion[idx+1];
		double dist = pose_distance(prev_state, current_state, prev_motion, current_motion, nDofs);
		double flow = flow_effect(prev_state, current_state, prev_motion, current_motion, nDofs);

		double reward = 0;
		double r1 = 1. / (pow(dist / 10, 4) + 0.1);
		double r2 = exp(-flow);	

		reward += 0.3 * r1 + 0.7 * r2;

		if(dist > 15) reward = -1;
		// TODO : End point control
		return reward;
	}
}

class MyWindow : public dart::gui::glut::SimWindow
{
public:
  /// Constructor
  MyWindow(WorldPtr world)
  {
		initial_world = world;
		
    setWorld(initial_world->clone());
    
    // Find the Skeleton named "pendulum" within the World
		biped = mWorld->getSkeleton("Humanoid"); // TODO name

    // Make sure that the pendulum was found in the World
    assert(biped != nullptr);
		
    mController = std::make_unique<Controller>(biped);
		
		state_size = (biped->getNumDofs()) * 2;
		action_size = biped->getNumDofs() - 6;

		step = 0;
		done = false;
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

	double applyAction(Action action)
	{
		double reward = 0;
		step += 1;
		int nDofs = biped->getNumDofs();
		Eigen::VectorXd act = biped->getPositions();
		Eigen::VectorXd prev_state = biped->getPositions();
		
		for (std::size_t i = 6, j = 0; i < nDofs; i++, j++)
			act[i] += action.angle[j] * 0.2;
		
		mController->setTargetPositions(act);
		
		for(int i = 0; i < 20; i++){
			timeStepping();
			mController->clearForces();
			mController->addSPDForces();
		}
		
		Eigen::VectorXd current_state = biped->getPositions();
		for(size_t i = 0, j = 0; i < biped->getNumDofs(); i++, j++){
			state.dofs[j*2] = state.dofs[j*2+1];
			state.dofs[j*2+1] = current_state[i];
		}
		reward = BipedEnv::get_reward(prev_state, current_state, nDofs);
		if(biped->getCOM()[1] < -0.6 || biped->getCOM()[1] > 1.5) done = true, reward = 0;
		if(step >= TIME_LIMIT) done = true;
//		printf("%lf ", reward);
		if(reward < 0.0) done = true;
		return reward;
	}

	State getState(){ 
		return state;
	}

	void resetWorld(){
		step = 0;
		done = false;
		setWorld(initial_world->clone());
		
		biped = mWorld->getSkeleton("Humanoid"); // TODO name
    mController = std::make_unique<Controller>(biped);
		
		Eigen::VectorXd act = biped->getPositions();
		Eigen::VectorXd sample = BipedEnv::motion[rand() % BipedEnv::motion.size()];
		for(int i = 6; i < biped->getNumDofs(); i++){
			act[i] = sample[i];
		}

		biped->setPositions(act);
		for(size_t i = 0, j = 0; i < biped->getNumDofs(); i++, j++){
			state.dofs[j*2] = state.dofs[j*2+1] = act[i];
		}
	}

	bool done;
	int action_size, state_size;
	bool tl(){ return step >= TIME_LIMIT; }

protected:
	int step;
	State state;
	WorldPtr initial_world;
	SkeletonPtr biped;
  std::unique_ptr<Controller> mController;
};

std::shared_ptr<MyWindow> window;

SkeletonPtr createFloor()
{
  SkeletonPtr floor = Skeleton::create("floor");

  // Give the floor a body
  BodyNodePtr body
      = floor->createJointAndBodyNodePair<WeldJoint>(nullptr).second;

  // Give the body a shape
  double floor_width = 10.0;
  double floor_height = 0.01;
  std::shared_ptr<BoxShape> box(
      new BoxShape(Eigen::Vector3d(floor_width, floor_height, floor_width)));
  auto shapeNode = body->createShapeNodeWith<
      VisualAspect,
      CollisionAspect,
      DynamicsAspect>(box);
  shapeNode->getVisualAspect()->setColor(dart::Color::Black());

  // Put the body into position
  Eigen::Isometry3d tf(Eigen::Isometry3d::Identity());
  tf.translation() = Eigen::Vector3d(0.0, -1.0, 0.0);
  body->getParentJoint()->setTransformFromParentBodyNode(tf);

  return floor;
}

void init(int argc, char* argv[])
{
	printf("Skeleton File: %s\n", argv[2]);
  SkeletonPtr biped = DPhy::SkeletonBuilder::BuildFromFile(argv[2]);
  SkeletonPtr ground = DPhy::SkeletonBuilder::BuildFromFile("./character/ground.xml");
	
	printf("Motion File: %s\n", argv[3]);
	double fps;
	myBVH::BVHNode *root;
	myBVH::MotionParser(BipedEnv::motion, argv[3], fps, root);

	// Set joint limits
  for (std::size_t i = 6; i < biped->getNumJoints(); ++i)
    biped->getJoint(i)->setPositionLimitEnforced(true);

  // Enable self collision check but ignore adjacent bodies
  biped->enableSelfCollisionCheck();
  biped->disableAdjacentBodyCheck();

  // Create a world and add the pendulum to the world
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3d(0.0, -9.81, 0.0));
  world->addSkeleton(biped);
  world->addSkeleton(ground);
	world->setTimeStep(1.0/600);

  // Create a window for rendering the world and handling user input
  window = std::shared_ptr<MyWindow>(new MyWindow(world));

  // Print instructions
	std::cout << "init Called!!" << std::endl;

  // Initialize glut, initialize the window, and begin the glut event loop
  glutInit(&argc, argv);
  window->initWindow(640, 480, "Biped Environment");
}

void render(){
	glutMainLoopEvent();
}

State reset()
{
	window->resetWorld();
	return window->getState();
}

Result step(Action action)
{
	double reward = window->applyAction(action);
	return Result(window->getState(), reward, window->done, window->tl());
}

int observation_size()
{
	return window->state_size;
}

int action_size()
{
	return window->action_size;
}
