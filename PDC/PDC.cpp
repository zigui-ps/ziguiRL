#include<iostream>
#include <dart/dart.hpp>
#include <dart/gui/gui.hpp>
#include <GL/freeglut.h>
#include "SkeletonBuilder.h"

using namespace dart::dynamics;
using namespace dart::simulation;

const int TIME_LIMIT = 999;

const int DOFS_SIZE = 200;
const int FORCES_SIZE = 200;

struct State{
	double dofs[DOFS_SIZE * 2];
};

struct Action{
	double forces[FORCES_SIZE];
};

struct Result{
	Result(State s, double r, bool done, bool tl):
		state(s), reward(r), done(done), tl(tl){}
	State state;
	double reward;
	bool done, tl;
};

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
		
		state_size = (biped->getNumDofs() - 6) * 2;
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
		Eigen::VectorXd forces = Eigen::VectorXd::Zero(nDofs);
		
		for (std::size_t i = 6; i < biped->getNumBodyNodes(); ++i)
			act[i] = action.forces[i];

		mBiped->setForces(mForces);

/*
		for(int i = 0; i < mPendulum->getNumBodyNodes(); i++){
			BodyNode *bn = mPendulum->getBodyNode(i);
			auto visualShapeNodes = bn->getShapeNodesWith<VisualAspect>();
			if(visualShapeNodes.size() == 3){
				visualShapeNodes[2]->remove();
			}
			
			ArrowShape::Properties arrow_properties;
			arrow_properties.mRadius = 0.05;
			if(action.force[i] > 0){
				bn->createShapeNodeWith<VisualAspect>(std::shared_ptr<ArrowShape>(new ArrowShape(
								Eigen::Vector3d(-action.force[i] - default_width / 2.0, 0.0, default_height / 2.0),
								Eigen::Vector3d(-default_width / 2.0, 0.0, default_height / 2.0),
								arrow_properties,
								dart::Color::Orange(1.0))));
			}
			else{
				bn->createShapeNodeWith<VisualAspect>(std::shared_ptr<ArrowShape>(new ArrowShape(
								Eigen::Vector3d(-action.force[i] + default_width / 2.0, 0.0, default_height / 2.0),
								Eigen::Vector3d(default_width / 2.0, 0.0, default_height / 2.0),
								arrow_properties,
								dart::Color::Orange(1.0))));
			}
		}
			// */

		for(int i = 0; i < 100; i++) timeStepping();

		for(size_t i = 6; i < biped->getNumDofs(); ++i){
			DegreeOfFreedom* dof = biped->getDof(i);

			state.dofs[i*2] = state.dofs[i*2+1];
			state.dofs[i*2+1] = dof->getPosition();

			reward += fabs(dof->getVelocity());
		}
		if(step >= TIME_LIMIT) done = true;
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

		for(size_t i = 6; i < biped->getNumDofs(); ++i){
			DegreeOfFreedom* dof = biped->getDof(i);

			state.dofs[i*2] = state.dofs[i*2+1] = dof->getPosition();
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
  SkeletonPtr ground = createFloor(); //DPhy::SkeletonBuilder::BuildFromFile("character/ground.xml");

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
