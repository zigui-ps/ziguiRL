#include<iostream>
#include <dart/dart.hpp>
#include <dart/gui/gui.hpp>
#include <GL/freeglut.h>

const double default_height = 1.0; // m
const double default_width = 0.2;  // m
const double default_depth = 0.2;  // m

const double default_torque = 15.0; // N-m
const double default_force = 15.0;  // N
const int default_countdown = 200;  // Number of timesteps for applying force

const double default_rest_position = 0.0;
const double delta_rest_position = 10.0 * M_PI / 180.0;

const double default_stiffness = 0.0;
const double delta_stiffness = 10;

const double default_damping = 5.0;
const double delta_damping = 1.0;

using namespace dart::dynamics;
using namespace dart::simulation;

const int TIME_LIMIT = 999;

struct State{
	double current[7];
	double prev[7];
};

struct Action{
	double force[5];
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
		SkeletonPtr mPendulum = mWorld->getSkeleton("pendulum");

    // Make sure that the pendulum was found in the World
    assert(mPendulum != nullptr);

		assert(mPendulum->getNumBodyNodes() == 5); // action
		assert(mPendulum->getNumDofs() == 7); // state
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
		SkeletonPtr mPendulum = mWorld->getSkeleton("pendulum");
		
		double reward = 0;
		step += 1;
		for(int i = 0; i < 5; i++){
			if(action.force[i] < -1) action.force[i] = -1;
			if(action.force[i] > 1) action.force[i] = 1;
		}
		for (std::size_t i = 0; i < mPendulum->getNumBodyNodes(); ++i)
		{
			BodyNode* bn = mPendulum->getBodyNode(i);

			Eigen::Vector3d force = (action.force[i] * 50) * Eigen::Vector3d::UnitX();
			Eigen::Vector3d location(-default_width / 2.0, 0.0, default_height / 2.0);

			bn->addExtForce(force, location, true, true);
		}

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

		for(int i = 0; i < 100; i++) timeStepping();

		for(size_t i = 0; i < mPendulum->getNumDofs(); ++i){
			DegreeOfFreedom* dof = mPendulum->getDof(i);

			state.prev[i] = state.current[i];
			state.current[i] = dof->getPosition();

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
		
		SkeletonPtr mPendulum = mWorld->getSkeleton("pendulum");

		for(size_t i = 0; i < mPendulum->getNumDofs(); ++i){
			DegreeOfFreedom* dof = mPendulum->getDof(i);

			state.prev[i] = state.current[i] = dof->getPosition();
		}
	}

	bool done;
	bool tl(){ return step >= TIME_LIMIT; }

protected:
	int step;
	State state;
	WorldPtr initial_world;
};

void setGeometry(const BodyNodePtr& bn)
{
	// Create a BoxShape to be used for both visualization and collision checking
	std::shared_ptr<BoxShape> box(new BoxShape(
				Eigen::Vector3d(default_width, default_depth, default_height)));

	// Create a shape node for visualization and collision checking
	auto shapeNode
		= bn->createShapeNodeWith<VisualAspect, CollisionAspect, DynamicsAspect>(
				box);
	shapeNode->getVisualAspect()->setColor(dart::Color::Blue());

	// Set the location of the shape node
	Eigen::Isometry3d box_tf(Eigen::Isometry3d::Identity());
	Eigen::Vector3d center = Eigen::Vector3d(0, 0, default_height / 2.0);
	box_tf.translation() = center;
	shapeNode->setRelativeTransform(box_tf);

	// Move the center of mass to the center of the object
	bn->setLocalCOM(center);
}

BodyNode* makeRootBody(const SkeletonPtr& pendulum, const std::string& name)
{
	BallJoint::Properties properties;
	properties.mName = name + "_joint";
	properties.mRestPositions = Eigen::Vector3d::Constant(default_rest_position);
	properties.mSpringStiffnesses = Eigen::Vector3d::Constant(default_stiffness);
	properties.mDampingCoefficients = Eigen::Vector3d::Constant(default_damping);

	BodyNodePtr bn
		= pendulum
		->createJointAndBodyNodePair<BallJoint>(
				nullptr, properties, BodyNode::AspectProperties(name))
		.second;

	// Make a shape for the Joint
	const double& R = default_width;
	std::shared_ptr<EllipsoidShape> ball(
			new EllipsoidShape(sqrt(2) * Eigen::Vector3d(R, R, R)));
	auto shapeNode = bn->createShapeNodeWith<VisualAspect>(ball);
	shapeNode->getVisualAspect()->setColor(dart::Color::Blue());

	// Set the geometry of the Body
	setGeometry(bn);

	return bn;
}

BodyNode* addBody(
		const SkeletonPtr& pendulum, BodyNode* parent, const std::string& name)
{
  // Set up the properties for the Joint
  RevoluteJoint::Properties properties;
  properties.mName = name + "_joint";
  properties.mAxis = Eigen::Vector3d::UnitY();
  properties.mT_ParentBodyToJoint.translation()
      = Eigen::Vector3d(0, 0, default_height);
  properties.mRestPositions[0] = default_rest_position;
  properties.mSpringStiffnesses[0] = default_stiffness;
  properties.mDampingCoefficients[0] = default_damping;

  // Create a new BodyNode, attached to its parent by a RevoluteJoint
  BodyNodePtr bn = pendulum
                       ->createJointAndBodyNodePair<RevoluteJoint>(
                           parent, properties, BodyNode::AspectProperties(name))
                       .second;

  // Make a shape for the Joint
  const double R = default_width / 2.0;
  const double h = default_depth;
  std::shared_ptr<CylinderShape> cyl(new CylinderShape(R, h));

  // Line up the cylinder with the Joint axis
  Eigen::Isometry3d tf(Eigen::Isometry3d::Identity());
  tf.linear() = dart::math::eulerXYZToMatrix(
      Eigen::Vector3d(90.0 * M_PI / 180.0, 0, 0));

  auto shapeNode = bn->createShapeNodeWith<VisualAspect>(cyl);
  shapeNode->getVisualAspect()->setColor(dart::Color::Blue());
  shapeNode->setRelativeTransform(tf);

  // Set the geometry of the Body
  setGeometry(bn);

  return bn;
}

std::shared_ptr<MyWindow> window;

void init(int argc, char* argv[])
{
  // Create an empty Skeleton with the name "pendulum"
  SkeletonPtr pendulum = Skeleton::create("pendulum");

  // Add each body to the last BodyNode in the pendulum
  BodyNode* bn = makeRootBody(pendulum, "body1");
  bn = addBody(pendulum, bn, "body2");
  bn = addBody(pendulum, bn, "body3");
  bn = addBody(pendulum, bn, "body4");
  bn = addBody(pendulum, bn, "body5");

  // Set the initial position of the first DegreeOfFreedom so that the pendulum
  // starts to swing right away
   pendulum->getDof(1)->setPosition(180 * M_PI / 180.0);

  // Create a world and add the pendulum to the world
  WorldPtr world = World::create();
  world->addSkeleton(pendulum);

  // Create a window for rendering the world and handling user input
  window = std::shared_ptr<MyWindow>(new MyWindow(world));

  // Print instructions
	std::cout << "init Called!!" << std::endl;

  // Initialize glut, initialize the window, and begin the glut event loop
  glutInit(&argc, argv);
  window->initWindow(640, 480, "Multi-Pendulum Environment");
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
