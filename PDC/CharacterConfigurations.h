#ifndef __DEEP_PHYSICS_CHARACTER_CONFIGURATIONS_H__
#define __DEEP_PHYSICS_CHARACTER_CONFIGURATIONS_H__

// #define MOTION_WALK_EXTENSION
// #define MOTION_BASKETBALL
// #define MOTION_ZOMBIE
// #define MOTION_GORILLA
// #define MOTION_WALKRUN
// #define MOTION_WALK_NEW
// #define MOTION_WALKFALL
// #define MOTION_JOG_ROLL
#define MOTION_WALKRUNFALL

#ifdef MOTION_WALK_EXTENSION
#define ALL_JOINTS
#define CONTROL_TYPE (0)
#define FOOT_OFFSET (-0.09)
#define ROOT_HEIGHT_OFFSET (-0.005)
#define TERMINAL_ROOT_DIFF_THRESHOLD (1.0)
#define TERMINAL_ROOT_DIFF_ANGLE_THRESHOLD (0.5*M_PI)
#define TERMINAL_ROOT_HEIGHT_LOWER_LIMIT (0.55)
#define TERMINAL_ROOT_HEIGHT_UPPER_LIMIT (2.0)
#endif

#ifdef MOTION_BASKETBALL
#define ALL_JOINTS
#define CONTROL_TYPE (0)
#define FOOT_OFFSET (-0.04)
#define ROOT_HEIGHT_OFFSET (-0.005)
#define TERMINAL_ROOT_DIFF_THRESHOLD (3.0)
#define TERMINAL_ROOT_DIFF_ANGLE_THRESHOLD (0.5*M_PI)
#define TERMINAL_ROOT_HEIGHT_LOWER_LIMIT (0.55)
#define TERMINAL_ROOT_HEIGHT_UPPER_LIMIT (2.0)
#endif

#ifdef MOTION_ZOMBIE
#define CMU_JOINTS
#define CONTROL_TYPE (0)
#define FOOT_OFFSET (-0.01)
#define ROOT_HEIGHT_OFFSET (-0.17)
#define TERMINAL_ROOT_DIFF_THRESHOLD (1.0)
#define TERMINAL_ROOT_DIFF_ANGLE_THRESHOLD (0.5*M_PI)
#define TERMINAL_ROOT_HEIGHT_LOWER_LIMIT (0.55)
#define TERMINAL_ROOT_HEIGHT_UPPER_LIMIT (2.0)
#endif

#ifdef MOTION_WALKRUN
#define CMU_JOINTS
#define CONTROL_TYPE (1)
#define FOOT_OFFSET (-0.1)
#define ROOT_HEIGHT_OFFSET (-0.19)
#define TERMINAL_ROOT_DIFF_THRESHOLD (1.0)
#define TERMINAL_ROOT_DIFF_ANGLE_THRESHOLD (0.5*M_PI)
#define TERMINAL_ROOT_HEIGHT_LOWER_LIMIT (0.55)
#define TERMINAL_ROOT_HEIGHT_UPPER_LIMIT (2.0)
#endif

#ifdef MOTION_GORILLA
#define CMU_JOINTS
#define CONTROL_TYPE (0)
#define FOOT_OFFSET (0.1)
#define ROOT_HEIGHT_OFFSET (-0.1)
#define TERMINAL_ROOT_DIFF_THRESHOLD (1.0)
#define TERMINAL_ROOT_DIFF_ANGLE_THRESHOLD (0.5*M_PI)
#define TERMINAL_ROOT_HEIGHT_LOWER_LIMIT (0.3)
#define TERMINAL_ROOT_HEIGHT_UPPER_LIMIT (2.5)
#endif

#ifdef MOTION_WALK_NEW
#define NEW_JOINTS
#define CONTROL_TYPE (0)
#define FOOT_OFFSET (0.0)
#define ROOT_HEIGHT_OFFSET (0.00)
#define TERMINAL_ROOT_DIFF_THRESHOLD (1.0)
#define TERMINAL_ROOT_DIFF_ANGLE_THRESHOLD (0.5*M_PI)
#define TERMINAL_ROOT_HEIGHT_LOWER_LIMIT (0.55)
#define TERMINAL_ROOT_HEIGHT_UPPER_LIMIT (2.0)
#endif

#ifdef MOTION_WALKFALL
#define NEW_JOINTS
#define CONTROL_TYPE (1)
#define FOOT_OFFSET (0.0)
#define ROOT_HEIGHT_OFFSET (0.00)
#define TERMINAL_ROOT_DIFF_THRESHOLD (1.0)
#define TERMINAL_ROOT_DIFF_ANGLE_THRESHOLD (0.5*M_PI)
#define TERMINAL_ROOT_HEIGHT_LOWER_LIMIT (0.0)
#define TERMINAL_ROOT_HEIGHT_UPPER_LIMIT (2.0)
#endif

#ifdef MOTION_WALKRUNFALL
#define NEW_JOINTS
#define CONTROL_TYPE (0)
#define FOOT_OFFSET (0.0)
#define ROOT_HEIGHT_OFFSET (0.00)
#define TERMINAL_ROOT_DIFF_THRESHOLD (3.0)
#define TERMINAL_ROOT_DIFF_ANGLE_THRESHOLD (0.5*M_PI)
#define TERMINAL_ROOT_HEIGHT_LOWER_LIMIT (0.0)
#define TERMINAL_ROOT_HEIGHT_UPPER_LIMIT (2.0)
#endif


#ifdef MOTION_JOG_ROLL
#define NEW_JOINTS
#define CONTROL_TYPE (0)
#define FOOT_OFFSET (0.0)
#define ROOT_HEIGHT_OFFSET (-0.08)
#define TERMINAL_ROOT_DIFF_THRESHOLD (1.0)
#define TERMINAL_ROOT_DIFF_ANGLE_THRESHOLD (0.5*M_PI)
#define TERMINAL_ROOT_HEIGHT_LOWER_LIMIT (0.0)
#define TERMINAL_ROOT_HEIGHT_UPPER_LIMIT (2.0)
#endif

#ifdef ALL_JOINTS
#define INPUT_MOTION_SIZE 72
#define FOOT_CONTACT_OFFSET 70
#endif

#ifdef CMU_JOINTS
#define INPUT_MOTION_SIZE 78
#define FOOT_CONTACT_OFFSET 70
#endif

#ifdef NEW_JOINTS
#define INPUT_MOTION_SIZE 54
#define OUTPUT_MOTION_SIZE 111
#define FOOT_CONTACT_OFFSET 52
#endif

#define REFERENCE_MANAGER_COUNT (1)

#define FUTURE_TIME (0.33)
#define FUTURE_COUNT (0)
#define FUTURE_DISPLAY_COUNT (1)
#define KV_RATIO (0.1)
#define JOINT_DAMPING (0.05)
#define FOOT_CONTACT_THERESHOLD (0.3)
#define FEEDBACK_INTERVAL (1)

#define FORCE_MAGNITUDE (20)
#define FORCE_APPLYING_FRAME (60)
#define FORCE_APPLYING_BODYNODE ("Neck")

#endif
