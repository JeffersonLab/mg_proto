#ifndef TEST_ENV_H
#define TEST_ENV_H

#include "test_env.h"
#include "gtest/gtest.h"

/** A Namespace for testing utilities */
namespace MGTesting {

/** A Test Environment to set up QMP */
class QMPTestEnv : public ::testing::Environment {
public:
	QMPTestEnv(int *argc, char ***argv);
	~QMPTestEnv();

private:
		int proc_geometry[4] = {1,1,1,1}; // Default processor geometry

};


} // Namespace MGTesting

#endif
