#ifndef TEST_ENV_H
#define TEST_ENV_H

#include "test_env.h"
#include "gtest/gtest.h"

/** A Namespace for testing utilities */
namespace MGTesting {

/** A Test Environment to set up QMP */
class TestEnv : public ::testing::Environment {
public:
	TestEnv(int *argc, char ***argv);
	~TestEnv();

private:
		int proc_geometry[4] = {1,1,1,1}; // Default processor geometry
};

int TestMain(int *argc, char **argv);

} // Namespace MGTesting

#endif
