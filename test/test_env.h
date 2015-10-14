#ifndef TEST_ENV_H
#define TEST_ENV_H

#include "gtest/gtest.h"

/** A Namespace for testing utilities */
namespace MGTesting {

/** A Test Environment to set up QMP */
class TestEnv : public ::testing::Environment {
public:
	TestEnv(int *argc, char ***argv);
	~TestEnv();
};

int TestMain(int *argc, char **argv);

} // Namespace MGTesting

#endif
