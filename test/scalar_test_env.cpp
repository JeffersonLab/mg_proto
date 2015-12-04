/*
 * scalar_test_env.cpp
 *
 *  Created on: Oct 2, 2015
 *      Author: bjoo
 */

#include "gtest/gtest.h"
#include "test_env.h"

namespace MGTesting {

	int TestMain(int *argc, char **argv)
	{

		::testing::InitGoogleTest(argc, argv);
		return RUN_ALL_TESTS();

	}

}


