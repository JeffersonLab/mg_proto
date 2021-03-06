#include "gtest/gtest.h"
#include "utils/print_utils.h"
#include <vector>
#include <iostream>
#include "MG_config.h"
#include "test_env.h"

using namespace MG; 


TEST(TestLogging, TestDefaultLoglevel)
{
   LogLevel current = GetLogLevel();
   ASSERT_EQ( current, MG_DEFAULT_LOGLEVEL );
}

TEST(TestLogging, TestLogLevelSet)
{
	SetLogLevel(DEBUG3);
	ASSERT_EQ( GetLogLevel(), DEBUG3);
	SetLogLevel(INFO);
	ASSERT_EQ(GetLogLevel(),INFO);
}


TEST(TestLogging, TestLogLevelSetOpenMP)
{
#pragma omp parallel
	{
	    	SetLogLevel(DEBUG3);
	}
	ASSERT_EQ( GetLogLevel(), DEBUG3);
#pragma omp parallel
	{
		SetLogLevel(INFO);
	}
	ASSERT_EQ(GetLogLevel(),INFO);
}


TEST(TestLogging, TestMasterLogLevels)
{
		SetLogLevel(DEBUG3);
#pragma omp parallel
{
		MasterLog(INFO, "Info Level");
		MasterLog(DEBUG, "Debug Level");
		MasterLog(DEBUG2, "Debug Level2");
		MasterLog(DEBUG3, "Debug Level3");
}
}

TEST(TestLogging, TestLocalLogLevels)
{
		SetLogLevel(DEBUG3);
		int int_var=1; double float_var=5.0e-8;
#pragma omp parallel
{
		LocalLog(INFO, "Info Level %d %e", int_var, float_var);
		LocalLog(DEBUG, "Debug Level");
		LocalLog(DEBUG2, "Debug Level2");
		LocalLog(DEBUG3, "Debug Level3");
}
}

TEST(TEstLogging, LocalErrorTerminates)
{
	/* This should be in an parallel region, but that interferes with the Death Tests */
	{
		EXPECT_DEATH( LocalLog(ERROR, "Terminate With Prejudice"), "");
	}
}


int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

