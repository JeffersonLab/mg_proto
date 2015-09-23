#include "gtest/gtest.h"
#include "utils/print_utils.h"
#include <vector>
#include <iostream>

#include "MG_config.h"
#ifdef QMP_COMMS
#include "qmp.h"
#endif

using namespace MGUtils; 


TEST(TestLogging, TestDefaultLoglevel)
{
   LogLevel current = GetLogLevel();
   ASSERT_EQ( current, MG_DEFAULT_LOGLEVEL );
}

TEST(TestLogging, TestLogLevelSet)
{
	LogLevel current = GetLogLevel();
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

		MasterLog(INFO, "Info Level");
		MasterLog(DEBUG, "Debug Level");
		MasterLog(DEBUG2, "Debug Level2");
		MasterLog(DEBUG3, "Debug Level3");

}

TEST(TestLogging, TestLocalLogLevels)
{
		SetLogLevel(DEBUG3);
		int int_var=1; double float_var=5.0e-8;
		LocalLog(INFO, "Info Level %d %e", int_var, float_var);
		LocalLog(DEBUG, "Debug Level");
		LocalLog(DEBUG2, "Debug Level2");
		LocalLog(DEBUG3, "Debug Level3");
}

TEST(TEstLogging, LocalErrorTerminates)
{
		EXPECT_DEATH( LocalLog(ERROR, "Terminate With Prejudice"), "");
}

class TestEnv : public ::testing::Environment {
public:

		TestEnv(int  *argc, char ***argv)  {
#ifdef QMP_COMMS
			QMP_thread_level_t prv;
			if( QMP_init_msg_passing(argc, argv, QMP_THREAD_SINGLE, &prv) != QMP_SUCCESS ) {
				std::cout << "Failed to initialize QMP" << std::endl;
				std::exit(EXIT_FAILURE);
			}
#endif
		}

		~TestEnv()
		{
#ifdef QMP_COMMS
				QMP_finalize_msg_passing();
#endif
		}

};

int main(int argc, char *argv[]) 
{
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::Environment* const chroma_env = ::testing::AddGlobalTestEnvironment(new TestEnv(&argc,&argv));
  return RUN_ALL_TESTS();
}

