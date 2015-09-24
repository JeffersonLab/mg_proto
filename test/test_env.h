#ifndef TEST_ENV_H
#define TEST_ENV_H

#include "gtest/gtest.h"
#include "MG_config.h"
#ifdef QMP_COMMS
#include "qmp.h"
#endif

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

#endif
