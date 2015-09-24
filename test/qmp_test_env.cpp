#include "qmp_test_env.h"

#include "MG_config.h"
#ifdef QMP_COMMS
#include "qmp.h"
#endif

namespace MGTesting {

	/** The Constructor to set up a test environment.
	 *   Its job is essentially to set up QMP
	 */
	QMPTestEnv::QMPTestEnv(int  *argc, char ***argv)
	{
		// Process args
		int i=1;
		int my_argc = (*argc);
		char **my_argv = (*argv);

		/* Process args here -- first step is to get the processor geomerty */
		while( i < my_argc ) {

		}

		/* Initialize QMP here */
#ifdef QMP_COMMS
		QMP_thread_level_t prv;
		if( QMP_init_msg_passing(argc, argv, QMP_THREAD_SINGLE, &prv) != QMP_SUCCESS ) {
			std::cout << "Failed to initialize QMP" << std::endl;
			std::exit(EXIT_FAILURE);
		}
#endif
	}

	QMPTestEnv::~QMPTestEnv() {
		/* Tear down QMP */
#ifdef QMP_COMMS
		QMP_finalize_msg_passing();
#endif
	}

	/* This is a convenience routine to setup the test environment for GTest and its layered test environments */
	int TestMain(int *argc, char **argv)
	{
		  ::testing::InitGoogleTest(argc, argv);
		  ::testing::Environment* const chroma_env = ::testing::AddGlobalTestEnvironment(new MGTesting::QMPTestEnv(argc,&argv));
		  return RUN_ALL_TESTS();
	}

}



