#include "qmp_test_env.h"
#include <string>
#include <cstdlib>
#include "MG_config.h"
#ifdef QMP_COMMS
#include "qmp.h"
#endif

#include "utils/print_utils.h"
using namespace MGUtils;

namespace MGTesting {

	/** The Constructor to set up a test environment.
	 *   Its job is essentially to set up QMP
	 */
	QMPTestEnv::QMPTestEnv(int  *argc, char ***argv)
	{
		// Process args
		int i=0;
		int my_argc = (*argc);
		char **my_argv = (*argv);

		/* Process args here -- first step is to get the processor geomerty */
		while( i < my_argc ) {
			if (std::string(my_argv[i]).compare("-geom") == 0 ) {
			      proc_geometry[0] = std::atoi(my_argv[i+1]);
			      proc_geometry[1] = std::atoi(my_argv[i+2]);
			      proc_geometry[2] = std::atoi(my_argv[i+3]);
			      proc_geometry[3] = std::atoi(my_argv[i+4]);
			      i+=4;
			}
			else {
				 ++i;
			}
		}

		/* Initialize QMP here */
#ifdef QMP_COMMS
		QMP_thread_level_t prv;
		if( QMP_init_msg_passing(argc, argv, QMP_THREAD_SINGLE, &prv) != QMP_SUCCESS ) {
			std::cout << "Failed to initialize QMP" << std::endl;
			std::exit(EXIT_FAILURE);
		}

		MasterLog(INFO, "QMP IS INITIALIZED");

		// Declare the logical topology
		if ( QMP_declare_logical_topology(proc_geometry, 4)!= QMP_SUCCESS ) {
		    MasterLog(ERROR,"Failed to declare QMP Logical Topology");
		    abort();
		}


		MasterLog(INFO, "Declared QMP Topology: %d %d %d %d\n",
				  proc_geometry[0], proc_geometry[1], proc_geometry[2], proc_geometry[3]);
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
		  ::testing::AddGlobalTestEnvironment(new MGTesting::QMPTestEnv(argc,&argv));
		  return RUN_ALL_TESTS();
	}

}



