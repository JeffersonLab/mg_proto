/*
 * initialize.cpp
 *
 *  Created on: Oct 13, 2015
 *      Author: bjoo
 */

#include "MG_config.h"
#ifdef QMP_COMMS
#include <qmp.h>
#endif

#include "utils/memory.h"
#include "utils/print_utils.h"
#include <string>
#include <cstdlib>

namespace MG
{
	void initialize(int *argc, char ***argv)
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
		MGUtils::InitMemory();
	}

	void finalize(void)
	{
		MGUtils::FinalizeMemory();
#ifdef QMP_COMMS
		QMP_finalize_msg_passing();
#endif
	}

	void abort(void)
	{
		MGUtils::FinalizeMemory();
#ifdef QMP_COMMS
		QMP_abort(1);
#else
		std::abort();
#endif
	}


}

