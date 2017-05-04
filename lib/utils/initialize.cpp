/*
 * initialize.cpp
 *
 *  Created on: Oct 13, 2015
 *      Author: bjoo
 */

#include "MG_config.h"
#ifdef MG_QMP_INIT
#include <qmp.h>
#endif

#ifdef MG_USE_QDPXX
#include "qdp.h"
#endif

#include "utils/memory.h"
#include "utils/print_utils.h"
#include <string>
#include <cstdlib>


// Kokkos here:
#ifdef  MG_USE_KOKKOS
#include <Kokkos_Core.hpp>
#endif

namespace MG
{

	void initialize(int *argc, char ***argv)
	{
		// Process args
#ifdef MG_QMP_INIT
		int proc_geometry[4] = {1,1,1,1}; // Default processor geometry
#endif

		int i=0;
		int my_argc = (*argc);

		/* Process args here -- first step is to get the processor geomerty */
		while( i < my_argc ) {
#ifdef MG_QMP_INIT
			if (std::string(argv[i]).compare("-geom") == 0 ) {
			      proc_geometry[0] = std::atoi((*argv)[i+1]);
			      proc_geometry[1] = std::atoi((*argv)[i+2]);
			      proc_geometry[2] = std::atoi((*argv)[i+3]);
			      proc_geometry[3] = std::atoi((*argv)[i+4]);
			      i+=4;
			}
			else {
				 ++i;
			}
#else
			++i;
#endif

		}

		/* Initialize QMP here */
#if defined(MG_QMP_INIT)
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
#elif defined(MG_USE_QDPXX)
		// MasterLog(INFO, "Initializing QDP++");
		QDP::QDP_initialize(argc,argv);
		MasterLog(INFO, "QDP++ Initialized");
#endif
		MG::InitMemory(argc,argv);

#ifdef MG_USE_KOKKOS
		MasterLog(INFO, "Initializing Kokkos");
		Kokkos::initialize(*argc,*argv);
#endif

	}

	void finalize(void)
	{
#ifdef MG_USE_KOKKOS
		MasterLog(INFO, "Finalizing Kokkos");
		Kokkos::finalize();
#endif
		MasterLog(INFO, "Finalizing Memory");
		MG::FinalizeMemory();
#if defined(MG_QMP_INIT)
		MasterLog(INFO, "Finalizing QMP");
		QMP_finalize_msg_passing();
#elif defined(MG_USE_QDPXX)
		MasterLog(INFO, "Finalizing QDP++");
		QDP::QDP_finalize();
#endif

	}

	void abort(void)
	{
		MG::FinalizeMemory();
#if defined(MG_QMP_INIT)
		QMP_abort(1);

#else
		std::abort();
#endif
	}


}

