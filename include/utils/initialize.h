/*
 * initialize.h
 *
 *  Created on: Oct 13, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_UTILS_INITIALIZE_H_
#define INCLUDE_UTILS_INITIALIZE_H_
#include "MG_config.h"

#ifdef MG_USE_QPHIX
#include <qphix/qphix_cli_args.h>
#include <memory>
#endif
namespace MG {
	bool isInitialized(void);
	void initialize(int *argc, char ***argv);  // Initialize our system
	void finalize();                        // Finalize our system
	void abort();                            // Abort the system

#ifdef MG_USE_QPHIX
	QPhiX::QPhiXCLIArgs& getQPhiXCLIArgs(void);
#endif

};



#endif /* INCLUDE_UTILS_INITIALIZE_H_ */
