/*
 * initialize.h
 *
 *  Created on: Oct 13, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_UTILS_INITIALIZE_H_
#define INCLUDE_UTILS_INITIALIZE_H_

namespace MG {
	void initialize(int *argc, char ***argv);  // Initialize our system
	void finalize();                         // Finalize our system
	void abort();                            // Abort the system
};



#endif /* INCLUDE_UTILS_INITIALIZE_H_ */
