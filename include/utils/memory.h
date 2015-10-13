/*
 * memory.h
 *
 *  Created on: Oct 13, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_UTILS_MEMORY_H_
#define INCLUDE_UTILS_MEMORY_H_

#include <cstdlib>

namespace MGUtils {

	// We typically have notions of slow and fast memory.
	// For sake of argument: on GPU Fast memory may be GPU memory
	// Slow memory may be CPU memory
	// On future GPU, fast memory may be on package memory
	// On Xeon Phi Fast memory may be HBM

	enum MemorySpace { REGULAR, FAST };

	// These are totally Opaque, and can consist of anything from
	void InitMemory(void);
	void FinalizeMemory(void);

	void *mem_allocate(std::size_t num_bytes, const MemorySpace space=REGULAR);
	void *mem_free(void *ptr);

}




#endif /* INCLUDE_UTILS_MEMORY_H_ */
