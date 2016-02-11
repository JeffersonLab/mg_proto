/*
 * memory.h
 *
 *  Created on: Oct 13, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_UTILS_MEMORY_H_
#define INCLUDE_UTILS_MEMORY_H_

#include <cstdlib>

// FIXME: Implement a C++ Allocator using these functions
//     Allocation Policies could include Tracking memory usage (baked into current)
//     as well as the Memory Space
//
//     Kokkos does a good job of this already I bet.
//

namespace MG {

	// We typically have notions of slow and fast memory.
	// For sake of argument: on GPU Fast memory may be GPU memory
	// Slow memory may be CPU memory
	// On future GPU, fast memory may be on package memory
	// On Xeon Phi Fast memory may be HBM

	// These simple allocators can be used to build cache
	// abstractions

	enum MemorySpace { REGULAR, FAST };

	// These are totally Opaque, and can consist of anything from
	void InitMemory(int *argc, char ***argv);
	void FinalizeMemory(void);

	void*
	MemoryAllocate(std::size_t num_bytes, const MemorySpace space=REGULAR);

	void
	MemoryFree(void *ptr);

	std::size_t
	GetCurrentRegularMemoryUsage(void);

	std::size_t
	GetMaxRegularMemoryUsage(void);

	std::size_t
	GetCurrentFastMemoryUsage(void);

	std::size_t
	GetMaxFastMemoryUsage(void);

	std::size_t
	GetMemoryAlignment(void);

	// Should I have CopyFastToSlow and CopySlowToFast ?

}




#endif /* INCLUDE_UTILS_MEMORY_H_ */
