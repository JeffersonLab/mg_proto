/*
 * buffer.h
 *
 *  Created on: Nov 24, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_BUFFER_H_
#define INCLUDE_LATTICE_BUFFER_H_

#include "utils/memory.h"
namespace MGGeometry {
// NB: At this level the BlockSpinor Does not need a ghost
// The block spinor will need block: norm2(), norm() innerProduct(), add(), sub(), madd(), msub() and scale() operations
// To perform the block orthogonalization.
// The add, sub are trivially parallel
// The mul, madd, msub involves manipulating complex components
// the norm2() and norm are simple reductions
// the innerProduct involves manipulating components


  template<typename T>
  class Buffer {
  public:
	  Buffer(IndexType n_elem, const MGUtils::MemorySpace Space=MGUtils::REGULAR) {
#pragma omp master
		  {
			  // Master thread allocates -- using MGUtils Allocato
			  _data = reinterpret_cast<T*>(MGUtils::MemoryAllocate(n_elem*sizeof(T),Space));
		  }
#pragma omp barrier
	  }

	  ~Buffer()
	  {
#pragma omp master
		  {
			  MGUtils::MemoryFree(_data);
		  }
#pragma omp barrier
	  }

	  /* Access data array -- beware threading */
	  T* GetData(void) {
		  return _data;
	  }

	  /** Get a pointer to a particular element. */
	  T* GetData(IndexType elem) {
		  return &_data[elem];
	  }
  private:
  	  T* _data;
  };


}

#endif /* INCLUDE_LATTICE_BUFFER_H_ */
