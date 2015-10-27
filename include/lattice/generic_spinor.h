/*
 * spinor.h
 *
 *  Created on: Oct 20, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_GENERIC_SPINOR_H_
#define INCLUDE_LATTICE_GENERIC_SPINOR_H_

#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/geometry_utils.h"
#include "utils/memory.h"
#include <memory>

namespace MGGeometry {

  /** A Class to hold spinors
   *
   *  These are essentially vectors
   *  Data is needed to store Spin, Color, Real and Imaginary components
   *
   *  This class is just a 'container' and can do only rudimentary indexing
   */

// From playpen code example:

// Spinor Needs GetLatticeInfo function
// BlockSpinor needs Index(chirality_aggregate, block, leftover_spin, thin_color, reim, blocksite) -- accessor (read & write)
// Spinor Needs Index(spin, color, reim, fullsite) -- accessor(read & write)

// NB: At this level the BlockSpinor Does not need a ghost
// The block spinor will need block: norm2(), norm() innerProduct(), add(), sub(), madd(), msub() and scale() operations
// To perform the block orthogonalization.
// The add, sub are trivially parallel
// The mul, madd, msub involves manipulating complex components
// the norm2() and norm are simple reductions
// the innerProduct involves manipulating components





  template<typename T,        // Type in the body
  	  	   template <typename T2> class Layout,  // This allows layout to templated
		                                         // But in the General Spinor I force T2 = T
		   const MGUtils::MemorySpace Space = MGUtils::REGULAR>
  class GeneralLatticeSpinor {
  public:

	GeneralLatticeSpinor(const GeneralLatticeSpinor<T, Layout, Space>& to_copy) :
			_layout(to_copy._layout), _data(to_copy._data) {}

	GeneralLatticeSpinor(const Layout<T>& layout) : _layout(layout) {

#pragma omp master
		  {
			  // Master thread allocates -- using MGUtils Allocator
			  MasterLog(DEBUG, "Allocating %d elements", _layout.DataNumElem());
			  MasterLog(DEBUG, "Allocating %u bytes", _layout.DataInBytes());
			  _data = reinterpret_cast<T*>(MGUtils::MemoryAllocate(_layout.DataInBytes(),Space));
		  }
#pragma omp barrier

  	  }


	  ~GeneralLatticeSpinor()
	  {
#pragma omp master
		  {
			  MGUtils::MemoryFree(_data);
		  }
#pragma omp barrier
	  }

	  inline
	  T&  Index(IndexType elem, IndexType spin, IndexType color, IndexType reim) {
		  return _data[ _layout.ContainerIndex(elem,spin,color,reim) ];
	  }

	  inline
	  const T& Index(IndexType elem, IndexType spin, IndexType color, IndexType reim) const {
		  return _data[ _layout.ContainerIndex(elem,spin,color,reim) ];
	  }

	  inline
	  T&  Index(IndexType cb, IndexType cb_index, IndexType spin, IndexType color, IndexType reim) {
		  return _data[ _layout.ContainerIndex(cb,cb_index,spin,color,reim) ];
	  }

	  inline
	  const T& Index(IndexType cb, IndexType cb_index, IndexType spin, IndexType color, IndexType reim) const {
		  return _data[ _layout.ContainerIndex(cb, cb_index, spin,color,reim) ];
	  }

	  inline
	  const  LatticeInfo& GetLatticeInfo() const {
		  return _layout.GetLatticeInfo();
	  }
  private:

	  const Layout<T> _layout;
	  T* _data;

  };



}




#endif /* INCLUDE_LATTICE_LATTICE_SPINOR_H_ */
