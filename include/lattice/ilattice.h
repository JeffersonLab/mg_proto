/*
 * ilattice.h
 *
 *  Created on: Dec 7, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_ILATTICE_H_
#define INCLUDE_LATTICE_ILATTICE_H_

#include "lattice/constants.h"
#include "lattice/basic_typetraits.h"
#include "lattice/virtual_node.h"

namespace MGGeometry {

// This has to be a leaf class
// F has to be something literal like float/double
// F cannot be like a complex class for now...
template<typename F, typename VN>
class ILattice {
public:
	ILattice() : _n_sites(VN::n_sites) {}
	~ILattice() {}

	constexpr IndexType size() {
		return _n_sites;
	}

	constexpr IndexType OffsetIndex(IndexType i) {
		return i*SizeTraits<F>::IndexSpaceSize();
	}
private:
	IndexType _n_sites;
};

template<typename F, typename VN>
struct SizeTraits<ILattice<F,VN>> {
	static
	inline
	constexpr
	IndexType IndexSpaceSize(void) {
		return VN::n_sites * SizeTraits<F>::IndexSpaceSize();
	}

	static
	inline
	constexpr
	size_t MemorySize(void) {
		return static_cast<size_t>(VN::n_sites)*SizeTraits<F>::MemorySize();
	}
};

}


#endif /* INCLUDE_LATTICE_ILATTICE_H_ */
