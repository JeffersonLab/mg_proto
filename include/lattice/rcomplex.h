/*
 * complex.h
 *
 *  Created on: Dec 8, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_RCOMPLEX_H_
#define INCLUDE_LATTICE_RCOMPLEX_H_

#include "lattice/constants.h"
#include "lattice/basic_typetraits.h"

namespace MGGeometry {

template<typename T>
class RComplex {
public:
	RComplex() {}
	~RComplex() {}

	constexpr IndexType OffsetIndex(IndexType i) const {
		return i*SizeTraits<T>::IndexSpaceSize();
	}

	constexpr IndexType RealIndex() const {
		return 0;
	}

	constexpr IndexType ImagIndex() const {
		return SizeTraits<T>::IndexSpaceSize();
	}
};

template<typename T>
struct SizeTraits< RComplex<T> > {
	static
	inline
	constexpr
	IndexType IndexSpaceSize(void) {
		return n_complex*SizeTraits<T>::IndexSpaceSize();
	}

	static
	inline
	constexpr
	size_t MemorySize(void) {
		return n_complex*SizeTraits<T>::MemorySize();
	}
};


}



#endif /* INCLUDE_LATTICE_RCOMPLEX_H_ */
