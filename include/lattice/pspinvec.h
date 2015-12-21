/*
 * pspinvec.h
 *
 *  Created on: Dec 8, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_PSPINVEC_H_
#define INCLUDE_LATTICE_PSPINVEC_H_

#include "lattice/constants.h"
#include "lattice/basic_typetraits.h"

namespace MGGeometry {

template<typename T>
class PSpinVec {
public:
	PSpinVec(IndexType n_spin) : _n_spin(n_spin) {} // This is now a runtime thing
	~PSpinVec(){}

	IndexType OffsetIndex(IndexType i) {
		return i*SizeTraits<T>::IndexSpaceSize();
	}

	static
	inline
	IndexType IndexSpaceSize() { return n_spin; }

	static
	inline
	IndexType MemorySize() {
		return n_spin * ::MemorySize();
	}

private:
	IndexType _n_spin;
};

template<typename T>
struct SizeTraits<PSpinVec<T>> {
	static
	inline
	IndexType IndexSpaceSize() const {
		return
	}
};

}


#endif /* INCLUDE_LATTICE_PSPINVEC_H_ */
