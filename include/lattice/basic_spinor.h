/*
 * basic_spinor.h
 *
 *  Created on: Dec 4, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_BASIC_SPINOR_H_
#define INCLUDE_LATTICE_BASIC_SPINOR_H_

#include <memory>
#include "lattice/constants.h"      // Everyone pretty much needs this
#include "lattice/lattice_info.h"   // This we need also
#include "lattice/layouts/cb_soa_spinor_layout.h"
#include "lattice/buffer.h"

namespace MGGeometry
{

/* Value Type */
template<typename T>
class LatticeGeneralSpinor {
public:
	explicit LatticeGeneralSpinor(const LatticeInfo& info,
								  const MGUtils::MemorySpace Space=MGUtils::REGULAR) : _layout(CBSOASpinorLayout<T>(info)),
		_buffer(new Buffer<T>(_layout.GetNumData(),Space)) {
		_data = _buffer->GetData();
	}

	LatticeGeneralSpinor(const LatticeGeneralSpinor<T>& to_copy) : _layout(to_copy._layout),
			_buffer(to_copy._buffer), _data(to_copy._data) {}

	LatticeGeneralSpinor<T> operator=(const LatticeGeneralSpinor<T>& to_copy_assign) {
		LatticeGeneralSpinor ret_val(to_copy_assign);
		return ret_val;
	}

	~LatticeGeneralSpinor() {} // Obviously will need to destroy also

	inline
	const LatticeInfo& GetLatticeInfo(void) const
	{
		return _layout.GetLatticeInfo();
	}

	/* Get at explicit element r/o */
	const T& operator()(IndexType cb, IndexType site, IndexType spin, IndexType color, IndexType reim) const
	{
		return _data[ _layout.Index(cb,site,spin,color,reim) ];
	}


	/* Get at explicit element r/w */
	T& operator()(IndexType cb, IndexType site, IndexType spin, IndexType color, IndexType reim)
	{
		return _data[ _layout.Index(cb,site,spin,color,reim) ];
	}

private:
	const CBSOASpinorLayout<T> _layout; // Keep the Info
	std::shared_ptr<Buffer<T>> _buffer; // We have ownership
	T* _data; // Direct pointer into the data.
};



}



#endif /* INCLUDE_LATTICE_BASIC_SPINOR_H_ */
