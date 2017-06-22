/*
 * kokkos_types.h
 *
 *  Created on: May 23, 2017
 *      Author: bjoo
 */

#ifndef TEST_KOKKOS_KOKKOS_TYPES_H_
#define TEST_KOKKOS_KOKKOS_TYPES_H_
#include <memory>

#include "kokkos_defaults.h"
#include "lattice/lattice_info.h"
#include "utils/print_utils.h"
namespace MG
{

	template<typename T, int _num_spins>
	class KokkosCBFineSpinor {
	public:
		KokkosCBFineSpinor(const LatticeInfo& info, IndexType cb)
		: _cb_data("cb_data", info.GetNumCBSites(), _num_spins), _info(info), _cb(cb) {

			if( _info.GetNumColors() != 3 ) {
				MasterLog(ERROR, "KokkosCBFineSpinor has to have 3 colors in info. Info has %d", _info.GetNumColors());
			}
			if( _info.GetNumSpins() != _num_spins )
			{
				MasterLog(ERROR, "KokkosCBFineSpinor has to have %d spins in info. Info has %d",
						_num_spins,_info.GetNumColors());
			}
		}

		inline
		const T& operator()(int cb_site, int spin, int color, int reim) const
		{
			return _cb_data(cb_site,spin,color,reim);
		}

		inline
		T& operator()(int cb_site, int spin, int color, int reim)
		{
			return _cb_data(cb_site,spin,color,reim);
		}

		inline
		const LatticeInfo& GetInfo() const {
			return _info;
		}

		inline
		IndexType GetCB() const {
			return _cb;
		}

		using DataType = Kokkos::View<T**[3][2],Layout>;


		const DataType& GetData() const {
			return _cb_data;
		}


		DataType& GetData() {
			return _cb_data;
		}

	private:
		DataType _cb_data;
		const LatticeInfo& _info;
		const IndexType _cb;
	};


#if 0
	template<typename T>
	class KokkosFineSpinor {
	public:
		KokkosFineSpinor(const LatticeInfo& info)
		: _data("_data", info.GetNumSites()), _info(info) {

			if( _info.GetNumColors() != 3 ) {
				MasterLog(ERROR, "KokkosFineSpinor has to have 3 colors in info. Info has %d", _info.GetNumColors());
			}
			if( _info.GetNumSpins() != 4 ) {
				MasterLog(ERROR, "KokkosFineSpinor has to have 4 spins in info. Info has %d", _info.GetNumSpins());
			}
		}

		inline
		const T& operator()(int site, int spin, int color,int reim) const
		{
			return _data(site,spin,color,reim);
		}

		inline
		T& operator()(int site, int spin, int color, int reim)
		{
			return _data(site,spin,color,reim);
		}

		inline
		const LatticeInfo& GetInfo() const {
			return _info;
		}

		using DataType = Kokkos::View<T*[4][3][2],Layout>;

		inline
		const DataType& GetData() const {
			return _data;
		}

		inline
		DataType& GetData() {
			return _data;
		}

	private:
		DataType _data;
		const LatticeInfo& _info;
	};
#endif

	template<typename T>
	class KokkosCBFineGaugeField {
	public:
		KokkosCBFineGaugeField(const LatticeInfo& info, IndexType cb)
		: _cb_gauge_data("cb_gauge_data", info.GetNumCBSites()), _info(info), _cb(cb) {

			if( _info.GetNumColors() != 3 ) {
				MasterLog(ERROR, "KokkosFineGaugeField needs to have 3 colors. Info has %d", _info.GetNumColors());
			}
		}

		inline
		 const T& operator()(int site, int dir, int color1, int color2, int reim) const {
			return _cb_gauge_data(site,dir,color1,color2,reim);
		}

		inline
		T& operator()(int site, int dir, int color1,int color2, int reim) {
			return _cb_gauge_data(site,dir,color1,color2,reim);
		}

		using DataType = Kokkos::View<T*[4][3][3][2],Layout>;
		DataType& GetData() {
			return _cb_gauge_data;
		}

		const DataType& GetData() const {
			return _cb_gauge_data;
		}

		IndexType GetCB() const {
			return _cb;
		}

		const LatticeInfo& GetInfo() const {
				return _info;
		}
	private:
		DataType _cb_gauge_data;
		const LatticeInfo& _info;
		IndexType _cb;
	};

	template<typename T>
	class KokkosFineGaugeField {
	private:
		std::shared_ptr< KokkosCBFineGaugeField<T> > _gauge_data[2];
		const LatticeInfo& _info;
	public:
		KokkosFineGaugeField(const LatticeInfo& info) :  _info(info)
		{
			_gauge_data[ EVEN ] = std::make_shared< KokkosCBFineGaugeField<T> >(info,EVEN);
			_gauge_data[ ODD  ] = std::make_shared< KokkosCBFineGaugeField<T> >(info,ODD);
		}

		const KokkosCBFineGaugeField<T>& operator()(IndexType cb) const
		{
			return *(_gauge_data[cb]);
		}

		KokkosCBFineGaugeField<T>& operator()(IndexType cb) {
			return *(_gauge_data[cb]);
		}
	};

	template<typename T>
	using SpinorView = typename KokkosCBFineSpinor<T,4>::DataType;

#if 0
	template<typename T>
	using SpinorSiteView = Kokkos::View<T[4][3][2]>;
#else
	template<typename T>
	struct SpinorSiteView {
		T _data[4][3][2];
		KOKKOS_INLINE_FUNCTION T& operator()(int idx1, int idx2, int idx3 ) {
			return _data[idx1][idx2][idx3];
		}
		KOKKOS_INLINE_FUNCTION const T& operator()(int idx1, int idx2, int idx3) const {
			return _data[idx1][idx2][idx3];
		}
	};
#endif
	template<typename T>
	using HalfSpinorView = typename KokkosCBFineSpinor<T,2>::DataType;


#if 0
	template<typename T>
	using HalfSpinorSiteView = Kokkos::View<T[2][3][2]>;
#else
	template<typename T>
	struct HalfSpinorSiteView {
		T _data[2][3][2];
		KOKKOS_FORCEINLINE_FUNCTION T& operator()(int idx1, int idx2, int idx3 ) {
			return _data[idx1][idx2][idx3];
		}
		KOKKOS_FORCEINLINE_FUNCTION const T& operator()(int idx1, int idx2, int idx3) const {
			return _data[idx1][idx2][idx3];
		}
	};
#endif
	template<typename T>
	using GaugeView = typename KokkosCBFineGaugeField<T>::DataType;


#if 0
	// Some shorthand
	// FIXME: Finger Saving... Maybe define these with kokkos_types
		template<typename T>
		using SpinorView = Kokkos::View<T[4][3][2],Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;

		template<typename T>
		using MatrixView = Kokkos::View<T[3][3][2],Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;

		template<typename T>
		using HalfSpinorView = Kokkos::View<T[2][3][2],Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;

#if 1
#if ! defined KOKKOS_USING_DEPRECATED_VIEW
typedef Kokkos::Experimental::Impl::ALL_t ALL_t;
#else
typedef Kokkos::ALL ALL_t;
#endif

		template<typename T>
		KOKKOS_INLINE_FUNCTION
		MatrixView<T> subview_me(const typename KokkosCBFineGaugeField<T>::DataType& view, int idx0, int idx1) {
		    return MatrixView<T>(&view(idx0,idx1,0,0,0));
		  }

		template<typename T>
		KOKKOS_INLINE_FUNCTION
		HalfSpinorView<T>subview_me(const typename KokkosCBFineSpinor<T,2>::DataType& view, int idx) {
			return HalfSpinorView<T>(&view(idx,0,0,0));
		}

		template<typename T>
		KOKKOS_INLINE_FUNCTION
		MatrixView<T> subview_me(typename KokkosCBFineGaugeField<T>::DataType& view, int idx0, int idx1) {
		    return MatrixView<T>(&view(idx0,idx1,0,0,0));
		  }

		template<typename T>
		KOKKOS_INLINE_FUNCTION
		HalfSpinorView<T>subview_me(typename KokkosCBFineSpinor<T,2>::DataType& view, int idx) {
			return HalfSpinorView<T>(&view(idx,0,0,0));
		}

		template<typename T>
		KOKKOS_INLINE_FUNCTION
		SpinorView<T> subview_me(const typename KokkosCBFineSpinor<T,2>::DataType& view, int idx) {
			return SpinorView<T>(&view(idx,0,0,0));
		}
#endif
#endif
};




#endif /* TEST_KOKKOS_KOKKOS_TYPES_H_ */
