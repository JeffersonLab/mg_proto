/*
 * kokkos_types.h
 *
 *  Created on: May 23, 2017
 *      Author: bjoo
 */

#ifndef TEST_KOKKOS_KOKKOS_TYPES_H_
#define TEST_KOKKOS_KOKKOS_TYPES_H_
#include <memory>

#include <Kokkos_Complex.hpp>
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
		const T& operator()(int cb_site, int spin, int color) const
		{
			return _cb_data(cb_site,spin,color);
		}

		inline
		T& operator()(int cb_site, int spin, int color)
		{
			return _cb_data(cb_site,spin,color);
		}

		inline
		const LatticeInfo& GetInfo() const {
			return _info;
		}

		inline
		IndexType GetCB() const {
			return _cb;
		}

		using DataType = Kokkos::View<T**[3],Layout>;


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
		 const T& operator()(int site, int dir, int color1, int color2) const {
			return _cb_gauge_data(site,dir,color1,color2);
		}

		inline
		T& operator()(int site, int dir, int color1,int color2) {
			return _cb_gauge_data(site,dir,color1,color2);
		}

		using DataType = Kokkos::View<T*[4][3][3],Layout>;
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

	template<typename T>
	using HalfSpinorView = typename KokkosCBFineSpinor<T,2>::DataType;

	template<typename T,const int S, const int C>
	struct SiteView {
		T _data[S][C];
		KOKKOS_INLINE_FUNCTION T& operator()(int idx1, int idx2) {
			return _data[idx1][idx2];
		}
		KOKKOS_INLINE_FUNCTION const T& operator()(int idx1, int idx2) const {
			return _data[idx1][idx2];
		}
	};

	template<typename T>
	using SpinorSiteView = SiteView<T,4,3>;

	template<typename T>
	using HalfSpinorSiteView = SiteView<T,2,3>;

	template<typename T>
	using GaugeView = typename KokkosCBFineGaugeField<T>::DataType;

};




#endif /* TEST_KOKKOS_KOKKOS_TYPES_H_ */
