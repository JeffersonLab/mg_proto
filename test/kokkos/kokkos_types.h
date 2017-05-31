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
		const IndexType GetCB() const {
			return _cb;
		}

		using DataType = Kokkos::View<T**[3][2],Layout>;

		inline
		const DataType& GetData() const {
			return _cb_data;
		}

		inline
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

		const IndexType GetCB() const {
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
		KokkosFineGaugeField(const LatticeInfo& info) : _gauge_data({nullptr,nullptr}), _info(info)
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

};




#endif /* TEST_KOKKOS_KOKKOS_TYPES_H_ */
