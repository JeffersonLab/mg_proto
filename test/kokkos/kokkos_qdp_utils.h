/*
 * kokkos_qdp_utils.h
 *
 *  Created on: May 23, 2017
 *      Author: bjoo
 */

#ifndef TEST_KOKKOS_KOKKOS_QDP_UTILS_H_
#define TEST_KOKKOS_KOKKOS_QDP_UTILS_H_

#include "qdp.h"
#include "kokkos_types.h"
#include <Kokkos_Core.hpp>

#include <utils/print_utils.h>

namespace MG
{
	template<typename T>
	void
	QDPLatticeFermionToKokkosCBSpinor(const QDP::LatticeFermion& qdp_in,
			KokkosCBFineSpinor<T,4>& kokkos_out)
	{
		auto cb = kokkos_out.GetCB();
		const QDP::Subset& sub = ( cb == EVEN ) ? QDP::rb[0] : QDP::rb[1];

		// Check conformance:
		int num_sites=static_cast<int>(kokkos_out.GetInfo().GetNumCBSites());

		if ( sub.numSiteTable() != num_sites ) {
			MasterLog(ERROR, "%s QDP++ Spinor has different number of sites per checkerboard than the KokkosCBFineSpinor",
					__FUNCTION__);
		}
		auto h_out = Kokkos::create_mirror_view( kokkos_out.GetData() );

		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_sites),
				[=](int i) {
					for(int spin=0; spin < 4; ++spin) {
						for(int color=0; color < 3; ++color) {
							const int qdp_index = sub.siteTable()[i];

							h_out(i,spin,color,RE)= qdp_in.elem(qdp_index).elem(spin).elem(color).real();
							h_out(i,spin,color,IM)= qdp_in.elem(qdp_index).elem(spin).elem(color).imag();

						} // color
					} // spin
			}// kokkos lambda
		);

		Kokkos::deep_copy(kokkos_out.GetData(), h_out);
	}

	template<typename T>
	void
	KokkosCBSpinorToQDPLatticeFermion(const KokkosCBFineSpinor<T,4>& kokkos_in,
			QDP::LatticeFermion& qdp_out) {

		auto cb = kokkos_in.GetCB();
		const QDP::Subset& sub = ( cb == EVEN ) ? QDP::rb[0] : QDP::rb[1];

		// Check conformance:
		int num_sites=static_cast<int>(kokkos_in.GetInfo().GetNumCBSites());

		if ( sub.numSiteTable() != num_sites ) {
			MasterLog(ERROR, "%s: QDP++ Spinor has different number of sites per checkerboard than the KokkosCBFineSpinor",
					__FUNCTION__);
		}

		auto h_in = Kokkos::create_mirror_view( kokkos_in.GetData() );
		Kokkos::deep_copy( h_in, kokkos_in.GetData() );

		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_sites),
				[&](int i) {
					for(int spin=0; spin < 4; ++spin) {
						for(int color=0; color < 3; ++color) {
							const int qdp_index = sub.siteTable()[i];
							qdp_out.elem(qdp_index).elem(spin).elem(color).real() = h_in(i,spin,color,RE);
							qdp_out.elem(qdp_index).elem(spin).elem(color).imag() = h_in(i,spin,color,IM);
						} // color
					} // spin
			}// kokkos lambda
		);


	}

	template<typename T>
	void
	QDPLatticeHalfFermionToKokkosCBSpinor2(const QDP::LatticeHalfFermion& qdp_in,
			KokkosCBFineSpinor<T,2>& kokkos_out)
	{
		auto cb = kokkos_out.GetCB();
		const QDP::Subset& sub = ( cb == EVEN ) ? QDP::rb[0] : QDP::rb[1];

		// Check conformance:
		int num_sites=static_cast<int>(kokkos_out.GetInfo().GetNumCBSites());

		if ( sub.numSiteTable() != num_sites ) {
			MasterLog(ERROR, "%s QDP++ Spinor has different number of sites per checkerboard than the KokkosCBFineSpinor",
					__FUNCTION__);
		}
		auto h_out = Kokkos::create_mirror_view( kokkos_out.GetData() );

		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_sites),
				[=](int i) {
					for(int spin=0; spin < 2; ++spin) {
						for(int color=0; color < 3; ++color) {
							const int qdp_index = sub.siteTable()[i];

							h_out(i,spin,color,RE)= qdp_in.elem(qdp_index).elem(spin).elem(color).real();
							h_out(i,spin,color,IM)= qdp_in.elem(qdp_index).elem(spin).elem(color).imag();

						} // color
					} // spin
			}// kokkos lambda
		);

		Kokkos::deep_copy(kokkos_out.GetData(), h_out);
	}

	template<typename T>
	void
	KokkosCBSpinor2ToQDPLatticeHalfFermion(const KokkosCBFineSpinor<T,2>& kokkos_in,
			QDP::LatticeHalfFermion& qdp_out) {

		auto cb = kokkos_in.GetCB();
		const QDP::Subset& sub = ( cb == EVEN ) ? QDP::rb[0] : QDP::rb[1];

		// Check conformance:
		int num_sites=static_cast<int>(kokkos_in.GetInfo().GetNumCBSites());

		if ( sub.numSiteTable() != num_sites ) {
			MasterLog(ERROR, "%s: QDP++ Spinor has different number of sites per checkerboard than the KokkosCBFineSpinor",
					__FUNCTION__);
		}

		auto h_in = Kokkos::create_mirror_view( kokkos_in.GetData() );
		Kokkos::deep_copy( h_in, kokkos_in.GetData() );

		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_sites),
				[&](int i) {
					for(int spin=0; spin < 2; ++spin) {
						for(int color=0; color < 3; ++color) {
							const int qdp_index = sub.siteTable()[i];
							qdp_out.elem(qdp_index).elem(spin).elem(color).real() = h_in(i,spin,color,RE);
							qdp_out.elem(qdp_index).elem(spin).elem(color).imag() = h_in(i,spin,color,IM);
						} // color
					} // spin
			}// kokkos lambda
		);


	}

	template<typename T>
	void
	QDPGaugeFieldToKokkosCBGaugeField(const QDP::multi1d<QDP::LatticeColorMatrix>& qdp_in,
			KokkosCBFineGaugeField<T>& kokkos_out)
	{
		auto cb = kokkos_out.GetCB();
		const QDP::Subset& sub = ( cb == EVEN ) ? QDP::rb[0] : QDP::rb[1];

		// Check conformance:
		int num_sites=static_cast<int>(kokkos_out.GetInfo().GetNumCBSites());

		if ( sub.numSiteTable() != num_sites ) {
			MasterLog(ERROR, "%s QDP++ Gauge has different number of sites per checkerboard than the KokkosCBFineSpinor",
					__FUNCTION__);
		}
		if ( qdp_in.size() != n_dim ) {
			MasterLog(ERROR, "%s QDP++ Gauge has wrong number of dimensions (%d instead of %d)",
						__FUNCTION__, qdp_in.size(), n_dim);
		}

		auto h_out = Kokkos::create_mirror_view( kokkos_out.GetData() );

		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_sites),
				[=](int i) {
					for(int mu=0; mu< 4; ++mu) {
						for(int color=0; color < 3; ++color) {
							for(int color2=0; color2 < 3; ++color2) {
								const int qdp_index = sub.siteTable()[i];

								h_out(i,mu,color,color2,RE)= (qdp_in[mu]).elem(qdp_index).elem().elem(color,color2).real();
								h_out(i,mu,color,color2,IM)= (qdp_in[mu]).elem(qdp_index).elem().elem(color,color2).imag();
							} //color2
						} // color
					} // mu
			}// kokkos lambda
		);

		Kokkos::deep_copy(kokkos_out.GetData(), h_out);
	}

	template<typename T>
	void
	KokkosCBGaugeFieldToQDPGaugeField(const KokkosCBFineGaugeField<T>& kokkos_in,
			QDP::multi1d<QDP::LatticeColorMatrix>& qdp_out)
	{
		auto cb = kokkos_in.GetCB();
		const QDP::Subset& sub = ( cb == EVEN ) ? QDP::rb[0] : QDP::rb[1];

		// Check conformance:
		int num_sites=static_cast<int>(kokkos_in.GetInfo().GetNumCBSites());

		if ( sub.numSiteTable() != num_sites ) {
			MasterLog(ERROR, "%s QDP++ Gauge has different number of sites per checkerboard than the KokkosCBFineSpinor",
					__FUNCTION__);
		}
		if ( qdp_out.size() != n_dim ) {
			qdp_out.resize(n_dim);
		}

		auto h_in = Kokkos::create_mirror_view( kokkos_in.GetData() );
		Kokkos::deep_copy(kokkos_in.GetData(), h_in);

		for(int mu=0; mu < 4; ++mu ) {
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_sites),
				[&](int i) {
			       for(int color=0; color < 3; ++color) {
			    	   for(int color2=0; color2 < 3; ++color2) {
			    		   const int qdp_index = sub.siteTable()[i];


			    		   (qdp_out[mu]).elem(qdp_index).elem().elem(color,color2).real()=
			    				   h_in(i,mu,color,color2,RE);
			    		   (qdp_out[mu]).elem(qdp_index).elem().elem(color,color2).imag()=
									h_in(i,mu,color,color2,IM);
						} //color2
					} // color

			   }// kokkos lambda
		);

		} // mu

	}

	template<typename T>
	void
	QDPGaugeFieldToKokkosGaugeField(const QDP::multi1d<QDP::LatticeColorMatrix>& qdp_in,
									KokkosFineGaugeField<T>& kokkos_out)
	{
		QDPGaugeFieldToKokkosCBGaugeField( qdp_in, kokkos_out(EVEN));
		QDPGaugeFieldToKokkosCBGaugeField( qdp_in, kokkos_out(ODD));
	}

	template<typename T>
	void
	KokkosGaugeFieldToQDPGaugeField(const KokkosFineGaugeField<T>& kokkos_in,
									QDP::multi1d<QDP::LatticeColorMatrix>& qdp_out)
	{
		KokkosCBGaugeFieldToQDPGaugeField( kokkos_in(EVEN),qdp_out);
		KokkosCBGaugeFieldToQDPGaugeField( kokkos_in(ODD), qdp_out);
	}

}




#endif /* TEST_KOKKOS_KOKKOS_QDP_UTILS_H_ */
