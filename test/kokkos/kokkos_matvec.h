/*
 * kokkos_matvec.h
 *
 *  Created on: May 26, 2017
 *      Author: bjoo
 */

#ifndef TEST_KOKKOS_MATVEC_H_
#define TEST_KOKKOS_MATVEC_H_

#include "kokkos_constants.h"
#include "kokkos_types.h"

namespace MG
{

	template<typename T>
	KOKKOS_FORCEINLINE_FUNCTION
	void mult_u_halfspinor(const GaugeView<T>& gauge_in,
			const HalfSpinorSiteView<T>& v_in,
			HalfSpinorSiteView<T>& v_out,int i,int dir)
	{

		for(int spin = 0; spin < 2; ++spin) {
			for(int row=0; row < 3; ++row) {
				v_out(spin,row,K_RE)=0;
				v_out(spin,row,K_IM)=0;
				for(int col=0; col < 3; ++col) {

					//v_out(spin,row) += u(row,col)*v_in(col);
					// complex mul:   real_part: a(K_RE)*b(K_RE)-a(K_IM)*b(K_IM)
					//                imag_part: a(K_RE)*b(K_IM) + a(K_IM)*b(K_RE);
					//
					v_out(spin,row,K_RE) += gauge_in(i,dir,row,col,K_RE)*v_in(spin,col,K_RE);
					v_out(spin,row,K_RE) -= gauge_in(i,dir,row,col,K_IM)*v_in(spin,col,K_IM);

					v_out(spin,row,K_IM) += gauge_in(i,dir,row,col,K_RE)*v_in(spin,col,K_IM);
					v_out(spin,row,K_IM) += gauge_in(i,dir,row,col,K_IM)*v_in(spin,col,K_RE);
				}
			}

		}
	}


	template<typename T>
	KOKKOS_FORCEINLINE_FUNCTION
	void mult_adj_u_halfspinor(const GaugeView<T>& gauge_in,
			const HalfSpinorSiteView<T>& v_in,
			HalfSpinorSiteView<T>& v_out, int i, int dir)
	{

		for(int spin = 0; spin < 2; ++spin) {
				for(int row=0; row < 3; ++row) {
					v_out(spin,row,K_RE)=0;
					v_out(spin,row,K_IM)=0;
				}

				for(int col=0; col < 3; ++col) {
					for(int row=0; row < 3; ++row) {
						//v_out(spin,row) += u(row,col)*v_in(col);
						// complex mul:   real_part: a(K_RE)*b(K_RE)-a(K_IM)*b(K_IM)
						//                imag_part: a(K_RE)*b(K_IM) + a(K_IM)*b(K_RE);
						//
						v_out(spin,row,K_RE) += gauge_in(i,dir,col,row,K_RE)*v_in(spin,col,K_RE);
						v_out(spin,row,K_RE) += gauge_in(i,dir,col,row,K_IM)*v_in(spin,col,K_IM);

						v_out(spin,row,K_IM) += gauge_in(i,dir,col,row,K_RE)*v_in(spin,col,K_IM);
						v_out(spin,row,K_IM) -= gauge_in(i,dir,col,row,K_IM)*v_in(spin,col,K_RE);
					}
				}

			}
	}

	template<typename T>
	void KokkosMVLattice(const KokkosCBFineGaugeField<T>& u_in,
			const KokkosCBFineSpinor<T,2>& hspinor_in,
			int dir,
			const KokkosCBFineSpinor<T,2>& hspinor_out)

	{
		int num_sites = u_in.GetInfo().GetNumCBSites();
		HalfSpinorView<T> hspinor_in_view = hspinor_in.GetData();
		GaugeView<T> u = u_in.GetData();
		HalfSpinorView<T> hspinor_out_view = hspinor_out.GetData();

		Kokkos::parallel_for(num_sites,
				KOKKOS_LAMBDA(int i) {

				// Site local workspace...
				HalfSpinorSiteView<T> site_in;
				for(int spin=0; spin < 2; ++spin) {
					for(int color=0; color <3; ++color) {
						for(int reim=0; reim < 2; ++reim) {
							site_in(spin,color,reim) = hspinor_in_view(i,spin,color,reim);
						}
					}
				}
				HalfSpinorSiteView<T> site_out;
				mult_u_halfspinor(u, site_in, site_out, i, dir);

				// Write out
				for(int spin=0; spin < 2; ++spin ) {
					for(int color=0; color < 3; ++color) {
						for(int reim=0; reim < 2; ++reim) {
							hspinor_out_view(i,spin,color,reim) = site_out(spin,color,reim);
						}
					}
				}


		});
	}



	template<typename T>
	void KokkosHVLattice(const KokkosCBFineGaugeField<T>& u_in,
				  const KokkosCBFineSpinor<T,2>& hspinor_in,
				  int dir,
				  const KokkosCBFineSpinor<T,2>& hspinor_out)

	{
		int num_sites = u_in.GetInfo().GetNumCBSites();
		HalfSpinorView<T> hspinor_in_view = hspinor_in.GetData();
		HalfSpinorView<T> hspinor_out_view = hspinor_out.GetData();

		Kokkos::parallel_for(num_sites,
				KOKKOS_LAMBDA(int i) {

			// Site local workspace...
			HalfSpinorSiteView<T> site_in;
			for(int spin=0; spin < 2; ++spin) {
				for(int color=0; color <3; ++color) {
					for(int reim=0; reim < 2; ++reim) {
						site_in(spin,color,reim) = hspinor_in_view(i,spin,color,reim);
					}
				}
			}
			HalfSpinorSiteView<T> site_out;
			mult_adj_u_halfspinor(u_in.GetData(), site_in, site_out, i, dir);

			// Write out
			for(int spin=0; spin < 2; ++spin ) {
				for(int color=0; color < 3; ++color) {
					for(int reim=0; reim < 2; ++reim) {
						hspinor_out_view(i,spin,color,reim) = site_out(spin,color,reim);
					}
				}
			}
		});
	}

}


#endif /* TEST_KOKKOS_MATVEC_H_ */
