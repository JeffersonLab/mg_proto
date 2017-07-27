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
#include "kokkos_ops.h"

namespace MG
{


	template<typename T>
	KOKKOS_FORCEINLINE_FUNCTION
	void mult_u_halfspinor(const GaugeView<Kokkos::complex<T>>& gauge_in,
			const HalfSpinorSiteView<Kokkos::complex<T>>& v_in,
			HalfSpinorSiteView<Kokkos::complex<T>>& v_out,int i,int dir)
	{

		for(int row=0; row < 3; ++row) {

			for(int spin = 0; spin < 2; ++spin) {
				ComplexZero(v_out(row,spin));
			}

			for(int col=0; col < 3; ++col) {
				for(int spin = 0; spin < 2; ++spin) {
					//v_out(row,spin) += u(row,col)*v_in(col);
					// complex mul:   real_part: a(K_RE)*b(K_RE)-a(K_IM)*b(K_IM)
					//                imag_part: a(K_RE)*b(K_IM) + a(K_IM)*b(K_RE);
					//
					ComplexCMadd(v_out(row,spin), gauge_in(i,dir,row,col), v_in(col,spin));
				}
			}
		}


	}


	template<typename T>
	KOKKOS_FORCEINLINE_FUNCTION
	void mult_adj_u_halfspinor(const GaugeView<Kokkos::complex<T>>& gauge_in,
			const HalfSpinorSiteView<Kokkos::complex<T>>& v_in,
			HalfSpinorSiteView<Kokkos::complex<T>>& v_out, int i, int dir)
	{

				for(int row=0; row < 3; ++row) {
					for(int spin = 0; spin < 2; ++spin) {

						ComplexZero(v_out(row,spin));
					}
				}

				for(int col=0; col < 3; ++col) {
					for(int row=0; row < 3; ++row) {
						//v_out(row,spin) += u(row,col)*v_in(col);
						// complex mul:   real_part: a(K_RE)*b(K_RE)-a(K_IM)*b(K_IM)
						//                imag_part: a(K_RE)*b(K_IM) + a(K_IM)*b(K_RE);
						//
#if 0
						v_out(row,spin,K_RE) += gauge_in(i,dir,col,row,K_RE)*v_in(col,spin,K_RE);
						v_out(row,spin,K_RE) += gauge_in(i,dir,col,row,K_IM)*v_in(col,spin,K_IM);

						v_out(row,spin,K_IM) += gauge_in(i,dir,col,row,K_RE)*v_in(col,spin,K_IM);
						v_out(row,spin,K_IM) -= gauge_in(i,dir,col,row,K_IM)*v_in(col,spin,K_RE);
#endif
						for(int spin=0; spin < 2; ++spin) {
							ComplexConjMadd(v_out(row,spin), gauge_in(i,dir,col,row), v_in(col,spin));
						}
					}
				}


	}

	template<typename T>
	void KokkosMVLattice(const KokkosCBFineGaugeField<Kokkos::complex<T>>& u_in,
			const KokkosCBFineSpinor<Kokkos::complex<T>,2>& hspinor_in,
			int dir,
			const KokkosCBFineSpinor<Kokkos::complex<T>,2>& hspinor_out)

	{
		int num_sites = u_in.GetInfo().GetNumCBSites();
		HalfSpinorView<Kokkos::complex<T>> hspinor_in_view = hspinor_in.GetData();
		GaugeView<Kokkos::complex<T>> u = u_in.GetData();
		HalfSpinorView<Kokkos::complex<T>> hspinor_out_view = hspinor_out.GetData();

		Kokkos::parallel_for(num_sites,
				KOKKOS_LAMBDA(int i) {

				// Site local workspace...
				HalfSpinorSiteView<Kokkos::complex<T>> site_in;

				for(int col=0; col <3; ++col) {
					for(int spin=0; spin < 2; ++spin) {
						ComplexCopy(site_in(col,spin), hspinor_in_view(i,col,spin));
					}
				}

				HalfSpinorSiteView<Kokkos::complex<T>> site_out;
				mult_u_halfspinor(u, site_in, site_out, i, dir);

				// Write out
				for(int col=0; col < 3; ++col) {
					for(int spin=0; spin < 2; ++spin ) {
						ComplexCopy(hspinor_out_view(i,col,spin),site_out(col,spin));
					}
				}
		});
	}



	template<typename T>
	void KokkosHVLattice(const KokkosCBFineGaugeField<Kokkos::complex<T>>& u_in,
				  const KokkosCBFineSpinor<Kokkos::complex<T>,2>& hspinor_in,
				  int dir,
				  const KokkosCBFineSpinor<Kokkos::complex<T>,2>& hspinor_out)

	{
		int num_sites = u_in.GetInfo().GetNumCBSites();
		HalfSpinorView<Kokkos::complex<T>> hspinor_in_view = hspinor_in.GetData();
		HalfSpinorView<Kokkos::complex<T>> hspinor_out_view = hspinor_out.GetData();

		Kokkos::parallel_for(num_sites,
				KOKKOS_LAMBDA(int i) {

			// Site local workspace...
			HalfSpinorSiteView<Kokkos::complex<T>> site_in;
			for(int col=0; col <3; ++col) {
				for(int spin=0; spin < 2; ++spin) {
					ComplexCopy(site_in(col,spin), hspinor_in_view(i,col,spin));
				}
			}

			HalfSpinorSiteView<Kokkos::complex<T>> site_out;
			mult_adj_u_halfspinor(u_in.GetData(), site_in, site_out, i, dir);

			// Write out
			for(int col=0; col < 3; ++col) {
				for(int spin=0; spin < 2; ++spin ) {
					ComplexCopy(hspinor_out_view(i,col,spin), site_out(col,spin));
				}
			}
		});
	}

}


#endif /* TEST_KOKKOS_MATVEC_H_ */
