/*
 * kokkos_matvec.h
 *
 *  Created on: May 26, 2017
 *      Author: bjoo
 */

#ifndef TEST_KOKKOS_MATVEC_H_
#define TEST_KOKKOS_MATVEC_H_

#include "kokkos_types.h"

namespace MG
{
	template<typename T>
	KOKKOS_INLINE_FUNCTION
	void mult_u_halfspinor(const Kokkos::View<T[3][3][2]>& u,
			const Kokkos::View<T[2][3][2]>& v_in,
			Kokkos::View<T[2][3][2]>& v_out)
	{
		for(int spin = 0; spin < 2; ++spin) {
			for(int row=0; row < 3; ++row) {
				v_out(spin,row,RE)=0;
				v_out(spin,row,IM)=0;
				for(int col=0; col < 3; ++col) {

					//v_out(spin,row) += u(row,col)*v_in(col);
					// complex mul:   real_part: a(RE)*b(RE)-a(IM)*b(IM)
					//                imag_part: a(RE)*b(IM) + a(IM)*b(RE);
					//
					v_out(spin,row,RE) += u(row,col,RE)*v_in(spin,col,RE);
					v_out(spin,row,RE) -= u(row,col,IM)*v_in(spin,col,IM);

					v_out(spin,row,IM) += u(row,col,RE)*v_in(spin,col,IM);
					v_out(spin,row,IM) += u(row,col,IM)*v_in(spin,col,RE);
				}
			}

		}
	}

	template<typename T>
	void KokkosMV(const KokkosCBFineGaugeField<T>& u_in,
				  const KokkosCBFineSpinor<T,2>& hspinor_in,
				  int dir,
				  const KokkosCBFineSpinor<T,2>& hspinor_out)

	{
		int num_sites = u_in.GetInfo().GetNumCBSites();

		Kokkos::parallel_for(num_sites,
				KOKKOS_LAMBDA(int i) {

				const Kokkos::View<T[3][3][2]> link=Kokkos::subview(u_in.GetData(),i,dir,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
				const Kokkos::View<T[2][3][2]> in = Kokkos::subview(hspinor_in.GetData(),i,Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
				Kokkos::View<T[2][3][2]> out = Kokkos::subview(hspinor_out.GetData(), i, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
				mult_u_halfspinor(link, in, out);


		});
	}

	template<typename T>
	KOKKOS_INLINE_FUNCTION
	void mult_adj_u_halfspinor(const Kokkos::View<T[3][3][2]>& u,
			const Kokkos::View<T[2][3][2]>& v_in,
			Kokkos::View<T[2][3][2]>& v_out)
	{
		for(int spin = 0; spin < 2; ++spin) {
				for(int row=0; row < 3; ++row) {
					v_out(spin,row,RE)=0;
					v_out(spin,row,IM)=0;
				}

				for(int col=0; col < 3; ++col) {
					for(int row=0; row < 3; ++row) {
						//v_out(spin,row) += u(row,col)*v_in(col);
						// complex mul:   real_part: a(RE)*b(RE)-a(IM)*b(IM)
						//                imag_part: a(RE)*b(IM) + a(IM)*b(RE);
						//
						v_out(spin,row,RE) += u(col,row,RE)*v_in(spin,col,RE);
						v_out(spin,row,RE) += u(col,row,IM)*v_in(spin,col,IM);

						v_out(spin,row,IM) += u(col,row,RE)*v_in(spin,col,IM);
						v_out(spin,row,IM) -= u(col,row,IM)*v_in(spin,col,RE);
					}
				}

			}
	}


	template<typename T>
	void KokkosHV(const KokkosCBFineGaugeField<T>& u_in,
				  const KokkosCBFineSpinor<T,2>& hspinor_in,
				  int dir,
				  const KokkosCBFineSpinor<T,2>& hspinor_out)

	{
		int num_sites = u_in.GetInfo().GetNumCBSites();

		Kokkos::parallel_for(num_sites,
				KOKKOS_LAMBDA(int i) {

				const Kokkos::View<T[3][3][2]> link=Kokkos::subview(u_in.GetData(),i,dir,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
				const Kokkos::View<T[2][3][2]> in = Kokkos::subview(hspinor_in.GetData(),i,Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
				Kokkos::View<T[2][3][2]> out = Kokkos::subview(hspinor_out.GetData(), i, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
				mult_adj_u_halfspinor(link, in, out);


		});
	}

}


#endif /* TEST_KOKKOS_MATVEC_H_ */
