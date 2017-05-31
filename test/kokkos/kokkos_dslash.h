/*
 * kokkos_dslash.h
 *
 *  Created on: May 30, 2017
 *      Author: bjoo
 */

#ifndef TEST_KOKKOS_KOKKOS_DSLASH_H_
#define TEST_KOKKOS_KOKKOS_DSLASH_H_

#include "kokkos_types.h"
#include "kokkos_spinproj.h"
#include "kokkos_matvec.h"

namespace MG {

	template<typename T>
	using Spinor = Kokkos::View<T[4][3][2]>;

	template<typename T>
	using Matrix = Kokkos::View<T[3][3][2]>;

	template<typename T>
	using HalfSpinor = Kokkos::View<T[2][3][2]>;


	template<typename T, const int dir, const bool accum>
	KOKKOS_INLINE_FUNCTION
	void DslashMVSiteDir(const Spinor<T>& spinor_in,
				   	   	 const Matrix<T>& gauge_in,
						 HalfSpinor<T> proj_res,
						 HalfSpinor<T> mult_proj_res,
						 Spinor<T>& spinor_out,
						 int plus_minus)

	{
		KokkosProjectDir<T,dir>(spinor_in,plus_minus,proj_res);
		mult_u_halfspinor(gauge_in,proj_res,mult_proj_res);
		KokkosReconsDir<T,dir,accum>(mult_proj_res,plus_minus,spinor_out);
	}

	template<typename T, const int dir, const bool accum >
	KOKKOS_INLINE_FUNCTION
	void DslashHVSiteDir(const Spinor<T>& spinor_in,
				   	   	 const Matrix<T>& gauge_in,
						 HalfSpinor<T> proj_res,
						 HalfSpinor<T> mult_proj_res,
						 Spinor<T>& spinor_out,
						 int plus_minus)

	{

		// Unify conventions for sign. Plus minus (int) or T
		KokkosProjectDir<T,dir>(spinor_in, plus_minus, proj_res);
		mult_adj_u_halfspinor(gauge_in,proj_res,mult_proj_res);
		KokkosReconsDir<T,dir,accum>(mult_proj_res,plus_minus,spinor_out);
	}




	template<typename T>
	void Dslash(const KokkosCBFineSpinor<T,4>& fine_in,
			const KokkosFineGaugeField<T>& gauge_in,
			KokkosCBFineSpinor<T,4>& fine_out,
			int plus_minus)
	{
		const IndexType target_cb = fine_out.GetCB();
		const IndexType source_cb = (target_cb == EVEN) ? ODD : EVEN;

		const KokkosCBFineGaugeField<T>& g_src_cb = gauge_in(source_cb);
		const KokkosCBFineGaugeField<T>& g_target_cb = gauge_in(target_cb);


		const LatticeInfo& info = fine_out.GetInfo();
		const int num_sites = info.GetNumCBSites();

		const IndexArray& latdims = info.GetCBLatticeDimensions();
		const IndexType _n_xh = latdims[0];
		const IndexType _n_x = 2*_n_xh;
		const IndexType _n_y = latdims[1];
		const IndexType _n_z = latdims[2];
		const IndexType _n_t = latdims[3];

		Kokkos::parallel_for(num_sites,
				KOKKOS_LAMBDA(int site) {

			// Break down site index into xcb, y,z and t
			IndexType tmp_yzt = site / _n_xh;
			IndexType xcb = site - _n_xh * tmp_yzt;
			IndexType tmp_zt = tmp_yzt / _n_y;
			IndexType y = tmp_yzt - _n_y * tmp_zt;
			IndexType t = tmp_zt / _n_z;
			IndexType z = tmp_zt - _n_z * t;
			IndexType x = 2*xcb + ((target_cb+y+z+t)&0x1);  // Global X

			Spinor<T> result("result");
			HalfSpinor<T> proj_result("proj_result");
			HalfSpinor<T> matmult_result("matmult_result");
			// adj(u[3](x-T)*projdDir3Plus(psi(x-T))
			int neigh_index=0;

			Spinor<T> result_site = Kokkos::subview(fine_out.GetData(),site, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );

			// T - minus
			{
				if( t > 0 ) {
					neigh_index = xcb + _n_xh*(y + _n_y*(z + _n_z*(t-1)));
				}
				else {
					neigh_index = xcb + _n_xh*(y + _n_y*(z + _n_z*(_n_t-1)));
				}

				const Spinor<T>& spinor_t_minus = Kokkos::subview(fine_in.GetData(),
						neigh_index, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

				const Matrix<T>& gauge_t_minus = Kokkos::subview(g_src_cb.GetData(),neigh_index,3,
						Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);


				DslashHVSiteDir<T,3,false>(spinor_t_minus,gauge_t_minus,proj_result,matmult_result,result,plus_minus);
			}

			// Z - minus
			{
				if( z > 0 ) {
					neigh_index = xcb + _n_xh*(y + _n_y*((z-1) + _n_z*t));
				}
				else {
					neigh_index = xcb + _n_xh*(y + _n_y*((_n_z-1) + _n_z*t));
				}

				const Spinor<T>& spinor_z_minus = Kokkos::subview(fine_in.GetData(),
						neigh_index, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

				const Matrix<T>& gauge_z_minus = Kokkos::subview(g_src_cb.GetData(),neigh_index,2,
						Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);


				DslashHVSiteDir<T,2,true>(spinor_z_minus,gauge_z_minus,proj_result,matmult_result,result,plus_minus);
			}

			// Y - minus
			{
				if( y > 0 ) {
					neigh_index = xcb + _n_xh*((y-1) + _n_y*(z + _n_z*t));
				}
				else {
					neigh_index = xcb + _n_xh*((_n_y-1) + _n_y*(z + _n_z*t));
				}

				const Spinor<T>& spinor_y_minus = Kokkos::subview(fine_in.GetData(),
						neigh_index, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

				const Matrix<T>& gauge_y_minus = Kokkos::subview(g_src_cb.GetData(),neigh_index,1,
						Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);


				DslashHVSiteDir<T,1,true>(spinor_y_minus,gauge_y_minus,proj_result,matmult_result,result,plus_minus);
			}

			// X - minus
			{
				if ( x > 0 ) {
					neigh_index= ((x-1)/2) + _n_xh*(y + _n_y*(z + _n_z*t));
				}
				else {
					neigh_index= ((_n_x-1)/2) + _n_xh*(y + _n_y*(z + _n_z*t));
				}

				const Spinor<T>& spinor_x_minus = Kokkos::subview(fine_in.GetData(),
						neigh_index, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

				const Matrix<T>& gauge_x_minus = Kokkos::subview(g_src_cb.GetData(),neigh_index,0,
						Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);


				DslashHVSiteDir<T,0,true>(spinor_x_minus,gauge_x_minus,proj_result,matmult_result,result,plus_minus);
			}


			// X - plus
			{
				if ( x < _n_x - 1 ) {
					neigh_index= ((x+1)/2)  + _n_xh*(y + _n_y*(z + _n_z*t));
				}
				else {
					neigh_index = 0 + _n_xh*(y + _n_y*(z + _n_z*t));
				}

				const Spinor<T>& spinor_x_plus = Kokkos::subview(fine_in.GetData(),
						neigh_index, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

				const Matrix<T>& gauge_x_plus = Kokkos::subview(g_target_cb.GetData(),site,0,
						Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);


				DslashMVSiteDir<T,0,true>(spinor_x_plus,gauge_x_plus,proj_result,matmult_result,result,-plus_minus);
			}
			// Y - plus
			{
				if( y < _n_y-1 ) {
					neigh_index = xcb + _n_xh*((y+1) + _n_y*(z + _n_z*t));
				}
				else {
					neigh_index = xcb + _n_xh*(0 + _n_y*(z + _n_z*t));
				}

				const Spinor<T>& spinor_y_plus = Kokkos::subview(fine_in.GetData(),
						neigh_index, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

				const Matrix<T>& gauge_y_plus = Kokkos::subview(g_target_cb.GetData(),site,1,
						Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);


				DslashMVSiteDir<T,1,true>(spinor_y_plus,gauge_y_plus,proj_result,matmult_result,result,-plus_minus);
			}

			// Z - plus
			{
				if( z < _n_z-1 ) {
					neigh_index = xcb + _n_xh*(y + _n_y*((z+1) + _n_z*t));
				}
				else {
					neigh_index = xcb + _n_xh*(y + _n_y*(0 + _n_z*t));
				}

				const Spinor<T>& spinor_z_plus = Kokkos::subview(fine_in.GetData(),
						neigh_index, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

				const Matrix<T>& gauge_z_plus = Kokkos::subview(g_target_cb.GetData(),site,2,
						Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);


				DslashMVSiteDir<T,2,true>(spinor_z_plus,gauge_z_plus,proj_result,matmult_result,result,-plus_minus);
			}


			// T - plus
			{
				if( t < _n_t-1 ) {
					neigh_index = xcb + _n_xh*(y + _n_y*(z + _n_z*(t+1)));
				}
				else {
					neigh_index = xcb + _n_xh*(y + _n_y*(z + _n_z*(0)));
				}

				const Spinor<T>& spinor_t_plus = Kokkos::subview(fine_in.GetData(),
						neigh_index, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

				const Matrix<T>& gauge_t_plus = Kokkos::subview(g_target_cb.GetData(),site,3,
						Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);


				DslashMVSiteDir<T,3,true>(spinor_t_plus,gauge_t_plus,proj_result,matmult_result,result,-plus_minus);
			}

			// Write result to memory.
			for(int spin=0; spin < 4; ++spin ) {
				for(int color=0; color < 3; ++color) {
					for(int reim=0; reim < 2; ++reim) {
						result_site(spin,color,reim) = result(spin,color,reim);
					}
				}
			}
		});


	}
};




#endif /* TEST_KOKKOS_KOKKOS_DSLASH_H_ */
