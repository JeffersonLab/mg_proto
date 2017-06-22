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




// One direction of dslash, multiplying U. Direction and whether to accumulate
// are 'compile time' through templates.
// Savvy compiler should inline these and optimize them
template<typename T,int dir, bool accum>
KOKKOS_FORCEINLINE_FUNCTION
void DslashMVSiteDir(const SpinorView<T>& spinor_in,    // Neighbor spinor
		const GaugeView<T>& gauge_in,     // Gauge Link
		SpinorView<T>& spinor_out,         // result
		int plus_minus,
		int source_spinor_idx,
		int gauge_idx,
		int dest_spinor_idx)                // Sign +1 for Dslash, -1 for Adjoint

{
	HalfSpinorSiteView<T> proj_res;
	HalfSpinorSiteView<T> mult_proj_res;

	KokkosProjectDir<T,dir>(spinor_in,plus_minus,proj_res,source_spinor_idx);
	mult_u_halfspinor(gauge_in,proj_res,mult_proj_res,gauge_idx,dir);
	KokkosReconsDir<T,dir,accum>(mult_proj_res,plus_minus,spinor_out,dest_spinor_idx);
}

// One direction of dslash, multiply with U^\dagger
// direction and whether to accumulate are compile time through templates
// Savvy compiler should inline and optimize them
template<typename T,  int dir,  bool accum >
KOKKOS_FORCEINLINE_FUNCTION
void DslashHVSiteDir(const SpinorView<T>& spinor_in,   // Neighbor spinor
		const GaugeView<T>& gauge_in,    // Gauge link
		SpinorView<T>& spinor_out,        // result
		int plus_minus,
		int source_spinor_idx,
		int gauge_idx,
		int dest_spinor_idx)         // sign +1 for Dslash, -1 for Adjoint

{
	HalfSpinorSiteView<T> proj_res;
	HalfSpinorSiteView<T> mult_proj_res;

	KokkosProjectDir<T,dir>(spinor_in, plus_minus, proj_res,source_spinor_idx);
	mult_adj_u_halfspinor(gauge_in,proj_res,mult_proj_res,gauge_idx,dir);
	KokkosReconsDir<T,dir,accum>(mult_proj_res,plus_minus,spinor_out,dest_spinor_idx);

}


template<typename T>
class KokkosDslash {
private:
	const LatticeInfo& _info;


public:
	KokkosDslash(const LatticeInfo& info) : _info(info) {}

	void operator()(const KokkosCBFineSpinor<T,4>& fine_in,
			const KokkosFineGaugeField<T>& gauge_in,
			KokkosCBFineSpinor<T,4>& fine_out,
			int plus_minus) const
	{
		// Source and target checkerboards
		IndexType target_cb = fine_out.GetCB();
		IndexType source_cb = (target_cb == EVEN) ? ODD : EVEN;
		int minus_plus = -plus_minus;

		// Gather all views just outside Parallel Loop.
		// Can these be references? Will the world collapse?

		const SpinorView<T>& s_in = fine_in.GetData();
		const GaugeView<T>& g_in_src_cb = (gauge_in(source_cb)).GetData();
		const GaugeView<T>&  g_in_target_cb = (gauge_in(target_cb)).GetData();
		SpinorView<T>& s_o = fine_out.GetData();

		IndexArray latdims=_info.GetCBLatticeDimensions();
		const int _n_xh = latdims[0];
		const int _n_x = 2*_n_xh;
		const int _n_y = latdims[1];
		const int _n_z = latdims[2];
		const int _n_t = latdims[3];
		const int _num_sites = _n_xh*_n_y*_n_z*_n_t;

		// Parallel site loop
		Kokkos::parallel_for(_num_sites,
				KOKKOS_LAMBDA(int site) {

			// Source and target checkerboard gauge fields

			SpinorView<T> s_o = fine_out.GetData();
			// Break down site index into xcb, y,z and t
			IndexType tmp_yzt = site / _n_xh;
			IndexType xcb = site - _n_xh * tmp_yzt;
			IndexType tmp_zt = tmp_yzt / _n_y;
			IndexType y = tmp_yzt - _n_y * tmp_zt;
			IndexType t = tmp_zt / _n_z;
			IndexType z = tmp_zt - _n_z * t;

			// Global, uncheckerboarded x, assumes cb = (x + y + z + t ) & 1
			IndexType x = 2*xcb + ((target_cb+y+z+t)&0x1);


			// Find index of neighbor
			int neigh_index=0;


			// T - minus
			{
				if( t > 0 ) {
					neigh_index = xcb + _n_xh*(y + _n_y*(z + _n_z*(t-1)));
				}
				else {
					neigh_index = xcb + _n_xh*(y + _n_y*(z + _n_z*(_n_t-1)));
				}


				DslashHVSiteDir<T,3,false>(s_in,
									       g_in_src_cb,
										   s_o,
										   plus_minus,
										   neigh_index,
										   neigh_index,
										   site);
			}



			// Z - minus
			{
				if( z > 0 ) {
					neigh_index = xcb + _n_xh*(y + _n_y*((z-1) + _n_z*t));
				}
				else {
					neigh_index = xcb + _n_xh*(y + _n_y*((_n_z-1) + _n_z*t));
				}
				DslashHVSiteDir<T,2,true>(s_in,
										       g_in_src_cb,
											   s_o,
											   plus_minus,
											   neigh_index,
											   neigh_index,
											   site);
			}

			// Y - minus
			{
				if( y > 0 ) {
					neigh_index = xcb + _n_xh*((y-1) + _n_y*(z + _n_z*t));
				}
				else {
					neigh_index = xcb + _n_xh*((_n_y-1) + _n_y*(z + _n_z*t));
				}

				DslashHVSiteDir<T,1,true>(s_in,
													       g_in_src_cb,
														   s_o,
														   plus_minus,
														   neigh_index,
														   neigh_index,
														   site);

			}

			// X - minus
			{
				if ( x > 0 ) {
					neigh_index= ((x-1)/2) + _n_xh*(y + _n_y*(z + _n_z*t));
				}
				else {
					neigh_index= ((_n_x-1)/2) + _n_xh*(y + _n_y*(z + _n_z*t));
				}

				DslashHVSiteDir<T,0,true>(s_in,
													       g_in_src_cb,
														   s_o,
														   plus_minus,
														   neigh_index,
														   neigh_index,
														   site);


			}


			// X - plus
			{
				if ( x < _n_x - 1 ) {
					neigh_index= ((x+1)/2)  + _n_xh*(y + _n_y*(z + _n_z*t));
				}
				else {
					neigh_index = 0 + _n_xh*(y + _n_y*(z + _n_z*t));
				}


				DslashMVSiteDir<T,0,true>(s_in,
						g_in_target_cb,
						s_o,
						minus_plus,
						neigh_index,
						site,
						site);

			}


			// Y - plus
			{
				if( y < _n_y-1 ) {
					neigh_index = xcb + _n_xh*((y+1) + _n_y*(z + _n_z*t));
				}
				else {
					neigh_index = xcb + _n_xh*(0 + _n_y*(z + _n_z*t));
				}

				DslashMVSiteDir<T,1,true>(s_in,
						g_in_target_cb,
						s_o,
						minus_plus,
						neigh_index,
						site,
						site);
			}

			// Z - plus
			{
				if( z < _n_z-1 ) {
					neigh_index = xcb + _n_xh*(y + _n_y*((z+1) + _n_z*t));
				}
				else {
					neigh_index = xcb + _n_xh*(y + _n_y*(0 + _n_z*t));
				}




				DslashMVSiteDir<T,2,true>(s_in,
						g_in_target_cb,
						s_o,
						minus_plus,
						neigh_index,
						site,
						site);

			}


			// T - plus
			{
				if( t < _n_t-1 ) {
					neigh_index = xcb + _n_xh*(y + _n_y*(z + _n_z*(t+1)));
				}
				else {
					neigh_index = xcb + _n_xh*(y + _n_y*(z + _n_z*(0)));
				}


				DslashMVSiteDir<T,3,true>(s_in,
						g_in_target_cb,
						s_o,
						minus_plus,
						neigh_index,
						site,
						site);

			}


		});

	}
};

};




#endif /* TEST_KOKKOS_KOKKOS_DSLASH_H_ */
