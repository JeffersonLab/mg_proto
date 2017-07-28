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
  template<typename T, int isign>
KOKKOS_FORCEINLINE_FUNCTION
void DslashMVSiteDir0(const SpinorView<Kokkos::complex<T>>& spinor_in,    // Neighbor spinor
		      const GaugeView<Kokkos::complex<T>>& gauge_in,     // Gauge Link
		      SpinorSiteView<Kokkos::complex<T>>& res_sum,         // result
		      int source_spinor_idx,
		      int gauge_idx)
{
	HalfSpinorSiteView<Kokkos::complex<T>> proj_res;
	HalfSpinorSiteView<Kokkos::complex<T>> mult_proj_res;

	KokkosProjectDir0<T,isign>(spinor_in,proj_res,source_spinor_idx);
	mult_u_halfspinor(gauge_in,proj_res,mult_proj_res,gauge_idx,0);
	KokkosRecons23Dir0<T,isign>(mult_proj_res, res_sum);
}

  template<typename T, int isign>
KOKKOS_FORCEINLINE_FUNCTION
void DslashMVSiteDir1(const SpinorView<Kokkos::complex<T>>& spinor_in,    // Neighbor spinor
		      const GaugeView<Kokkos::complex<T>>& gauge_in,     // Gauge Link
		      SpinorSiteView<Kokkos::complex<T>>& res_sum,         // result
		      int source_spinor_idx,
		      int gauge_idx)
    
{
	HalfSpinorSiteView<Kokkos::complex<T>> proj_res;
	HalfSpinorSiteView<Kokkos::complex<T>> mult_proj_res;

	KokkosProjectDir1<T,isign>(spinor_in,proj_res,source_spinor_idx);
	mult_u_halfspinor(gauge_in,proj_res,mult_proj_res,gauge_idx,1);
	KokkosRecons23Dir1<T,isign>(mult_proj_res, res_sum);
}

  template<typename T, int isign>
KOKKOS_FORCEINLINE_FUNCTION
void DslashMVSiteDir2(const SpinorView<Kokkos::complex<T>>& spinor_in,    // Neighbor spinor
		      const GaugeView<Kokkos::complex<T>>& gauge_in,     // Gauge Link
		      SpinorSiteView<Kokkos::complex<T>>& res_sum,         // result
		      int source_spinor_idx,
		      int gauge_idx)
{
	HalfSpinorSiteView<Kokkos::complex<T>> proj_res;
	HalfSpinorSiteView<Kokkos::complex<T>> mult_proj_res;

	KokkosProjectDir2<T,isign>(spinor_in,proj_res,source_spinor_idx);
	mult_u_halfspinor(gauge_in,proj_res,mult_proj_res,gauge_idx,2);
	KokkosRecons23Dir2<T,isign>(mult_proj_res, res_sum);
}

  template<typename T, int isign>
KOKKOS_FORCEINLINE_FUNCTION
void DslashMVSiteDir3(const SpinorView<Kokkos::complex<T>>& spinor_in,    // Neighbor spinor
		      const GaugeView<Kokkos::complex<T>>& gauge_in,     // Gauge Link
		      SpinorSiteView<Kokkos::complex<T>>& res_sum,         // result
		      int source_spinor_idx,
		      int gauge_idx)
{
	HalfSpinorSiteView<Kokkos::complex<T>> proj_res;
	HalfSpinorSiteView<Kokkos::complex<T>> mult_proj_res;

	KokkosProjectDir3<T,isign>(spinor_in,proj_res,source_spinor_idx);
	mult_u_halfspinor(gauge_in,proj_res,mult_proj_res,gauge_idx,3);
	KokkosRecons23Dir3<T,isign>(mult_proj_res, res_sum);
}

// One direction of dslash, multiply with U^\dagger
// direction and whether to accumulate are compile time through templates
// Savvy compiler should inline and optimize them
  template<typename T, int isign>
KOKKOS_FORCEINLINE_FUNCTION
void DslashHVSiteDir0(const SpinorView<Kokkos::complex<T>>& spinor_in,   // Neighbor spinor
		      const GaugeView<Kokkos::complex<T>>& gauge_in,    // Gauge link
		      SpinorSiteView<Kokkos::complex<T>>& res_sum,        // result
		      int source_spinor_idx,
		      int gauge_idx)
{
	HalfSpinorSiteView<Kokkos::complex<T>> proj_res;
	HalfSpinorSiteView<Kokkos::complex<T>> mult_proj_res;

	KokkosProjectDir0<T,isign>(spinor_in, proj_res,source_spinor_idx);
	mult_adj_u_halfspinor(gauge_in,proj_res,mult_proj_res,gauge_idx,0);
	KokkosRecons23Dir0<T,isign>(mult_proj_res,res_sum);

}

  template<typename T, int isign>
KOKKOS_FORCEINLINE_FUNCTION
void DslashHVSiteDir1(const SpinorView<Kokkos::complex<T>>& spinor_in,   // Neighbor spinor
		      const GaugeView<Kokkos::complex<T>>& gauge_in,    // Gauge link
		      SpinorSiteView<Kokkos::complex<T>>& res_sum,        // result
		      int source_spinor_idx,
		      int gauge_idx)
{
	HalfSpinorSiteView<Kokkos::complex<T>> proj_res;
	HalfSpinorSiteView<Kokkos::complex<T>> mult_proj_res;

	KokkosProjectDir1<T,isign>(spinor_in, proj_res,source_spinor_idx);
	mult_adj_u_halfspinor(gauge_in,proj_res,mult_proj_res,gauge_idx,1);
	KokkosRecons23Dir1<T,isign>(mult_proj_res,res_sum);

}

  template<typename T, int isign>
KOKKOS_FORCEINLINE_FUNCTION
void DslashHVSiteDir2(const SpinorView<Kokkos::complex<T>>& spinor_in,   // Neighbor spinor
		      const GaugeView<Kokkos::complex<T>>& gauge_in,    // Gauge link
		      SpinorSiteView<Kokkos::complex<T>>& res_sum,        // result
		      int source_spinor_idx,
		      int gauge_idx)


{
	HalfSpinorSiteView<Kokkos::complex<T>> proj_res;
	HalfSpinorSiteView<Kokkos::complex<T>> mult_proj_res;

	KokkosProjectDir2<T,isign>(spinor_in, proj_res,source_spinor_idx);
	mult_adj_u_halfspinor(gauge_in,proj_res,mult_proj_res,gauge_idx,2);
	KokkosRecons23Dir2<T,isign>(mult_proj_res,res_sum);

}

  template<typename T, int isign>
KOKKOS_FORCEINLINE_FUNCTION
void DslashHVSiteDir3(const SpinorView<Kokkos::complex<T>>& spinor_in,   // Neighbor spinor
		      const GaugeView<Kokkos::complex<T>>& gauge_in,    // Gauge link
		      SpinorSiteView<Kokkos::complex<T>>& res_sum,        // result
		      int source_spinor_idx,
		      int gauge_idx)

{
	HalfSpinorSiteView<Kokkos::complex<T>> proj_res;
	HalfSpinorSiteView<Kokkos::complex<T>> mult_proj_res;

	KokkosProjectDir3<T,isign>(spinor_in, proj_res,source_spinor_idx);
	mult_adj_u_halfspinor(gauge_in,proj_res,mult_proj_res,gauge_idx,3);
	KokkosRecons23Dir3<T,isign>(mult_proj_res,res_sum);

}


template<typename T>
class KokkosDslash {
private:
	const LatticeInfo& _info;


public:
	KokkosDslash(const LatticeInfo& info) : _info(info) {}

	template<const int isign>
	void apply(const KokkosCBFineSpinor<Kokkos::complex<T>,4>& fine_in,
			const KokkosFineGaugeField<Kokkos::complex<T>>& gauge_in,
			KokkosCBFineSpinor<Kokkos::complex<T>,4>& fine_out) const
	{
		// Source and target checkerboards
		IndexType target_cb = fine_out.GetCB();
		IndexType source_cb = (target_cb == EVEN) ? ODD : EVEN;

		// Gather all views just outside Parallel Loop.
		// Can these be references? Will the world collapse?

		const SpinorView<Kokkos::complex<T>>& s_in = fine_in.GetData();
		const GaugeView<Kokkos::complex<T>>& g_in_src_cb = (gauge_in(source_cb)).GetData();
		const GaugeView<Kokkos::complex<T>>&  g_in_target_cb = (gauge_in(target_cb)).GetData();
//		SpinorView<Kokkos::complex<T>>& s_o = fine_out.GetData();

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

			SpinorView<Kokkos::complex<T>> s_o = fine_out.GetData();

			//Init result
			SpinorSiteView<Kokkos::complex<T>> res_sum;
			for(int color=0; color < 3; ++color) {
			  for(int spin=0; spin < 4; ++spin) {
			    ComplexZero(res_sum(color,spin));
			  }
			}

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


				DslashHVSiteDir3<T,isign>(s_in,
							  g_in_src_cb,
							  res_sum,
							  neigh_index,
							  neigh_index);

			}



			// Z - minus
			{
				if( z > 0 ) {
					neigh_index = xcb + _n_xh*(y + _n_y*((z-1) + _n_z*t));
				}
				else {
					neigh_index = xcb + _n_xh*(y + _n_y*((_n_z-1) + _n_z*t));
				}
				DslashHVSiteDir2<T,isign>(s_in,
							  g_in_src_cb,
							  res_sum,
							  neigh_index,
							  neigh_index);

			}

			// Y - minus
			{
				if( y > 0 ) {
					neigh_index = xcb + _n_xh*((y-1) + _n_y*(z + _n_z*t));
				}
				else {
					neigh_index = xcb + _n_xh*((_n_y-1) + _n_y*(z + _n_z*t));
				}

				DslashHVSiteDir1<T,isign>(s_in,
							  g_in_src_cb,
							  res_sum,
							  neigh_index,
							  neigh_index);


			}

			// X - minus
			{
				if ( x > 0 ) {
					neigh_index= ((x-1)/2) + _n_xh*(y + _n_y*(z + _n_z*t));
				}
				else {
					neigh_index= ((_n_x-1)/2) + _n_xh*(y + _n_y*(z + _n_z*t));
				}

				DslashHVSiteDir0<T,isign>(s_in,
							  g_in_src_cb,
							  res_sum,
							  neigh_index,
							  neigh_index);


			}


			// X - plus
			{
				if ( x < _n_x - 1 ) {
					neigh_index= ((x+1)/2)  + _n_xh*(y + _n_y*(z + _n_z*t));
				}
				else {
					neigh_index = 0 + _n_xh*(y + _n_y*(z + _n_z*t));
				}


				DslashMVSiteDir0<T,-isign>(s_in,
							   g_in_target_cb,
							   res_sum,
							   neigh_index,
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

				DslashMVSiteDir1<T,-isign>(s_in,
							   g_in_target_cb,
							   res_sum,
							   neigh_index,
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




				DslashMVSiteDir2<T,-isign>(s_in,
							   g_in_target_cb,
							   res_sum,
							   neigh_index,
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


				DslashMVSiteDir3<T,-isign>(s_in,
							   g_in_target_cb,
							   res_sum,
							   neigh_index,
							   site);

			}
			for(int color=0; color < 3; ++color) {
			  for(int spin=0; spin < 4; ++spin) {
			    ComplexCopy(s_o(site,color,spin),res_sum(color,spin));
			  }
			}

		});

	}

	void operator()(const KokkosCBFineSpinor<Kokkos::complex<T>,4>& fine_in,
		      const KokkosFineGaugeField<Kokkos::complex<T>>& gauge_in,
		      KokkosCBFineSpinor<Kokkos::complex<T>,4>& fine_out,
		      int plus_minus) const
	{
	  if( plus_minus == 1 ) {
	    apply<1>(fine_in, gauge_in,fine_out);
	  }
	  else {
	    apply<-1>(fine_in, gauge_in, fine_out);
	  }
	  
	}

};

};




#endif /* TEST_KOKKOS_KOKKOS_DSLASH_H_ */
