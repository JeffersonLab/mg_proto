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



#if 0
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
#endif


template<typename GT, typename ST>
class KokkosDslash {
private:
	const LatticeInfo& _info;

	enum DirIdx { T_MINUS=0, Z_MINUS=1, Y_MINUS=2, X_MINUS=3, X_PLUS=4, Y_PLUS=5, Z_PLUS=6, T_PLUS=7 };
	Kokkos::View<int*[2][8],Layout> _neigh_table;

public:
 KokkosDslash(const LatticeInfo& info) : _info(info), _neigh_table("neigh_table", info.GetNumCBSites()) 
	  {
	    // Now fill the neighbor table
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
				   for(int target_cb=0; target_cb < 2; ++target_cb) {
				     // Break down site index into xcb, y,z and t
				     IndexType tmp_yzt = site / _n_xh;
				     IndexType xcb = site - _n_xh * tmp_yzt;
				     IndexType tmp_zt = tmp_yzt / _n_y;
				     IndexType y = tmp_yzt - _n_y * tmp_zt;
				     IndexType t = tmp_zt / _n_z;
				     IndexType z = tmp_zt - _n_z * t;
				     
				     // Global, uncheckerboarded x, assumes cb = (x + y + z + t ) & 1
				     IndexType x = 2*xcb + ((target_cb+y+z+t)&0x1);
				     
				     if( t > 0 ) {
				       _neigh_table(site,target_cb,T_MINUS) = xcb + _n_xh*(y + _n_y*(z + _n_z*(t-1)));
				     }
				     else {
				       _neigh_table(site,target_cb,T_MINUS) = xcb + _n_xh*(y + _n_y*(z + _n_z*(_n_t-1)));
				     }
				     
				     if( z > 0 ) {
				       _neigh_table(site,target_cb,Z_MINUS) = xcb + _n_xh*(y + _n_y*((z-1) + _n_z*t));
				     }
				     else {
				       _neigh_table(site,target_cb,Z_MINUS) = xcb + _n_xh*(y + _n_y*((_n_z-1) + _n_z*t));
				     }
				     
				     if( y > 0 ) {
				       _neigh_table(site,target_cb,Y_MINUS) = xcb + _n_xh*((y-1) + _n_y*(z + _n_z*t));
				     }
				     else {
				       _neigh_table(site,target_cb,Y_MINUS) = xcb + _n_xh*((_n_y-1) + _n_y*(z + _n_z*t));
				     }
				     
				     if ( x > 0 ) {
				       _neigh_table(site,target_cb,X_MINUS)= ((x-1)/2) + _n_xh*(y + _n_y*(z + _n_z*t));
				     }
				     else {
				       _neigh_table(site,target_cb,X_MINUS)= ((_n_x-1)/2) + _n_xh*(y + _n_y*(z + _n_z*t));
				     }

				     if ( x < _n_x - 1 ) {
				       _neigh_table(site,target_cb,X_PLUS) = ((x+1)/2)  + _n_xh*(y + _n_y*(z + _n_z*t));
				     }
				     else {
				       _neigh_table(site,target_cb,X_PLUS) = 0 + _n_xh*(y + _n_y*(z + _n_z*t));
				     }
				     
				     if( y < _n_y-1 ) {
				       _neigh_table(site,target_cb,Y_PLUS) = xcb + _n_xh*((y+1) + _n_y*(z + _n_z*t));
				     }
				     else {
				       _neigh_table(site,target_cb,Y_PLUS) = xcb + _n_xh*(0 + _n_y*(z + _n_z*t));
				     }
				     
				     if( z < _n_z-1 ) {
				       _neigh_table(site,target_cb,Z_PLUS) = xcb + _n_xh*(y + _n_y*((z+1) + _n_z*t));
				     }
				     else {
				       _neigh_table(site,target_cb,Z_PLUS) = xcb + _n_xh*(y + _n_y*(0 + _n_z*t));
				     }
				     
				     if( t < _n_t-1 ) {
				       _neigh_table(site,target_cb,T_PLUS) = xcb + _n_xh*(y + _n_y*(z + _n_z*(t+1)));
				     }
				     else {
				       _neigh_table(site,target_cb,T_PLUS) = xcb + _n_xh*(y + _n_y*(z + _n_z*(0)));
				     }
			




				   }


				 });

	  }

	template<const int isign>
	void apply(const KokkosCBFineSpinor<ST,4>& fine_in,
			const KokkosFineGaugeField<GT>& gauge_in,
			KokkosCBFineSpinor<ST,4>& fine_out) const
	{
		// Source and target checkerboards
		IndexType target_cb = fine_out.GetCB();
		IndexType source_cb = (target_cb == EVEN) ? ODD : EVEN;

		// Gather all views just outside Parallel Loop.
		// Can these be references? Will the world collapse?

		const SpinorView<ST>& s_in = fine_in.GetData();
		const GaugeView<GT>& g_in_src_cb = (gauge_in(source_cb)).GetData();
		const GaugeView<GT>&  g_in_target_cb = (gauge_in(target_cb)).GetData();
		SpinorView<ST>& s_o = fine_out.GetData();
		const int num_sites = _info.GetNumCBSites();


		// Parallel site loop
		Kokkos::parallel_for(num_sites, KOKKOS_LAMBDA(int site) {

		    // Warning: GCC Alignment Attribute!
		    // Site Sum: Not a true Kokkos View
		    SpinorSiteView<ST> res_sum __attribute__((aligned(64)));

		    // Temporaries: Not a true Kokkos View
		    HalfSpinorSiteView<ST> proj_res __attribute__((aligned(64)));
		    HalfSpinorSiteView<ST> mult_proj_res __attribute__((aligned(64)));
		    

		    for(int color=0; color < 3; ++color) {
		      for(int spin=0; spin < 4; ++spin) {
			ComplexZero(res_sum(color,spin));
		      }
		    }
			
		    // T - minus
		    KokkosProjectDir3<ST,isign>(s_in, proj_res,_neigh_table(site,target_cb,T_MINUS));
		    mult_adj_u_halfspinor<GT,ST>(g_in_src_cb,proj_res,mult_proj_res,_neigh_table(site,target_cb,T_MINUS),3);
		    KokkosRecons23Dir3<ST,isign>(mult_proj_res,res_sum);
		    
		    // Z - minus
		    KokkosProjectDir2<ST,isign>(s_in, proj_res,_neigh_table(site,target_cb,Z_MINUS));
		    mult_adj_u_halfspinor<GT,ST>(g_in_src_cb,proj_res,mult_proj_res,_neigh_table(site,target_cb,Z_MINUS),2);
		    KokkosRecons23Dir2<ST,isign>(mult_proj_res,res_sum);
		    
		    // Y - minus
		    KokkosProjectDir1<ST,isign>(s_in, proj_res,_neigh_table(site,target_cb,Y_MINUS));
		    mult_adj_u_halfspinor<GT,ST>(g_in_src_cb,proj_res,mult_proj_res,_neigh_table(site,target_cb,Y_MINUS),1);
		    KokkosRecons23Dir1<ST,isign>(mult_proj_res,res_sum);
		    
		    // X - minus
		    KokkosProjectDir0<ST,isign>(s_in, proj_res,_neigh_table(site,target_cb,X_MINUS));
		    mult_adj_u_halfspinor<GT,ST>(g_in_src_cb,proj_res,mult_proj_res,_neigh_table(site,target_cb,X_MINUS),0);
		    KokkosRecons23Dir0<ST,isign>(mult_proj_res,res_sum);
		    
		    // X - plus
		    KokkosProjectDir0<ST,-isign>(s_in,proj_res,_neigh_table(site,target_cb,X_PLUS));
		    mult_u_halfspinor<GT,ST>(g_in_target_cb,proj_res,mult_proj_res,site,0);
		    KokkosRecons23Dir0<ST,-isign>(mult_proj_res, res_sum);
		    
		    // Y - plus
		    KokkosProjectDir1<ST,-isign>(s_in,proj_res,_neigh_table(site,target_cb,Y_PLUS));
		    mult_u_halfspinor<GT,ST>(g_in_target_cb,proj_res,mult_proj_res,site,1);
		    KokkosRecons23Dir1<ST,-isign>(mult_proj_res, res_sum);
		    
		    // Z - plus
		    KokkosProjectDir2<ST,-isign>(s_in,proj_res,_neigh_table(site,target_cb,Z_PLUS));
		    mult_u_halfspinor<GT,ST>(g_in_target_cb,proj_res,mult_proj_res,site,2);
		    KokkosRecons23Dir2<ST,-isign>(mult_proj_res, res_sum);
		    
		    // T - plus
		    KokkosProjectDir3<ST,-isign>(s_in,proj_res,_neigh_table(site,target_cb,T_PLUS));
		    mult_u_halfspinor<GT,ST>(g_in_target_cb,proj_res,mult_proj_res,site,3);
		    KokkosRecons23Dir3<ST,-isign>(mult_proj_res, res_sum);
		    
		    // Stream out spinor
		    for(int color=0; color < 3; ++color) {
		      for(int spin=0; spin < 4; ++spin) {
				Store(s_o(site,color,spin),res_sum(color,spin));
		      }
		    }
		    
		  });

	}

	void operator()(const KokkosCBFineSpinor<ST,4>& fine_in,
		      const KokkosFineGaugeField<GT>& gauge_in,
		      KokkosCBFineSpinor<ST,4>& fine_out,
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
