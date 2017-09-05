/*
 * kokkos_dslash.h
 *
 *  Created on: May 30, 2017
 *      Author: bjoo
 */

#ifndef TEST_KOKKOS_KOKKOS_VDSLASH_H_
#define TEST_KOKKOS_KOKKOS_VDSLASH_H_
#include "Kokkos_Macros.hpp"
#include "Kokkos_Core.hpp"
#include "kokkos_defaults.h"
#include "kokkos_types.h"
#include "kokkos_vtypes.h"
#include "kokkos_spinproj.h"
#include "kokkos_vspinproj.h"
#include "kokkos_vnode.h"
#include "kokkos_vmatvec.h"
#include "kokkos_traits.h"
#include "MG_config.h"

namespace MG {



enum DirIdx { T_MINUS=0, Z_MINUS=1, Y_MINUS=2, X_MINUS=3, X_PLUS=4, Y_PLUS=5, Z_PLUS=6, T_PLUS=7 };


#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
   template<typename VN>
   void ComputeSiteTable(int _n_xh, int _n_x, int _n_y, int _n_z, int _n_t,  Kokkos::View<Kokkos::pair<int,typename VN::MaskType>*[2][8],NeighLayout, MemorySpace> _table) {
		int num_sites =  _n_xh*_n_y*_n_z*_n_t;
			Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,num_sites), KOKKOS_LAMBDA(int site) {
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
			       _table(site,target_cb,T_MINUS) = Kokkos::make_pair(xcb + _n_xh*(y + _n_y*(z + _n_z*(t-1))),VN::NoPermuteMask);
			     }
			     else {
			       _table(site,target_cb,T_MINUS) = Kokkos::make_pair(xcb + _n_xh*(y + _n_y*(z + _n_z*(_n_t-1))),VN::TPermuteMask);
			     }

			     if( z > 0 ) {
			       _table(site,target_cb,Z_MINUS) = Kokkos::make_pair( xcb + _n_xh*(y + _n_y*((z-1) + _n_z*t)), VN::NoPermuteMask);
			     }
			     else {
			       _table(site,target_cb,Z_MINUS) = Kokkos::make_pair( xcb + _n_xh*(y + _n_y*((_n_z-1) + _n_z*t)),VN::ZPermuteMask);
			     }

			     if( y > 0 ) {
			       _table(site,target_cb,Y_MINUS) = Kokkos::make_pair( xcb + _n_xh*((y-1) + _n_y*(z + _n_z*t)), VN::NoPermuteMask);
			     }
			     else {
			       _table(site,target_cb,Y_MINUS) = Kokkos::make_pair( xcb + _n_xh*((_n_y-1) + _n_y*(z + _n_z*t)),VN::YPermuteMask);
			     }

			     if ( x > 0 ) {
			       _table(site,target_cb,X_MINUS)= Kokkos::make_pair(((x-1)/2) + _n_xh*(y + _n_y*(z + _n_z*t)), VN::NoPermuteMask);
			     }
			     else {
			       _table(site,target_cb,X_MINUS)= Kokkos::make_pair(((_n_x-1)/2) + _n_xh*(y + _n_y*(z + _n_z*t)),VN::XPermuteMask);
			     }

			     if ( x < _n_x - 1 ) {
			       _table(site,target_cb,X_PLUS) = Kokkos::make_pair(((x+1)/2)  + _n_xh*(y + _n_y*(z + _n_z*t)),VN::NoPermuteMask);
			     }
			     else {
			       _table(site,target_cb,X_PLUS) = Kokkos::make_pair(0 + _n_xh*(y + _n_y*(z + _n_z*t)),VN::XPermuteMask);
			     }

			     if( y < _n_y-1 ) {
			       _table(site,target_cb,Y_PLUS) = Kokkos::make_pair(xcb + _n_xh*((y+1) + _n_y*(z + _n_z*t)),VN::NoPermuteMask);
			     }
			     else {
			       _table(site,target_cb,Y_PLUS) = Kokkos::make_pair(xcb + _n_xh*(0 + _n_y*(z + _n_z*t)),VN::YPermuteMask);
			     }

			     if( z < _n_z-1 ) {
			       _table(site,target_cb,Z_PLUS) = Kokkos::make_pair(xcb + _n_xh*(y + _n_y*((z+1) + _n_z*t)),VN::NoPermuteMask);
			     }
			     else {
			       _table(site,target_cb,Z_PLUS) = Kokkos::make_pair(xcb + _n_xh*(y + _n_y*(0 + _n_z*t)), VN::ZPermuteMask);
			     }

			     if( t < _n_t-1 ) {
			       _table(site,target_cb,T_PLUS) = Kokkos::make_pair(xcb + _n_xh*(y + _n_y*(z + _n_z*(t+1))),VN::NoPermuteMask);
			     }
			     else {
			       _table(site,target_cb,T_PLUS) = Kokkos::make_pair(xcb + _n_xh*(y + _n_y*(z + _n_z*(0))),VN::TPermuteMask);
			     }
			    } // target CB
		        });

	}
#endif


template<typename VN>
class SiteTable {
public:


	  SiteTable(int n_xh,
		    int n_y,
		    int n_z,
		    int n_t) : 
	 _n_x(2*n_xh),
	 _n_xh(n_xh),
	 _n_y(n_y),
	 _n_z(n_z),
	 _n_t(n_t) {

#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
	   _table = Kokkos::View<Kokkos::pair<int,typename VN::MaskType>*[2][8],NeighLayout,MemorySpace>("table", n_xh*n_y*n_z*n_t);
	   ComputeSiteTable<VN>(n_xh, 2*n_xh, n_y, n_z, n_t, _table);
#endif
	}

	  KOKKOS_FORCEINLINE_FUNCTION
	  	int  coords_to_idx(const int& xcb, const int& y, const int& z, const int& t) const
	  	{
	  	  return xcb+_n_xh*(y + _n_y*(z + _n_z*t));
	  	}


#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborTMinus(int site, int target_cb, int& n_idx, typename VN::MaskType& mask) const {
		const Kokkos::pair<int,typename VN::MaskType>& lookup = _table(site,target_cb,T_MINUS);
		n_idx = lookup.first;
		mask = lookup.second;
	}

	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborTPlus(int site, int target_cb, int& n_idx, typename VN::MaskType& mask) const {
		const Kokkos::pair<int,typename VN::MaskType>& lookup = _table(site,target_cb,T_PLUS);
		n_idx = lookup.first;
		mask = lookup.second;
	}

	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborZMinus(int site, int target_cb, int& n_idx, typename VN::MaskType& mask) const {
		const Kokkos::pair<int,typename VN::MaskType>& lookup = _table(site,target_cb,Z_MINUS);
		n_idx = lookup.first;
		mask = lookup.second;
	}

	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborZPlus(int site, int target_cb, int& n_idx, typename VN::MaskType& mask) const {
		const Kokkos::pair<int,typename VN::MaskType>& lookup = _table(site,target_cb,Z_PLUS);
		n_idx = lookup.first;
		mask = lookup.second;
	}

	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborYMinus(int site, int target_cb, int& n_idx, typename VN::MaskType& mask) const {
		const Kokkos::pair<int,typename VN::MaskType>& lookup = _table(site,target_cb,Y_MINUS);
		n_idx = lookup.first;
		mask = lookup.second;
	}

	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborYPlus(int site, int target_cb, int& n_idx, typename VN::MaskType& mask) const {
		const Kokkos::pair<int,typename VN::MaskType>& lookup = _table(site,target_cb,Y_PLUS);
		n_idx = lookup.first;
		mask = lookup.second;
	}

	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborXMinus(int site, int target_cb, int& n_idx, typename VN::MaskType& mask) const {
		const Kokkos::pair<int,typename VN::MaskType>& lookup = _table(site,target_cb,X_MINUS);
		n_idx = lookup.first;
		mask = lookup.second;
	}

	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborXPlus(int site, int target_cb, int& n_idx, typename VN::MaskType& mask) const {
		const Kokkos::pair<int,typename VN::MaskType>& lookup = _table(site,target_cb,X_PLUS);
		n_idx = lookup.first;
		mask = lookup.second;
	}



#else


	KOKKOS_FORCEINLINE_FUNCTION
	void idx_to_coords(int site, int& xcb, int& y, int& z, int& t) const
	{
		IndexType tmp_yzt = site / _n_xh;
		xcb = site - _n_xh * tmp_yzt;
		IndexType tmp_zt = tmp_yzt / _n_y;
		y = tmp_yzt - _n_y * tmp_zt;
		t = tmp_zt / _n_z;
		z = tmp_zt - _n_z * t;
	}

	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborTMinus(int xcb, int y, int z, int t, int& n_idx, typename VN::MaskType& mask) const {
		if( t >  0) {
			n_idx=xcb + _n_xh*(y + _n_y*(z + _n_z*(t-1)));
			mask=VN::NoPermuteMask;
		}
		else {
			n_idx = xcb + _n_xh*(y + _n_y*(z + _n_z*(_n_t-1)));
			mask=VN::TPermuteMask;
		}
	}

	KOKKOS_FORCEINLINE_FUNCTION
	 void NeighborZMinus(int xcb, int y, int z, int t, int& n_idx, typename VN::MaskType& mask) const {
		if( z >  0 ) {
			n_idx=xcb + _n_xh*(y + _n_y*((z-1) + _n_z*t));
		    mask=VN::NoPermuteMask;
		}
		else {
			n_idx=xcb + _n_xh*(y + _n_y*((_n_z-1) + _n_z*t));
			mask=VN::ZPermuteMask;
		}
	}


	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborYMinus(int xcb, int y, int z, int t, int& n_idx, typename VN::MaskType& mask) const {
		if ( y > 0 ) {
			n_idx = xcb + _n_xh*((y-1) + _n_y*(z + _n_z*t));
			mask  = VN::NoPermuteMask;
		}
		else {
			n_idx = xcb + _n_xh*((_n_y-1) + _n_y*(z + _n_z*t));
			mask  = VN::YPermuteMask;
		}
	}


	KOKKOS_FORCEINLINE_FUNCTION
	void NeighborXMinus(int xcb, int y, int z, int t, int target_cb, int& n_idx, typename VN::MaskType& mask) const {
		int x = 2*xcb + ((target_cb+y+z+t)&0x1);
		if ( x > 0 ) {
				n_idx = ((x-1)/2) + _n_xh*(y + _n_y*(z + _n_z*t));
				mask  = VN::NoPermuteMask;
		}
		else {
				n_idx = ((_n_x-1)/2) + _n_xh*(y + _n_y*(z + _n_z*t));
				mask = VN::XPermuteMask;
		}
	}


	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborXPlus(int xcb, int y, int z, int t, int target_cb, int& n_idx, typename VN::MaskType& mask) const {
		int x = 2*xcb + ((target_cb+y+z+t)&0x1);
		if ( x < _n_x - 1) {
			n_idx = ((x+1)/2)  + _n_xh*(y + _n_y*(z + _n_z*t));
			mask = VN::NoPermuteMask;
		}
		else {
			n_idx = _n_xh*(y + _n_y*(z + _n_z*t));
			mask = VN::XPermuteMask;

		}
	}



	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborYPlus(int xcb, int y, int z, int t, int& n_idx, typename VN::MaskType& mask) const {
		if (y < _n_y - 1) {
			n_idx = xcb + _n_xh*((y+1) + _n_y*(z + _n_z*t));
			mask = VN::NoPermuteMask;
		}
		else {
			n_idx = xcb + _n_xh*(0 + _n_y*(z + _n_z*t));
			mask = VN::YPermuteMask;
		}
	}


	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborZPlus(int xcb, int y, int z, int t, int& n_idx, typename VN::MaskType& mask) const {


		if(z < _n_z - 1) {
			n_idx = xcb + _n_xh*(y + _n_y*((z+1) + _n_z*t));
			mask = VN::NoPermuteMask;
		}
		else {
		    n_idx = xcb + _n_xh*(y + _n_y*(0 + _n_z*t));
		    mask = VN::ZPermuteMask;
		}
	}


	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborTPlus(int xcb, int y, int z, int t, int& n_idx, typename VN::MaskType& mask) const {

		if (t < _n_t - 1) {
			n_idx = xcb + _n_xh*(y + _n_y*(z + _n_z*(t+1)));
			mask = VN::NoPermuteMask;
		}
		else {
			n_idx = xcb + _n_xh*(y + _n_y*(z + _n_z*(0)));
			mask = VN::TPermuteMask;
		}
	}
#endif


	KOKKOS_INLINE_FUNCTION
	  SiteTable( const SiteTable& st):
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
	  _table(st._table),
#endif
	  _n_x(st._n_x),
	  _n_xh(st._n_xh),
	  _n_y(st._n_y),
	  _n_z(st._n_z),
	  _n_t(st._n_t) {}

	KOKKOS_INLINE_FUNCTION
	  SiteTable& operator=(const  SiteTable& st) {
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
	  _table = st._table;
#endif
	  _n_x = st._n_x;
	  _n_xh = st._n_xh;
	  _n_y = st._n_y;
	  _n_z = st._n_z;
	  _n_t = st._n_t;

	  return *this;
	}

private:
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
	Kokkos::View<Kokkos::pair<int,typename VN::MaskType>*[2][8], NeighLayout,MemorySpace > _table;
#endif
       int _n_x;
       int _n_xh;
       int _n_y;
       int _n_z;
       int _n_t;

};




 template<typename VN,
   typename GT, 
   typename ST, 
   typename TGT, 
   typename TST, 
   const int isign, const int target_cb>
   struct VDslashFunctor { 

     VSpinorView<ST,VN> s_in;
     VGaugeView<GT,VN> g_in_src_cb;
     VGaugeView<GT,VN> g_in_target_cb;
     VSpinorView<ST,VN> s_out;
     SiteTable<VN> neigh_table;

     KOKKOS_FORCEINLINE_FUNCTION
       void operator()(const int& xcb, const int& y, const int& z, const int& t) const
     {

       int site = neigh_table.coords_to_idx(xcb,y,z,t);

       int n_idx;
       typename VN::MaskType mask;
     
      // Warning: GCC Alignment Attribute!
    		 // Site Sum: Not a true Kokkos View
    		 SpinorSiteView<TST> res_sum __attribute__((aligned(64)));
    		 HalfSpinorSiteView<TST> proj_res  __attribute__((aligned(64)));
    		 HalfSpinorSiteView<TST> mult_proj_res  __attribute__((aligned(64)));


    		 // Zero Result
    		 for(int color=0; color < 3; ++color) {
    			 for(int spin=0; spin < 4; ++spin) {
    				 ComplexZero(res_sum(color,spin));
    			 }
    		 }

    		 // T - minus
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
    		 neigh_table.NeighborTMinus(site,target_cb,n_idx,mask);
#else
    		 neigh_table.NeighborTMinus(xcb,y,z,t,n_idx,mask);
#endif
    		 KokkosProjectDir3Perm<ST,VN,TST,isign>(s_in, proj_res,n_idx,mask);
    		 mult_adj_u_halfspinor_perm<GT,VN,TST,3>(g_in_src_cb, proj_res,mult_proj_res,n_idx,mask);
    		 KokkosRecons23Dir3<TST,isign>(mult_proj_res,res_sum);

    		 // Z - minus
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
    		 neigh_table.NeighborZMinus(site,target_cb,n_idx,mask);
#else
    		 neigh_table.NeighborZMinus(xcb,y,z,t,n_idx,mask);
#endif
    		 KokkosProjectDir2Perm<ST,VN,TST,isign>(s_in, proj_res, n_idx, mask);
    		 mult_adj_u_halfspinor_perm<GT,VN,TST,2>(g_in_src_cb, proj_res,mult_proj_res,n_idx,mask);
    		 KokkosRecons23Dir2<TST,isign>(mult_proj_res,res_sum);

    		 // Y - minus
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
    		 neigh_table.NeighborYMinus(site,target_cb,n_idx,mask);
#else
    		 neigh_table.NeighborYMinus(xcb,y,z,t,n_idx,mask);
#endif
    		 KokkosProjectDir1Perm<ST,VN,TST,isign>(s_in, proj_res, n_idx,mask);
    		 mult_adj_u_halfspinor_perm<GT,VN,TST,1>(g_in_src_cb, proj_res,mult_proj_res,n_idx,mask);
    		 KokkosRecons23Dir1<TST,isign>(mult_proj_res,res_sum);

    		 // X - minus
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
    		 neigh_table.NeighborXMinus(site,target_cb,n_idx,mask);
#else
    		 neigh_table.NeighborXMinus(xcb,y,z,t,target_cb,n_idx,mask);
#endif
    		 KokkosProjectDir0Perm<ST,VN,TST,isign>(s_in, proj_res, n_idx,mask);
    		 mult_adj_u_halfspinor_perm<GT,VN,TST,0>(g_in_src_cb, proj_res,mult_proj_res,n_idx,mask);
    		 KokkosRecons23Dir0<TST,isign>(mult_proj_res,res_sum);


    		 // X - plus
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
    		 neigh_table.NeighborXPlus(site, target_cb, n_idx, mask);
#else

    		 neigh_table.NeighborXPlus(xcb,y,z,t,target_cb,n_idx, mask);
#endif
    		 KokkosProjectDir0Perm<ST,VN, TST,-isign>(s_in,proj_res,n_idx,mask);
    		 mult_u_halfspinor<GT,VN,TST,0>(g_in_target_cb,proj_res,mult_proj_res,site);
    		 KokkosRecons23Dir0<TST,-isign>(mult_proj_res, res_sum);

    		 // Y - plus
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
    		 neigh_table.NeighborYPlus(site, target_cb, n_idx, mask);
#else
    		 neigh_table.NeighborYPlus(xcb,y,z,t, n_idx, mask);
#endif
    		 KokkosProjectDir1Perm<ST,VN, TST,-isign>(s_in,proj_res,n_idx,mask);
    		 mult_u_halfspinor<GT,VN,TST,1>(g_in_target_cb,proj_res,mult_proj_res,site);
    		 KokkosRecons23Dir1<TST,-isign>(mult_proj_res, res_sum);

    		 // Z - plus
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
			 neigh_table.NeighborZPlus(site, target_cb, n_idx, mask);
#else
			 neigh_table.NeighborZPlus(xcb,y,z,t, n_idx, mask);
#endif
			 KokkosProjectDir2Perm<ST,VN, TST,-isign>(s_in,proj_res,n_idx,mask);
			 mult_u_halfspinor<GT,VN,TST,2>(g_in_target_cb,proj_res,mult_proj_res,site);
			 KokkosRecons23Dir2<TST,-isign>(mult_proj_res, res_sum);

			 // T - plus
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
			 neigh_table.NeighborTPlus(site,target_cb, n_idx, mask);
#else
			 neigh_table.NeighborTPlus(xcb,y,z,t, n_idx, mask);
#endif
			 KokkosProjectDir3Perm<ST,VN, TST,-isign>(s_in,proj_res,n_idx,mask);
			 mult_u_halfspinor<GT,VN,TST,3>(g_in_target_cb,proj_res,mult_proj_res,site);
			 KokkosRecons23Dir3<TST,-isign>(mult_proj_res, res_sum);

			 // Stream out spinor
			 for(int color=0; color < 3; ++color) {
				 for(int spin=0; spin < 4; ++spin) {
					 Stream(s_out(site,color,spin),res_sum(color,spin));
				 }
			 }

     }

     
   };

 template<typename VN, typename GT, typename ST,  typename TGT, typename TST>
   class KokkosVDslash {
 public:
	const LatticeInfo& _info;
	SiteTable<VN> _neigh_table;
public:

 KokkosVDslash(const LatticeInfo& info, int sites_per_team=1) : _info(info),
	  _neigh_table(info.GetCBLatticeDimensions()[0],info.GetCBLatticeDimensions()[1],info.GetCBLatticeDimensions()[2],info.GetCBLatticeDimensions()[3])
	  {}
	
	void operator()(const KokkosCBFineVSpinor<ST,VN,4>& fine_in,
			const KokkosFineVGaugeField<GT,VN>& gauge_in,
			KokkosCBFineVSpinor<ST,VN,4>& fine_out,
		      int plus_minus) const
	{
	  int source_cb = fine_in.GetCB();
	  int target_cb = (source_cb == EVEN) ? ODD : EVEN;
	  const VSpinorView<ST,VN>& s_in = fine_in.GetData();
	  const VGaugeView<GT,VN>& g_in_src_cb = (gauge_in(source_cb)).GetData();
	  const VGaugeView<GT,VN>&  g_in_target_cb = (gauge_in(target_cb)).GetData();
	  VSpinorView<ST,VN>& s_out = fine_out.GetData();

	  IndexArray cb_latdims = _info.GetCBLatticeDimensions();
	  MDPolicy policy({0,0,0,0},
			  	  {cb_latdims[0],cb_latdims[1],cb_latdims[2],cb_latdims[3]},
	  	  	  	  {3,3,3,cb_latdims[3]/4});
	  if( plus_minus == 1 ) {
	    if (target_cb == 0 ) {
	      VDslashFunctor<VN,GT,ST,TGT,TST,1,0> f = {s_in, g_in_src_cb, g_in_target_cb, s_out,
	    		  _neigh_table};
	      Kokkos::parallel_for(policy, f); // Outer Lambda

	    }
	    else {
	      VDslashFunctor<VN,GT,ST,TGT,TST,1,1> f = {s_in, g_in_src_cb, g_in_target_cb, s_out,
	    		   _neigh_table};
	      Kokkos::parallel_for(policy, f); // Outer Lambda
	    }
	  }
	  else {
	    if( target_cb == 0 ) { 
	      VDslashFunctor<VN,GT,ST,TGT,TST,-1,0> f = {s_in, g_in_src_cb, g_in_target_cb, s_out,
	    		  _neigh_table};
	      Kokkos::parallel_for(policy, f); // Outer Lambda
	      // Kokkos::parallel_for(num_sites,f);
	    }
	    else {
	      VDslashFunctor<VN,GT,ST,TGT,TST,-1,1> f = {s_in, g_in_src_cb, g_in_target_cb, s_out,
	    		  _neigh_table };
	      Kokkos::parallel_for(policy, f); // Outer Lambda
	      // Kokkos::parallel_for(num_sites,f);
	    }
	  }
	  
	}

};




};




#endif /* TEST_KOKKOS_KOKKOS_DSLASH_H_ */
