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

   void ComputeSiteTable(int _n_xh, int _n_x, int _n_y, int _n_z, int _n_t,  Kokkos::View<Kokkos::pair<int,bool>*[2][8],NeighLayout, MemorySpace> _table) {
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
			       _table(site,target_cb,T_MINUS) = Kokkos::make_pair(xcb + _n_xh*(y + _n_y*(z + _n_z*(t-1))),false);
			     }
			     else {
			       _table(site,target_cb,T_MINUS) = Kokkos::make_pair(xcb + _n_xh*(y + _n_y*(z + _n_z*(_n_t-1))),true);
			     }

			     if( z > 0 ) {
			       _table(site,target_cb,Z_MINUS) = Kokkos::make_pair( xcb + _n_xh*(y + _n_y*((z-1) + _n_z*t)), false);
			     }
			     else {
			       _table(site,target_cb,Z_MINUS) = Kokkos::make_pair( xcb + _n_xh*(y + _n_y*((_n_z-1) + _n_z*t)),true);
			     }

			     if( y > 0 ) {
			       _table(site,target_cb,Y_MINUS) = Kokkos::make_pair( xcb + _n_xh*((y-1) + _n_y*(z + _n_z*t)), false);
			     }
			     else {
			       _table(site,target_cb,Y_MINUS) = Kokkos::make_pair( xcb + _n_xh*((_n_y-1) + _n_y*(z + _n_z*t)),true);
			     }

			     if ( x > 0 ) {
			       _table(site,target_cb,X_MINUS)= Kokkos::make_pair(((x-1)/2) + _n_xh*(y + _n_y*(z + _n_z*t)), false);
			     }
			     else {
			       _table(site,target_cb,X_MINUS)= Kokkos::make_pair(((_n_x-1)/2) + _n_xh*(y + _n_y*(z + _n_z*t)),true);
			     }

			     if ( x < _n_x - 1 ) {
			       _table(site,target_cb,X_PLUS) = Kokkos::make_pair(((x+1)/2)  + _n_xh*(y + _n_y*(z + _n_z*t)),false);
			     }
			     else {
			       _table(site,target_cb,X_PLUS) = Kokkos::make_pair(0 + _n_xh*(y + _n_y*(z + _n_z*t)),true);
			     }

			     if( y < _n_y-1 ) {
			       _table(site,target_cb,Y_PLUS) = Kokkos::make_pair(xcb + _n_xh*((y+1) + _n_y*(z + _n_z*t)),false);
			     }
			     else {
			       _table(site,target_cb,Y_PLUS) = Kokkos::make_pair(xcb + _n_xh*(0 + _n_y*(z + _n_z*t)),true);
			     }

			     if( z < _n_z-1 ) {
			       _table(site,target_cb,Z_PLUS) = Kokkos::make_pair(xcb + _n_xh*(y + _n_y*((z+1) + _n_z*t)),false);
			     }
			     else {
			       _table(site,target_cb,Z_PLUS) = Kokkos::make_pair(xcb + _n_xh*(y + _n_y*(0 + _n_z*t)), true);
			     }

			     if( t < _n_t-1 ) {
			       _table(site,target_cb,T_PLUS) = Kokkos::make_pair(xcb + _n_xh*(y + _n_y*(z + _n_z*(t+1))),false);
			     }
			     else {
			       _table(site,target_cb,T_PLUS) = Kokkos::make_pair(xcb + _n_xh*(y + _n_y*(z + _n_z*(0))),true);
			     }
			    } // target CB
		        });

	}
#endif



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
	   _table = Kokkos::View<Kokkos::pair<int,bool>*[2][8],NeighLayout,MemorySpace>("table", n_xh*n_y*n_z*n_t);
	   ComputeSiteTable(n_xh, 2*n_xh, n_y, n_z, n_t, _table);
#endif
	}


#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
	KOKKOS_INLINE_FUNCTION
	  const Kokkos::pair<int,bool>&  NeighborTMinus(int site, int target_cb) const {
		return _table(site,target_cb,T_MINUS);
	}

	KOKKOS_INLINE_FUNCTION
	 const Kokkos::pair<int,bool>& NeighborTPlus(int site, int target_cb) const {
		return _table(site,target_cb,T_PLUS);
	}
	KOKKOS_INLINE_FUNCTION
	const Kokkos::pair<int,bool>& NeighborZMinus(int site, int target_cb) const {
		return _table(site,target_cb,Z_MINUS);
	}
	KOKKOS_INLINE_FUNCTION
	 const  Kokkos::pair<int,bool>& NeighborZPlus(int site, int target_cb) const {
		return _table(site,target_cb,Z_PLUS);
	}
	KOKKOS_INLINE_FUNCTION
	 const  Kokkos::pair<int,bool>& NeighborYMinus(int site, int target_cb) const {
		return _table(site,target_cb,Y_MINUS);
	}
	KOKKOS_INLINE_FUNCTION
	const  Kokkos::pair<int,bool>& NeighborYPlus(int site, int target_cb) const {
		return _table(site,target_cb,Y_PLUS);
	}
	KOKKOS_INLINE_FUNCTION
	 const Kokkos::pair<int,bool>& NeighborXMinus(int site, int target_cb) const {
		return _table(site,target_cb,X_MINUS);
	}
	KOKKOS_INLINE_FUNCTION
	  const Kokkos::pair<int,bool>& NeighborXPlus(int site, int target_cb) const {
		return _table(site,target_cb,X_PLUS);
	}
#else



	KOKKOS_INLINE_FUNCTION
	  Kokkos::pair<int,bool> NeighborTMinus(int site, int target_cb) const {
		// Break down site index into xcb, y,z and t
		IndexType tmp_yzt = site / _n_xh;
		IndexType xcb = site - _n_xh * tmp_yzt;
		IndexType tmp_zt = tmp_yzt / _n_y;
		IndexType y = tmp_yzt - _n_y * tmp_zt;
		IndexType t = tmp_zt / _n_z;
		IndexType z = tmp_zt - _n_z * t;


		return  ( t > 0 ) ? Kokkos::make_pair(xcb + _n_xh*(y + _n_y*(z + _n_z*(t-1))),false) 
		  : Kokkos::make_pair( xcb + _n_xh*(y + _n_y*(z + _n_z*(_n_t-1))),true) ;
	}



	KOKKOS_INLINE_FUNCTION
	  Kokkos::pair<int,bool> NeighborZMinus(int site, int target_cb) const {
		// Break down site index into xcb, y,z and t
		IndexType tmp_yzt = site / _n_xh;
		IndexType xcb = site - _n_xh * tmp_yzt;
		IndexType tmp_zt = tmp_yzt / _n_y;
		IndexType y = tmp_yzt - _n_y * tmp_zt;
		IndexType t = tmp_zt / _n_z;
		IndexType z = tmp_zt - _n_z * t;


		return  ( z > 0 ) ? Kokkos::make_pair( xcb + _n_xh*(y + _n_y*((z-1) + _n_z*t)),false) :
		  Kokkos::make_pair(xcb + _n_xh*(y + _n_y*((_n_z-1) + _n_z*t)),true);
	}


	KOKKOS_INLINE_FUNCTION
	  Kokkos::pair<int,bool> NeighborYMinus(int site, int target_cb) const {
		// Break down site index into xcb, y,z and t
		IndexType tmp_yzt = site / _n_xh;
		IndexType xcb = site - _n_xh * tmp_yzt;
		IndexType tmp_zt = tmp_yzt / _n_y;
		IndexType y = tmp_yzt - _n_y * tmp_zt;
		IndexType t = tmp_zt / _n_z;
		IndexType z = tmp_zt - _n_z * t;


		return  ( y > 0 ) ? Kokkos::make_pair(xcb + _n_xh*((y-1) + _n_y*(z + _n_z*t)),false) : 
		  Kokkos::make_pair(xcb + _n_xh*((_n_y-1) + _n_y*(z + _n_z*t)),true);

	}


	KOKKOS_INLINE_FUNCTION
	  Kokkos::pair<int,bool> NeighborXMinus(int site, int target_cb) const {
		// Break down site index into xcb, y,z and t
		IndexType tmp_yzt = site / _n_xh;
		IndexType xcb = site - _n_xh * tmp_yzt;
		IndexType tmp_zt = tmp_yzt / _n_y;
		IndexType y = tmp_yzt - _n_y * tmp_zt;
		IndexType t = tmp_zt / _n_z;
		IndexType z = tmp_zt - _n_z * t;

		// Global, uncheckerboarded x, assumes cb = (x + y + z + t ) & 1
		IndexType x = 2*xcb + ((target_cb+y+z+t)&0x1);

		return  (x > 0) ? Kokkos::make_pair( ((x-1)/2) + _n_xh*(y + _n_y*(z + _n_z*t)),false) :
		  Kokkos::make_pair(((_n_x-1)/2) + _n_xh*(y + _n_y*(z + _n_z*t)),true);
	}


	KOKKOS_INLINE_FUNCTION
	  Kokkos::pair<int,bool> NeighborXPlus(int site, int target_cb) const {
		// Break down site index into xcb, y,z and t
		IndexType tmp_yzt = site / _n_xh;
		IndexType xcb = site - _n_xh * tmp_yzt;
		IndexType tmp_zt = tmp_yzt / _n_y;
		IndexType y = tmp_yzt - _n_y * tmp_zt;
		IndexType t = tmp_zt / _n_z;
		IndexType z = tmp_zt - _n_z * t;

		// Global, uncheckerboarded x, assumes cb = (x + y + z + t ) & 1
		IndexType x = 2*xcb + ((target_cb+y+z+t)&0x1);

		return  (x < _n_x - 1) ? Kokkos::make_pair(((x+1)/2)  + _n_xh*(y + _n_y*(z + _n_z*t)),false) : 
		  Kokkos::make_pair(0 + _n_xh*(y + _n_y*(z + _n_z*t)),true);
	}



	KOKKOS_INLINE_FUNCTION
	  Kokkos::pair<int,bool> NeighborYPlus(int site, int target_cb) const {
		// Break down site index into xcb, y,z and t
		IndexType tmp_yzt = site / _n_xh;
		IndexType xcb = site - _n_xh * tmp_yzt;
		IndexType tmp_zt = tmp_yzt / _n_y;
		IndexType y = tmp_yzt - _n_y * tmp_zt;
		IndexType t = tmp_zt / _n_z;
		IndexType z = tmp_zt - _n_z * t;



		return  (y < _n_y - 1) ? Kokkos::make_pair(xcb + _n_xh*((y+1) + _n_y*(z + _n_z*t)),false) : 
		  Kokkos::make_pair(xcb + _n_xh*(0 + _n_y*(z + _n_z*t)),true);
	}


	KOKKOS_INLINE_FUNCTION
	  Kokkos::pair<int,bool> NeighborZPlus(int site, int target_cb) const {
		// Break down site index into xcb, y,z and t
		IndexType tmp_yzt = site / _n_xh;
		IndexType xcb = site - _n_xh * tmp_yzt;
		IndexType tmp_zt = tmp_yzt / _n_y;
		IndexType y = tmp_yzt - _n_y * tmp_zt;
		IndexType t = tmp_zt / _n_z;
		IndexType z = tmp_zt - _n_z * t;

		return  (z < _n_z - 1) ? Kokkos::make_pair( xcb + _n_xh*(y + _n_y*((z+1) + _n_z*t)),false) : 
		  Kokkos::make_pair(xcb + _n_xh*(y + _n_y*(0 + _n_z*t)),true);
	}


	KOKKOS_INLINE_FUNCTION
	  Kokkos::pair<int,bool> NeighborTPlus(int site, int target_cb) const {
		// Break down site index into xcb, y,z and t
		IndexType tmp_yzt = site / _n_xh;
		IndexType xcb = site - _n_xh * tmp_yzt;
		IndexType tmp_zt = tmp_yzt / _n_y;
		IndexType y = tmp_yzt - _n_y * tmp_zt;
		IndexType t = tmp_zt / _n_z;
		IndexType z = tmp_zt - _n_z * t;


		return  (t < _n_t - 1) ? Kokkos::make_pair(xcb + _n_xh*(y + _n_y*(z + _n_z*(t+1))),false) :
		  Kokkos::make_pair(xcb + _n_xh*(y + _n_y*(z + _n_z*(0))),true);
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
	Kokkos::View<Kokkos::pair<int,bool>*[2][8], NeighLayout,MemorySpace > _table;
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
     int num_sites;
     int sites_per_team;
     SiteTable neigh_table;

     KOKKOS_FORCEINLINE_FUNCTION
     void operator()(const TeamHandle& team) const {
		    const int start_idx = team.league_rank()*sites_per_team;
		    const int end_idx = start_idx + sites_per_team  < num_sites ? start_idx + sites_per_team : num_sites;

		    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,start_idx,end_idx),[=](const int site) {

			// Warning: GCC Alignment Attribute!
			// Site Sum: Not a true Kokkos View
			SpinorSiteView<TST> res_sum;// __attribute__((aligned(64)));
			
			// Temporaries: Not a true Kokkos View
			HalfSpinorSiteView<TST> proj_res; // __attribute__((aligned(64)));
			HalfSpinorSiteView<TST> mult_proj_res; // __attribute__((aligned(64)));
		    
			// A GaugeLink
			GaugeSiteView<TGT> gauge_in;


			// Zero Result
			for(int color=0; color < 3; ++color) {
			  for(int spin=0; spin < 4; ++spin) {
			    ComplexZero(res_sum(color,spin));
			  }
			}
			
			// T - minus
			// spinor
		       
			Kokkos::pair<int,bool> neighbor = neigh_table.NeighborTMinus(site,target_cb);
		        int n_idx = neighbor.first;
			bool  permuteP = neighbor.second;

			SpinorSiteView<TST> spinor_in;
			for(int color=0; color < 3; ++color) {
			  for(int spin=0; spin < 4; ++spin) {
			    Load( spinor_in(color,spin), s_in(n_idx, color,spin));
			    if( permuteP ) VN::permuteT(spinor_in(color,spin));
			  }
			}


			// gauge 
			for(int color=0; color < 3; ++color) {
			  for(int color2=0; color2 < 3; ++color2) {
			    Load( gauge_in(color,color2), g_in_src_cb( n_idx, 3, color,color2) );
			    if( permuteP ) VN::permuteT(gauge_in(color,color2));
			  }
			}
			

			KokkosProjectDir3<TST,isign>(spinor_in, proj_res);
			mult_adj_u_halfspinor<TGT,TST>(gauge_in, proj_res,mult_proj_res);
			KokkosRecons23Dir3<TST,isign>(mult_proj_res,res_sum);


			// Z - minus
			neighbor = neigh_table.NeighborZMinus(site,target_cb);
			n_idx = neighbor.first;
			permuteP = neighbor.second;

			for(int color=0; color < 3; ++color) {
			  for(int spin=0; spin < 4; ++spin) {
			    Load( spinor_in(color,spin), s_in( n_idx,color,spin));
			    if( permuteP ) VN::permuteZ(spinor_in(color,spin));
			  }
			}
			
			for(int color=0; color < 3; ++color) {
			  for(int color2=0; color2 < 3; ++color2) {
			    Load( gauge_in(color,color2), g_in_src_cb(n_idx, 2, color,color2) );
			    if( permuteP) VN::permuteZ(gauge_in(color,color2));
			  }
			}
			
			KokkosProjectDir2<TST,isign>(spinor_in, proj_res);
			mult_adj_u_halfspinor<TGT,TST>(gauge_in, proj_res,mult_proj_res);
			KokkosRecons23Dir2<TST,isign>(mult_proj_res,res_sum);

			// Y - minus
			neighbor = neigh_table.NeighborYMinus(site,target_cb);
		        n_idx = neighbor.first;
			permuteP = neighbor.second;

			for(int color=0; color < 3; ++color) {
			  for(int spin=0; spin < 4; ++spin) {
			    Load( spinor_in(color,spin), s_in( n_idx,color,spin));
			    if( permuteP ) VN::permuteY(spinor_in(color,spin));
			  }
			}

			for(int color=0; color < 3; ++color) {
			  for(int color2=0; color2 < 3; ++color2) {
			    Load( gauge_in(color,color2), g_in_src_cb( n_idx, 1, color,color2) );
			    if( permuteP) VN::permuteY(gauge_in(color,color2));
			  }
			}
			

			KokkosProjectDir1<TST,isign>(spinor_in, proj_res);
			mult_adj_u_halfspinor<TGT,TST>(gauge_in, proj_res, mult_proj_res);
			KokkosRecons23Dir1<TST,isign>(mult_proj_res,res_sum);
			

			// X - minus
			neighbor = neigh_table.NeighborXMinus(site,target_cb);
		        n_idx = neighbor.first;
			permuteP = neighbor.second;

			for(int color=0; color < 3; ++color) {
			  for(int spin=0; spin < 4; ++spin) {
			    Load( spinor_in(color,spin), s_in( n_idx,color,spin));
			    if( permuteP ) VN::permuteX( spinor_in(color,spin));
			  }
			}

			for(int color=0; color < 3; ++color) {
			  for(int color2=0; color2 < 3; ++color2) {
			    Load( gauge_in(color,color2), g_in_src_cb(n_idx, 0, color,color2) );
			    if( permuteP ) VN::permuteX( gauge_in(color,color2));
			  }
			}

			KokkosProjectDir0<TST,isign>(spinor_in, proj_res);
			mult_adj_u_halfspinor<TGT,TST>(gauge_in,proj_res,mult_proj_res);
			KokkosRecons23Dir0<TST,isign>(mult_proj_res,res_sum);

		    
			// X - plus
			neighbor = neigh_table.NeighborXPlus(site,target_cb);
		        n_idx = neighbor.first;
			permuteP = neighbor.second;

			for(int color=0; color < 3; ++color) {
			  for(int spin=0; spin < 4; ++spin) {
			    Load( spinor_in(color,spin), s_in( n_idx,color,spin));
			    if( permuteP ) VN::permuteX( spinor_in(color,spin));
			  }
			}

			for(int color=0; color < 3; ++color) {
			  for(int color2=0; color2 < 3; ++color2) {
			    Load( gauge_in(color,color2), g_in_target_cb(site, 0, color,color2));
			  }
			}

			KokkosProjectDir0<TST,-isign>(spinor_in,proj_res);
			mult_u_halfspinor<TGT,TST>(gauge_in,proj_res,mult_proj_res);
			KokkosRecons23Dir0<TST,-isign>(mult_proj_res, res_sum);

		    
			// Y - plus
			neighbor = neigh_table.NeighborYPlus(site,target_cb);
		        n_idx = neighbor.first;
			permuteP = neighbor.second;

			for(int color=0; color < 3; ++color) {
			  for(int spin=0; spin < 4; ++spin) {
			    Load( spinor_in(color,spin), s_in( n_idx,color,spin));
			    if( permuteP ) VN::permuteY( spinor_in(color,spin));
			  }
			}
			
			for(int color=0; color < 3; ++color) {
			  for(int color2=0; color2 < 3; ++color2) {
			    Load( gauge_in(color,color2), g_in_target_cb(site, 1, color,color2));
			  }
			}
			
			KokkosProjectDir1<TST,-isign>(spinor_in,proj_res);
			mult_u_halfspinor<TGT,TST>(gauge_in,proj_res,mult_proj_res);
			KokkosRecons23Dir1<TST,-isign>(mult_proj_res, res_sum);
			

		    // Z - plus
			neighbor = neigh_table.NeighborZPlus(site,target_cb);
		        n_idx = neighbor.first;
			permuteP = neighbor.second;

			for(int color=0; color < 3; ++color) {
			  for(int spin=0; spin < 4; ++spin) {
			    Load( spinor_in(color,spin), s_in(n_idx,color,spin));
			    if( permuteP ) VN::permuteZ( spinor_in(color,spin));
			  }
			}

			for(int color=0; color < 3; ++color) {
			  for(int color2=0; color2 < 3; ++color2) {
			    Load( gauge_in(color,color2), g_in_target_cb(site, 2, color,color2));
			  }
			}


			KokkosProjectDir2<TST,-isign>(spinor_in,proj_res);
			mult_u_halfspinor<TGT,TST>(gauge_in,proj_res,mult_proj_res);
			KokkosRecons23Dir2<TST,-isign>(mult_proj_res, res_sum);
		    
			// T - plus
			neighbor = neigh_table.NeighborTPlus(site,target_cb);
		        n_idx = neighbor.first;
			permuteP = neighbor.second;

			for(int color=0; color < 3; ++color) {
			  for(int spin=0; spin < 4; ++spin) {
			    Load( spinor_in(color,spin), s_in(n_idx,color,spin));
			    if( permuteP ) VN::permuteT( spinor_in(color,spin));
			  }
			}
			
			for(int color=0; color < 3; ++color) {
			  for(int color2=0; color2 < 3; ++color2) {
			    Load( gauge_in(color,color2), g_in_target_cb(site, 3, color,color2));
			  }
			}
			

			KokkosProjectDir3<TST,-isign>(spinor_in,proj_res);
			mult_u_halfspinor<TGT,TST>(gauge_in, proj_res,mult_proj_res);
			KokkosRecons23Dir3<TST,-isign>(mult_proj_res, res_sum);

			// Stream out spinor
			for(int color=0; color < 3; ++color) {
			  for(int spin=0; spin < 4; ++spin) {
			    Stream(s_out(site,color,spin),res_sum(color,spin));
			  }
			}
		      });
     }
     
     
   };

 template<typename VN, typename GT, typename ST,  typename TGT, typename TST>
   class KokkosVDslash {
 public:
	const LatticeInfo& _info;

	SiteTable _neigh_table;
	const int _sites_per_team;
public:

 KokkosVDslash(const LatticeInfo& info, int sites_per_team=1) : _info(info),
	  _neigh_table(info.GetCBLatticeDimensions()[0],info.GetCBLatticeDimensions()[1],info.GetCBLatticeDimensions()[2],info.GetCBLatticeDimensions()[3]),
	  _sites_per_team(sites_per_team)
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
	  const int num_sites = _info.GetNumCBSites();

	  ThreadExecPolicy policy(num_sites/_sites_per_team,Kokkos::AUTO(), VN::VecLen);
	  if( plus_minus == 1 ) {
	    if (target_cb == 0 ) {
	      VDslashFunctor<VN,GT,ST,TGT,TST,1,0> f = {s_in, g_in_src_cb, g_in_target_cb, s_out,
	    		  num_sites, _sites_per_team,_neigh_table};
	      Kokkos::parallel_for(policy, f); // Outer Lambda 
	    }
	    else {
	      VDslashFunctor<VN,GT,ST,TGT,TST,1,1> f = {s_in, g_in_src_cb, g_in_target_cb, s_out,
	    		  num_sites, _sites_per_team, _neigh_table};
	      Kokkos::parallel_for(policy, f); // Outer Lambda 
	    }
	  }
	  else {
	    if( target_cb == 0 ) { 
	      VDslashFunctor<VN,GT,ST,TGT,TST,-1,0> f = {s_in, g_in_src_cb, g_in_target_cb, s_out,
	    		  num_sites, _sites_per_team, _neigh_table};
	      Kokkos::parallel_for(policy, f); // Outer Lambda 
	    }
	    else {
	      VDslashFunctor<VN,GT,ST,TGT,TST,-1,1> f = {s_in, g_in_src_cb, g_in_target_cb, s_out,
	    		  num_sites, _sites_per_team, _neigh_table };
	      Kokkos::parallel_for(policy, f); // Outer Lambda 
	    }
	  }
	  
	}

};




};




#endif /* TEST_KOKKOS_KOKKOS_DSLASH_H_ */
