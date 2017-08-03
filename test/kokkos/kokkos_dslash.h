/*
 * kokkos_dslash.h
 *
 *  Created on: May 30, 2017
 *      Author: bjoo
 */

#ifndef TEST_KOKKOS_KOKKOS_DSLASH_H_
#define TEST_KOKKOS_KOKKOS_DSLASH_H_
#include "Kokkos_Macros.hpp"
#include "Kokkos_Core.hpp"
#include "kokkos_defaults.h"
#include "kokkos_types.h"
#include "kokkos_spinproj.h"
#include "kokkos_matvec.h"
#include "kokkos_traits.h"

namespace MG {


;

  // Try an N-dimensional threading policy for cache blocking
typedef Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<4,Kokkos::Experimental::Iterate::Left,Kokkos::Experimental::Iterate::Left>> t_policy;

enum DirIdx { T_MINUS=0, Z_MINUS=1, Y_MINUS=2, X_MINUS=3, X_PLUS=4, Y_PLUS=5, Z_PLUS=6, T_PLUS=7 };


 void ComputeSiteTable(int _n_xh, int _n_y, int _n_z, int _n_t, Kokkos::View<int*[2][8],NeighLayout,MemorySpace>& _neigh_table)
{
		int _n_x = 2*_n_xh;
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
		    } // target CB
	        });

}



 template<typename GT, typename ST, typename TST, const int isign>
   struct DslashFunctor { 

#if 1
     SpinorView<ST> s_in;
     GaugeView<GT> g_in_src_cb;
     GaugeView<GT> g_in_target_cb;
     SpinorView<ST> s_out;
     Kokkos::View<int*[2][8],NeighLayout,MemorySpace> neigh_table;
     int source_cb; 
     int target_cb;
     int num_sites;
     int sites_per_team;
#else 
     const SpinorView<ST>& s_in;
     const GaugeView<GT>& g_in_src_cb;
     const GaugeView<GT>& g_in_target_cb;
     SpinorView<ST>& s_out;
     const Kokkos::View<int*[2][8],NeighLayout,MemorySpace>& neigh_table;
     int source_cb; 
     int target_cb;
     int num_sites;
     int sites_per_team;

     explicit DslashFunctor(     const SpinorView<ST>& _s_in,
				 const GaugeView<GT>& _g_in_src_cb,
				 const GaugeView<GT>& _g_in_target_cb,
				 SpinorView<ST>& _s_out,
				 const Kokkos::View<int*[2][8],NeighLayout,MemorySpace>&  _neigh_table,
				 int _source_cb,
				 int _target_cb,
				 int _num_sites,
				 int _sites_per_team) : s_in(_s_in),
       g_in_src_cb(_g_in_src_cb), g_in_target_cb(_g_in_target_cb),
       s_out(_s_out), neigh_table(_neigh_table), source_cb(_source_cb), 
       target_cb(_target_cb), num_sites(_num_sites), sites_per_team(_sites_per_team) {}
#endif


     KOKKOS_FORCEINLINE_FUNCTION
     void operator()(const TeamHandle& team) const {
		    const int start_idx = team.league_rank()*sites_per_team;
		    const int end_idx = start_idx + sites_per_team  < num_sites ? start_idx + sites_per_team : num_sites;

		    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,start_idx,end_idx),[=](const int site) {

#if 1
		    // Warning: GCC Alignment Attribute!
		    // Site Sum: Not a true Kokkos View
			SpinorSiteView<TST> res_sum;// __attribute__((aligned(64)));
			
		    // Temporaries: Not a true Kokkos View
			HalfSpinorSiteView<TST> proj_res; // __attribute__((aligned(64)));
			HalfSpinorSiteView<TST> mult_proj_res; // __attribute__((aligned(64)));
		    

		    for(int color=0; color < 3; ++color) {
		      for(int spin=0; spin < 4; ++spin) {
			ComplexZero(res_sum(color,spin));
		      }
		    }
			
		    // T - minus
		    KokkosProjectDir3<ST,TST,isign>(s_in, proj_res,neigh_table(site,target_cb,T_MINUS));
		    mult_adj_u_halfspinor<GT,TST>(g_in_src_cb,proj_res,mult_proj_res,neigh_table(site,target_cb,T_MINUS),3);
		    KokkosRecons23Dir3<TST,isign>(mult_proj_res,res_sum);
		    
		    // Z - minus
		    KokkosProjectDir2<ST,TST,isign>(s_in, proj_res,neigh_table(site,target_cb,Z_MINUS));
		    mult_adj_u_halfspinor<GT,TST>(g_in_src_cb,proj_res,mult_proj_res,neigh_table(site,target_cb,Z_MINUS),2);
		    KokkosRecons23Dir2<TST,isign>(mult_proj_res,res_sum);
		    
		    // Y - minus
		    KokkosProjectDir1<ST,TST,isign>(s_in, proj_res,neigh_table(site,target_cb,Y_MINUS));
		    mult_adj_u_halfspinor<GT,TST>(g_in_src_cb,proj_res,mult_proj_res,neigh_table(site,target_cb,Y_MINUS),1);
		    KokkosRecons23Dir1<TST,isign>(mult_proj_res,res_sum);
		    
		    // X - minus
		    KokkosProjectDir0<ST,TST,isign>(s_in, proj_res,neigh_table(site,target_cb,X_MINUS));
		    mult_adj_u_halfspinor<GT,TST>(g_in_src_cb,proj_res,mult_proj_res,neigh_table(site,target_cb,X_MINUS),0);
		    KokkosRecons23Dir0<TST,isign>(mult_proj_res,res_sum);
		    
		    // X - plus
		    KokkosProjectDir0<ST,TST,-isign>(s_in,proj_res,neigh_table(site,target_cb,X_PLUS));
		    mult_u_halfspinor<GT,TST>(g_in_target_cb,proj_res,mult_proj_res,site,0);
		    KokkosRecons23Dir0<TST,-isign>(mult_proj_res, res_sum);
		    
		    // Y - plus
		    KokkosProjectDir1<ST,TST,-isign>(s_in,proj_res,neigh_table(site,target_cb,Y_PLUS));
		    mult_u_halfspinor<GT,TST>(g_in_target_cb,proj_res,mult_proj_res,site,1);
		    KokkosRecons23Dir1<TST,-isign>(mult_proj_res, res_sum);
		    
		    // Z - plus
		    KokkosProjectDir2<ST,TST,-isign>(s_in,proj_res,neigh_table(site,target_cb,Z_PLUS));
		    mult_u_halfspinor<GT,TST>(g_in_target_cb,proj_res,mult_proj_res,site,2);
		    KokkosRecons23Dir2<TST,-isign>(mult_proj_res, res_sum);
		    
		    // T - plus
		    KokkosProjectDir3<ST,TST,-isign>(s_in,proj_res,neigh_table(site,target_cb,T_PLUS));
		    mult_u_halfspinor<GT,TST>(g_in_target_cb,proj_res,mult_proj_res,site,3);
		    KokkosRecons23Dir3<TST,-isign>(mult_proj_res, res_sum);
		    
		    // Stream out spinor
		    for(int color=0; color < 3; ++color) {
		      for(int spin=0; spin < 4; ++spin) {
		    	  Stream(s_out(site,color,spin),res_sum(color,spin));
		      }
		    }
#endif
		      });
     }


   };

template<typename GT, typename ST, typename TST>
class KokkosDslash {
 public:
	const LatticeInfo& _info;

	Kokkos::View<int*[2][8],NeighLayout,MemorySpace> _neigh_table;
	const int _sites_per_team;
public:

KokkosDslash(const LatticeInfo& info, int sites_per_team=1) : _info(info), _neigh_table("neigh_table", info.GetNumCBSites()), _sites_per_team(sites_per_team)
	  {
	    // Now fill the neighbor table
	    IndexArray latdims=_info.GetCBLatticeDimensions();
	    const int _n_xh = latdims[0];
	    const int _n_y = latdims[1];
	    const int _n_z = latdims[2];
	    const int _n_t = latdims[3];
	    // Parallel site loop
	    ComputeSiteTable( _n_xh, _n_y,  _n_z, _n_t,_neigh_table);
	  }
	
	void operator()(const KokkosCBFineSpinor<ST,4>& fine_in,
		      const KokkosFineGaugeField<GT>& gauge_in,
		      KokkosCBFineSpinor<ST,4>& fine_out,
		      int plus_minus) const
	{
	  int source_cb = fine_in.GetCB();
	  int target_cb = (source_cb == EVEN) ? ODD : EVEN;
	  const SpinorView<ST>& s_in = fine_in.GetData();
	  const GaugeView<GT>& g_in_src_cb = (gauge_in(source_cb)).GetData();
	  const GaugeView<GT>&  g_in_target_cb = (gauge_in(target_cb)).GetData();
	  SpinorView<ST>& s_out = fine_out.GetData();
	  const int num_sites = _info.GetNumCBSites();

	  if( plus_minus == 1 ) {
#if 1
		DslashFunctor<GT,ST,TST,1> f = {s_in, g_in_src_cb, g_in_target_cb, s_out, _neigh_table,source_cb, target_cb, num_sites, _sites_per_team};
#else
		DslashFunctor<GT,ST,TST,1> f ( s_in, g_in_src_cb, g_in_target_cb, s_out, _neigh_table,source_cb, target_cb, num_sites, _sites_per_team);
#endif
		ThreadExecPolicy policy(num_sites/_sites_per_team,Kokkos::AUTO(),Veclen<ST>::value);
		Kokkos::parallel_for(policy, f); // Outer Lambda 
	  }
	  else {

#if 1
		DslashFunctor<GT,ST,TST,-1> f = {s_in, g_in_src_cb, g_in_target_cb, s_out, _neigh_table,source_cb, target_cb, num_sites, _sites_per_team};
#else 
		DslashFunctor<GT,ST,TST,-1> f(s_in, g_in_src_cb, g_in_target_cb, s_out, _neigh_table,source_cb, target_cb, num_sites, _sites_per_team);
#endif
		ThreadExecPolicy policy(num_sites/_sites_per_team,Kokkos::AUTO(),Veclen<ST>::value);
		Kokkos::parallel_for(policy, f); // Outer Lambda 
	  }
	  
	}

};




};




#endif /* TEST_KOKKOS_KOKKOS_DSLASH_H_ */
