/*
 * kokkos_qdp_utils.h
 *
 *  Created on: May 23, 2017
 *      Author: bjoo
 */

#ifndef TEST_KOKKOS_KOKKOS_QDP_VUTILS_H_
#define TEST_KOKKOS_KOKKOS_QDP_VUTILS_H_

#include "qdp.h"
#include "kokkos_types.h"
#include <Kokkos_Core.hpp>

#include <utils/print_utils.h>
#include "kokkos_defaults.h"
#include "kokkos_vectype.h"
#include "lattice/geometry_utils.h"
namespace MG
{

	// Single QDP++ Vector
  template<typename T, typename VN, typename LF>
	void
	QDPLatticeFermionToKokkosCBVSpinor(const LF& qdp_in,
					  KokkosCBFineVSpinor<MGComplex<T>,VN,4>& kokkos_out)
	{
	  auto cb = kokkos_out.GetCB();
	  const QDP::Subset& sub = ( cb == EVEN ) ? QDP::rb[0] : QDP::rb[1];
	  
	  // Check conformance:
	  int num_gsites=static_cast<int>(kokkos_out.GetGlobalInfo().GetNumCBSites());
	  
	  if ( sub.numSiteTable() != num_gsites ) {
	    MasterLog(ERROR, "%s QDP++ Spinor has different number of sites per checkerboard than the KokkosCBFineSpinor",
		      __FUNCTION__);
	  }

	  int num_sites = static_cast<int>(kokkos_out.GetInfo().GetNumCBSites());
	  if ( num_sites * VN::VecLen != num_gsites ) { 
	    MasterLog(ERROR, "%s Veclen of Vector type x num_coarse_sites != num_fine_sites", 
		      __FUNCTION__);
	  }

	  auto h_out = Kokkos::create_mirror_view( kokkos_out.GetData() );

	  IndexArray coarse_dims = kokkos_out.GetInfo().GetCBLatticeDimensions();
	  IndexArray fine_dims = kokkos_out.GetGlobalInfo().GetCBLatticeDimensions();

	  Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_sites),
			       [=](int i) {
				 IndexArray c_coords;
				 IndexToCoords(i, coarse_dims,c_coords);
				 for(int color=0; color < 3; ++color) {
				   for(int spin=0; spin < 4; ++spin) {
				     
				     for(int lane =0; lane < VN::VecLen; ++lane) { 
				       IndexArray p_coords;
				       IndexArray vn_dims={VN::Dims[0], VN::Dims[1], VN::Dims[2], VN::Dims[3]};
				       IndexToCoords(lane, vn_dims,p_coords);

				       IndexArray g_coords;
				       for(int mu=0; mu < 4; ++mu) { 
					 g_coords[mu] = c_coords[mu] + p_coords[mu]*coarse_dims[mu];
				       }
				
				       int g_idx = CoordsToIndex(g_coords, fine_dims);
				       int qdp_index = sub.siteTable()[g_idx];
					     
				       h_out(i,color,spin)(lane) = MGComplex<T>(qdp_in.elem(qdp_index).elem(spin).elem(color).real(),
																	qdp_in.elem(qdp_index).elem(spin).elem(color).imag());
				     }//lane
				   } // spin
				 } // color
				 
			       }// kokkos lambda
			       );

		Kokkos::deep_copy(kokkos_out.GetData(), h_out);
	}

	// Single QDP++ vector
  template<typename T, typename VN, typename LF>
	void
    KokkosCBVSpinorToQDPLatticeFermion(const KokkosCBFineVSpinor<MGComplex<T>,VN, 4>& kokkos_in,
			LF& qdp_out) {

	  auto cb = kokkos_in.GetCB();
	  const QDP::Subset& sub = ( cb == EVEN ) ? QDP::rb[0] : QDP::rb[1];
	  
	  // Check conformance:
	  int num_csites=static_cast<int>(kokkos_in.GetInfo().GetNumCBSites());
	  int num_gsites=static_cast<int>(kokkos_in.GetGlobalInfo().GetNumCBSites());

	  if ( sub.numSiteTable() != num_gsites ) {
	    MasterLog(ERROR, "%s: QDP++ Spinor has different number of sites per checkerboard than the KokkosCBFineSpinor",
		      __FUNCTION__);
	  }

	  if( num_csites * VN::VecLen != num_gsites ) { 
	    MasterLog(ERROR, "%s: num_csites * veclen != num_gsites",
		      __FUNCTION__);
	  }

	  auto h_in = Kokkos::create_mirror_view( kokkos_in.GetData() );
	  Kokkos::deep_copy( h_in, kokkos_in.GetData() );

	  IndexArray c_dims = kokkos_in.GetInfo().GetCBLatticeDimensions();
	  IndexArray g_dims = kokkos_in.GetGlobalInfo().GetCBLatticeDimensions();

	  Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_csites),
			       [&](int i) {
				 IndexArray c_coords;
				     IndexToCoords(i,c_dims,c_coords);
				     
				     for(int color=0; color < 3; ++color) {
				       for(int spin=0; spin < 4; ++spin) {

					 for(int lane=0; lane < VN::VecLen;++lane) {
					   IndexArray p_coords;
					   IndexArray vn_dims={VN::Dims[0], VN::Dims[1], VN::Dims[2], VN::Dims[3]};
					   IndexToCoords(lane, vn_dims, p_coords);
					   IndexArray g_coords;
					   for(int mu=0; mu < 4; ++mu ) {
					     g_coords[mu] = c_coords[mu] + p_coords[mu]*c_dims[mu];
					   }
					   int g_index=CoordsToIndex(g_coords,g_dims);
					   int qdp_index = sub.siteTable()[g_index];
					   
					   qdp_out.elem(qdp_index).elem(spin).elem(color).real() = (h_in(i,color,spin))(lane).real();
					   qdp_out.elem(qdp_index).elem(spin).elem(color).imag() = (h_in(i,color,spin))(lane).imag();
				     } // lane 
				   } // spin
				 } // color
			       }// kokkos lambda
			       );
	}


  template<typename T, typename VN, typename GF>
	void
	QDPGaugeFieldToKokkosCBVGaugeField(const GF& qdp_in,
					   KokkosCBFineVGaugeField<MGComplex<T>,VN>& kokkos_out)
	{
	  auto cb = kokkos_out.GetCB();
	  const QDP::Subset& sub = ( cb == EVEN ) ? QDP::rb[0] : QDP::rb[1];

	  // Check conformance:
	  int num_gsites=static_cast<int>(kokkos_out.GetGlobalInfo().GetNumCBSites());
	  
	  if ( sub.numSiteTable() != num_gsites ) {
	    MasterLog(ERROR, "%s QDP++ Gauge has different number of sites per checkerboard than the KokkosCBFineVGaugeField",
		      __FUNCTION__);
	  }

	  int num_sites = static_cast<int>(kokkos_out.GetInfo().GetNumCBSites());
	  if ( num_sites * VN::VecLen != num_gsites ) { 
	    MasterLog(ERROR, "%s Veclen of Vector type x num_coarse_sites != num_fine_sites", 
		      __FUNCTION__);
	  }

	  auto h_out = Kokkos::create_mirror_view( kokkos_out.GetData() );

	  IndexArray coarse_dims = kokkos_out.GetInfo().GetCBLatticeDimensions();
	  IndexArray fine_dims = kokkos_out.GetGlobalInfo().GetCBLatticeDimensions();

	  Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_sites),
			       [=](int i) {
				 IndexArray c_coords;
				 IndexToCoords(i, coarse_dims,c_coords);
				 for(int dir=0; dir < 4; ++dir) { 
				   for(int color=0; color < 3; ++color) {
				     for(int color2=0; color2 < 3; ++color2) {
				     
				       for(int lane =0; lane < VN::VecLen; ++lane) { 
					 IndexArray p_coords;
				       IndexArray vn_dims={VN::Dims[0], VN::Dims[1], VN::Dims[2], VN::Dims[3]};
					 IndexToCoords(lane, vn_dims, p_coords);

					 IndexArray g_coords;
					 for(int mu=0; mu < 4; ++mu) { 
					   g_coords[mu] = c_coords[mu] + p_coords[mu]*coarse_dims[mu];
					 }
				
					 int g_idx = CoordsToIndex(g_coords, fine_dims);
					 int qdp_index = sub.siteTable()[g_idx];
					     
					 h_out(i,dir,color,color2)(lane) = MGComplex<T>(qdp_in[dir].elem(qdp_index).elem().elem(color,color2).real(),
										       qdp_in[dir].elem(qdp_index).elem().elem(color,color2).imag());
				     }//lane
				   } // color2
				 } // color
				 } // dir
			       }// kokkos lambda
			       );

		Kokkos::deep_copy(kokkos_out.GetData(), h_out);
	}


	    template<typename T, typename VN, typename GF>
	void
	    KokkosCBVGaugeFieldToQDPGaugeField(const KokkosCBFineVGaugeField<MGComplex<T>,VN>& kokkos_in,
					       GF& qdp_out)
	{
		auto cb = kokkos_in.GetCB();

		const QDP::Subset& sub = ( cb == EVEN ) ? QDP::rb[0] : QDP::rb[1];
	  // Check conformance:
	  int num_csites=static_cast<int>(kokkos_in.GetInfo().GetNumCBSites());
	  int num_gsites=static_cast<int>(kokkos_in.GetGlobalInfo().GetNumCBSites());

	  if ( sub.numSiteTable() != num_gsites ) {
	    MasterLog(ERROR, "%s: QDP++ Spinor has different number of sites per checkerboard than the KokkosCBFineSpinor",
		      __FUNCTION__);
	  }

	  if( num_csites * VN::VecLen != num_gsites ) { 
	    MasterLog(ERROR, "%s: num_csites * veclen != num_gsites",
		      __FUNCTION__);
	  }

	  auto h_in = Kokkos::create_mirror_view( kokkos_in.GetData() );
	  Kokkos::deep_copy( h_in, kokkos_in.GetData() );

	  IndexArray c_dims = kokkos_in.GetInfo().GetCBLatticeDimensions();
	  IndexArray g_dims = kokkos_in.GetGlobalInfo().GetCBLatticeDimensions();

	  Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_csites),
			       [&](int i) {
			
				 for(int dir=0; dir < 4; ++dir) {
				 for(int color=0; color < 3; ++color) {
				   for(int color2=0; color2 < 3; ++color2) {


				     // Convert i to an outer/lane index.
				     IndexArray c_coords;
				     IndexToCoords(i,c_dims,c_coords);
				     for(int lane=0; lane < VN::VecLen;++lane) {
				       IndexArray p_coords;
				       IndexArray vn_dims={VN::Dims[0], VN::Dims[1], VN::Dims[2], VN::Dims[3]};
				       IndexToCoords(lane, vn_dims, p_coords);
				       IndexArray g_coords;
				       for(int mu=0; mu < 4; ++mu ) {
					 g_coords[mu] = c_coords[mu] + p_coords[mu]*c_dims[mu];
				       }
				       int g_index=CoordsToIndex(g_coords,g_dims);
				       int qdp_index = sub.siteTable()[g_index];
				       qdp_out[dir].elem(qdp_index).elem().elem(color,color2).real() = (h_in(i,dir,color,color2))(lane).real();
				       qdp_out[dir].elem(qdp_index).elem().elem(color,color2).imag() = (h_in(i,dir,color,color2))(lane).imag();
				     } // lane 
				   } // color2
				 } // color
				 }// mu
			       }// kokkos lambda
			       );
	}

}




#endif /* TEST_KOKKOS_KOKKOS_QDP_VUTILS_H_ */
