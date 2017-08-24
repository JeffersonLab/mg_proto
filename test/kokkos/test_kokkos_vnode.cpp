#include "gtest/gtest.h"
#include "../test_env.h"
#include "../mock_nodeinfo.h"
#include "../qdpxx/qdpxx_utils.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/fine_qdpxx/dslashm_w.h"

#include "./kokkos_types.h"
#include "./kokkos_defaults.h"
#include "./kokkos_qdp_utils.h"
#include "./kokkos_spinproj.h"
#include "./kokkos_matvec.h"
#include "./kokkos_dslash.h"
#include "MG_config.h"
#include "kokkos_vnode.h"
#include "kokkos_vtypes.h"
#include "kokkos_qdp_vutils.h"
#include "kokkos_traits.h"
#include <type_traits>

using namespace MG;
using namespace MGTesting;
using namespace QDP;

TEST(TestVNode,TestVSpinor)
{

  IndexArray latdims={{8,8,8,8}};
  initQDPXXLattice(latdims);
  QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
  LatticeInfo info(latdims,4,3,NodeInfo());
  
  using VN = VNode<MGComplex<float>,16>;
  using SpinorType = KokkosCBFineVSpinor<MGComplex<float>,VN,4>;
  // 4 spins
  SpinorType vnode_spinor(info, MG::EVEN);

  const LatticeInfo& c_info = vnode_spinor.GetInfo();
  for(int mu=0; mu < 4;++mu) {
    ASSERT_EQ( c_info.GetLatticeDimensions()[mu],
	       info.GetLatticeDimensions()[mu]/VN::Dims[mu] );
  }
  bool same_global_vectype = std::is_same< SpinorType::VecType, SIMDComplex<float,16> >::value;
  ASSERT_EQ( same_global_vectype, true); 

  bool same_thread_vectype = std::is_same< VN::VecType, ThreadSIMDComplex<float,16> >::value;
  ASSERT_EQ( same_thread_vectype, true);


}
TEST(TestVNode,TestVGauge)
{

  IndexArray latdims={{8,8,8,8}};
  initQDPXXLattice(latdims);
  QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
  LatticeInfo info(latdims,4,3,NodeInfo());
  
  using VN = VNode<MGComplex<float>,16>;
  using GaugeType = KokkosCBFineVGaugeField<MGComplex<float>,VN>;
  
  GaugeType vnode_spinor(info, MG::EVEN);

  const LatticeInfo& c_info = vnode_spinor.GetInfo();
  for(int mu=0; mu < 4;++mu) {
    ASSERT_EQ( c_info.GetLatticeDimensions()[mu],
	       info.GetLatticeDimensions()[mu]/VN::Dims[mu] );
  }
  bool same_global_vectype = std::is_same< GaugeType::VecType, SIMDComplex<float,16> >::value;
  ASSERT_EQ( same_global_vectype, true); 

  bool same_thread_vectype = std::is_same< VN::VecType, ThreadSIMDComplex<float,16> >::value;
  ASSERT_EQ( same_thread_vectype, true);


}
float computeLane(const IndexArray& coords, const IndexArray& cb_latdims)
{
  float value;
 			 if(    coords[0] < cb_latdims[0]/2 
			     && coords[1] < cb_latdims[1]/2  
			     && coords[2] < cb_latdims[2]/2 
			     && coords[3] < cb_latdims[3]/2 ) {
			   value = 0;
			 }

 			 if(    coords[0] >= cb_latdims[0]/2 
			     && coords[1] < cb_latdims[1]/2  
			     && coords[2] < cb_latdims[2]/2 
			     && coords[3] < cb_latdims[3]/2 ) {
			   value = 1;
			 }
			 
 			 if(    coords[0] <  cb_latdims[0]/2 
			     && coords[1] >= cb_latdims[1]/2  
			     && coords[2] <  cb_latdims[2]/2 
			     && coords[3] <  cb_latdims[3]/2 ) {
			   value = 2;
			 }

 			 if(    coords[0] >= cb_latdims[0]/2 
			     && coords[1] >= cb_latdims[1]/2  
			     && coords[2] < cb_latdims[2]/2 
			     && coords[3] < cb_latdims[3]/2 ) {
			   value = 3;
			 }

 			 if(    coords[0] < cb_latdims[0]/2 
			     && coords[1] < cb_latdims[1]/2  
			     && coords[2] >= cb_latdims[2]/2 
			     && coords[3] < cb_latdims[3]/2 ) {
			   value = 4;
			 }

 			 if(    coords[0] >= cb_latdims[0]/2 
			     && coords[1] < cb_latdims[1]/2  
			     && coords[2] >= cb_latdims[2]/2 
			     && coords[3] < cb_latdims[3]/2 ) {
			   value = 5;
			 }
			 
 			 if(    coords[0] <  cb_latdims[0]/2 
			     && coords[1] >= cb_latdims[1]/2  
			     && coords[2] >=  cb_latdims[2]/2 
			     && coords[3] <  cb_latdims[3]/2 ) {
			   value = 6;
			 }

 			 if(    coords[0] >= cb_latdims[0]/2 
			     && coords[1] >= cb_latdims[1]/2  
			     && coords[2] >= cb_latdims[2]/2 
			     && coords[3] < cb_latdims[3]/2 ) {
			   value = 7;
			 }

 			 if(    coords[0] < cb_latdims[0]/2 
			     && coords[1] < cb_latdims[1]/2  
			     && coords[2] < cb_latdims[2]/2 
			     && coords[3] >= cb_latdims[3]/2 ) {
			   value = 8;
			 }

 			 if(    coords[0] >= cb_latdims[0]/2 
			     && coords[1] < cb_latdims[1]/2  
			     && coords[2] < cb_latdims[2]/2 
			     && coords[3] >= cb_latdims[3]/2 ) {
			   value = 9;
			 }
			 
 			 if(    coords[0] <  cb_latdims[0]/2 
			     && coords[1] >= cb_latdims[1]/2  
			     && coords[2] <  cb_latdims[2]/2 
			     && coords[3] >=  cb_latdims[3]/2 ) {
			   value = 10;
			 }

 			 if(    coords[0] >= cb_latdims[0]/2 
			     && coords[1] >= cb_latdims[1]/2  
			     && coords[2] < cb_latdims[2]/2 
			     && coords[3] >= cb_latdims[3]/2 ) {
			   value = 11;
			 }

 			 if(    coords[0] < cb_latdims[0]/2 
			     && coords[1] < cb_latdims[1]/2  
			     && coords[2] >= cb_latdims[2]/2 
			     && coords[3] >= cb_latdims[3]/2 ) {
			   value = 12;
			}

 			 if(    coords[0] >= cb_latdims[0]/2 
			     && coords[1] < cb_latdims[1]/2  
			     && coords[2] >= cb_latdims[2]/2 
			     && coords[3] >= cb_latdims[3]/2 ) {
			   value = 13;
			 }
			 
 			 if(    coords[0] <  cb_latdims[0]/2 
			     && coords[1] >= cb_latdims[1]/2  
			     && coords[2] >=  cb_latdims[2]/2 
			     && coords[3] >=  cb_latdims[3]/2 ) {
			   value = 14;
			 }

 			 if(    coords[0] >= cb_latdims[0]/2 
			     && coords[1] >= cb_latdims[1]/2  
			     && coords[2] >= cb_latdims[2]/2 
			     && coords[3] >= cb_latdims[3]/2 ) {
			   value = 15;
			 }


			 return value;
}

TEST(TestVNode, TestPackSpinor)
{
  IndexArray latdims={{4,4,4,4}};
  initQDPXXLattice(latdims);
  QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
  LatticeInfo info(latdims,4,3,NodeInfo());
  
  using VN = VNode<MGComplex<float>,16>;
  using SpinorType = KokkosCBFineVSpinor<MGComplex<float>,VN,4>;

  LatticeFermion qdp_in = zero;

  IndexArray cb_latdims=info.GetCBLatticeDimensions();
  int num_cbsites= info.GetNumCBSites();
  MasterLog(INFO, "Num_cbsites=%d", num_cbsites);
  MasterLog(INFO, "cb_latdims=(%d,%d,%d,%d)", cb_latdims[0], cb_latdims[1],cb_latdims[2],cb_latdims[3]);
  Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_cbsites),[&](int i) { 
      IndexArray coords;
      IndexToCoords(i,cb_latdims,coords);
      float value=computeLane(coords,cb_latdims);
      int qdp_idx = rb[EVEN].siteTable()[i];
      
      for(int spin=0; spin < 4; ++spin) {
        for(int color=0; color < 3; ++color) {
          qdp_in.elem(qdp_idx).elem(spin).elem(color).real() = value;
          qdp_in.elem(qdp_idx).elem(spin).elem(color).imag() = 0;
        }
      }
    });


      SpinorType kokkos_spinor(info,EVEN);
      QDPLatticeFermionToKokkosCBVSpinor(qdp_in, kokkos_spinor);

      auto kokkos_h = Kokkos::create_mirror_view( kokkos_spinor.GetData() );
      Kokkos::deep_copy( kokkos_h, kokkos_spinor.GetData() );

      bool same_global_vectype = std::is_same< SpinorType::VecType, SIMDComplex<float,VN::VecLen> >::value;
      ASSERT_EQ( same_global_vectype, true);

      const LatticeInfo& vinfo = kokkos_spinor.GetInfo();
      int num_vcbsites = vinfo.GetNumCBSites();
      Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_vcbsites), [=](int i) {
      for(int color=0; color <3; ++color) { 
      for(int spin=0; spin < 4; ++spin) { 
      auto vec_data = kokkos_h(i,color,spin);
	  for(int lane=0; lane < VN::VecLen; ++lane) { 
            float ref = lane;
	    ASSERT_FLOAT_EQ( ref, vec_data(lane).real() );
	    ASSERT_FLOAT_EQ(  0 , vec_data(lane).imag() );
	    
    }//  lane
    } // spin
    } // color
    });
  LatticeFermion back_spinor=zero;
  KokkosCBVSpinorToQDPLatticeFermion(kokkos_spinor, back_spinor);
  Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_cbsites), [=](int i) {
      for(int color=0; color <3; ++color) {                                                                                             
      for(int spin=0; spin < 4; ++spin) {                                                                                               
      float ref_re = qdp_in.elem(rb[EVEN].siteTable()[i]).elem(spin).elem(color).real();
      float ref_im = qdp_in.elem(rb[EVEN].siteTable()[i]).elem(spin).elem(color).imag();

      float out_re = back_spinor.elem(rb[EVEN].siteTable()[i]).elem(spin).elem(color).real();
      float out_im = back_spinor.elem(rb[EVEN].siteTable()[i]).elem(spin).elem(color).imag();

      ASSERT_FLOAT_EQ( ref_re, out_re); 
      ASSERT_FLOAT_EQ( ref_im, out_im);
    } // spin                                                                                                                           
    } // color            
    });
}

TEST(TestVNode, TestPackSpinor2)
{
  IndexArray latdims={{4,4,4,4}};
  initQDPXXLattice(latdims);
  QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
  LatticeInfo info(latdims,4,3,NodeInfo());
  int num_cbsites = info.GetNumCBSites();

  using VN = VNode<MGComplex<float>,16>;
  using SpinorType = KokkosCBFineVSpinor<MGComplex<float>,VN,4>;

  LatticeFermion qdp_in;
  gaussian(qdp_in);

  SpinorType kokkos_spinor_e(info,EVEN);
  SpinorType kokkos_spinor_o(info,ODD);

  // Import 
  QDPLatticeFermionToKokkosCBVSpinor(qdp_in, kokkos_spinor_e);
  QDPLatticeFermionToKokkosCBVSpinor(qdp_in, kokkos_spinor_o);

  // Export 
  LatticeFermion qdp_out;
  KokkosCBVSpinorToQDPLatticeFermion(kokkos_spinor_e, qdp_out);
  KokkosCBVSpinorToQDPLatticeFermion(kokkos_spinor_o, qdp_out);

  for(int cb=EVEN; cb <= ODD; ++cb) { 
      Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_cbsites), [=](int i) {
      for(int color=0; color <3; ++color) {                                                                                             
      for(int spin=0; spin < 4; ++spin) {                                                                                               
      float ref_re = qdp_in.elem(rb[cb].siteTable()[i]).elem(spin).elem(color).real();
      float ref_im = qdp_in.elem(rb[cb].siteTable()[i]).elem(spin).elem(color).imag();

      float out_re = qdp_out.elem(rb[cb].siteTable()[i]).elem(spin).elem(color).real();
      float out_im = qdp_out.elem(rb[cb].siteTable()[i]).elem(spin).elem(color).imag();

      ASSERT_FLOAT_EQ( ref_re, out_re); 
      ASSERT_FLOAT_EQ( ref_im, out_im);
    } // spin                                                                                                                           
    } // color            
    });
    } // cb
    }


TEST(TestVNode, TestPackGauge)
{
  IndexArray latdims={{4,4,4,4}};
  initQDPXXLattice(latdims);
  QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
  LatticeInfo info(latdims,4,3,NodeInfo());
  
  using VN = VNode<MGComplex<float>,16>;
  using GaugeType = KokkosCBFineVGaugeField<MGComplex<float>,VN>;

  multi1d<LatticeColorMatrix> u(Nd);
  IndexArray cb_latdims=info.GetCBLatticeDimensions();
  int num_cbsites= info.GetNumCBSites();
  MasterLog(INFO, "Num_cbsites=%d", num_cbsites);
  MasterLog(INFO, "cb_latdims=(%d,%d,%d,%d)", cb_latdims[0], cb_latdims[1],cb_latdims[2],cb_latdims[3]);

  for(int mu=0; mu < 4; ++mu)  {
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_cbsites),[&](int i) { 
	IndexArray coords;
	IndexToCoords(i,cb_latdims,coords);
	float value=computeLane(coords,cb_latdims);
	int qdp_idx = rb[EVEN].siteTable()[i];
	
	for(int color=0; color < 3; ++color) {
	  for(int color2=0; color2 < 3; ++color2) {
	 
	    u[mu].elem(qdp_idx).elem().elem(color,color2).real() = value;
	    u[mu].elem(qdp_idx).elem().elem(color,color2).imag() = -value;
	  }
	}
      });
  }

      GaugeType kokkos_u(info,EVEN);
      QDPGaugeFieldToKokkosCBVGaugeField(u, kokkos_u);

      auto kokkos_h = Kokkos::create_mirror_view( kokkos_u.GetData() );
      Kokkos::deep_copy( kokkos_h, kokkos_u.GetData() );

      bool same_global_vectype = std::is_same< GaugeType::VecType, SIMDComplex<float,VN::VecLen> >::value;
      ASSERT_EQ( same_global_vectype, true);

      const LatticeInfo& vinfo = kokkos_u.GetInfo();
      int num_vcbsites = vinfo.GetNumCBSites();

      Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_vcbsites), [=](int i) {
	  for(int dir=0; dir < 4; ++dir) {
	    for(int color=0; color <3; ++color) { 
	      for(int color2=0; color2 < 3; ++color2) { 

		auto vec_data = kokkos_h(i,dir,color,color2);
		for(int lane=0; lane < VN::VecLen; ++lane) { 
		  float ref = lane;
		  ASSERT_FLOAT_EQ( ref, vec_data(lane).real() );
		  ASSERT_FLOAT_EQ( -ref , vec_data(lane).imag() );
		  
		}//  lane
	      } // color2
	    } // color
	  } // dir
	});

      multi1d<LatticeColorMatrix> u_back(Nd);

      KokkosCBVGaugeFieldToQDPGaugeField(kokkos_u, u_back);

      for(int mu=0; mu < Nd; ++mu) {
  Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_cbsites), [=](int i) {
   
  for(int color=0; color <3; ++color) {                                                                                             
  for(int color2=0; color2 < 3; ++color2) {                                                                                               
  float ref_re = u[mu].elem(rb[EVEN].siteTable()[i]).elem().elem(color,color2).real();
  float ref_im = u[mu].elem(rb[EVEN].siteTable()[i]).elem().elem(color,color2).imag();

  float out_re = u_back[mu].elem(rb[EVEN].siteTable()[i]).elem().elem(color,color2).real();
  float out_im = u_back[mu].elem(rb[EVEN].siteTable()[i]).elem().elem(color,color2).imag();

      ASSERT_FLOAT_EQ( ref_re, out_re); 
      ASSERT_FLOAT_EQ( ref_im, out_im);
    } // color2                                                                                                                    
    } // color            
    });
}// mu

}


TEST(TestVNode, TestPackGauge2)
{
  IndexArray latdims={{4,4,4,4}};
  initQDPXXLattice(latdims);
  QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
  LatticeInfo info(latdims,4,3,NodeInfo());
  int num_cbsites = info.GetNumCBSites();

  using VN = VNode<MGComplex<float>,16>;
  using GaugeType = KokkosCBFineVGaugeField<MGComplex<float>,VN>;

  multi1d<LatticeColorMatrix> qdp_in(Nd);
  for(int mu=0; mu < Nd; ++mu) { 
    gaussian(qdp_in[mu]);
    reunit(qdp_in[mu]);
  } 

  GaugeType kokkos_u_e(info,EVEN);
  GaugeType kokkos_u_o(info,ODD);

  // Import 
  QDPGaugeFieldToKokkosCBVGaugeField(qdp_in, kokkos_u_e);
  QDPGaugeFieldToKokkosCBVGaugeField(qdp_in, kokkos_u_o);

  // Export 
  multi1d<LatticeColorMatrix> qdp_out(Nd);
  KokkosCBVGaugeFieldToQDPGaugeField(kokkos_u_e, qdp_out);
  KokkosCBVGaugeFieldToQDPGaugeField(kokkos_u_o, qdp_out);
  
  for(int mu=0; mu < Nd; ++mu) { 
  for(int cb=EVEN; cb <= ODD; ++cb) { 
      Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_cbsites), [=](int i) {
      for(int color=0; color <3; ++color) {                                                                                             
      for(int color2=0; color2 < 3; ++color2) {                                                                                               
  float ref_re = qdp_in[mu].elem(rb[cb].siteTable()[i]).elem().elem(color,color2).real();
  float ref_im = qdp_in[mu].elem(rb[cb].siteTable()[i]).elem().elem(color,color2).imag();

  float out_re = qdp_out[mu].elem(rb[cb].siteTable()[i]).elem().elem(color,color2).real();
  float out_im = qdp_out[mu].elem(rb[cb].siteTable()[i]).elem().elem(color,color2).imag();

      ASSERT_FLOAT_EQ( ref_re, out_re); 
      ASSERT_FLOAT_EQ( ref_im, out_im);
    } // spin                                                                                                                           
    } // color            
    });
    } // cb
}//mu
    }



int main(int argc, char *argv[]) 
{
	return ::MGTesting::TestMain(&argc, argv);
}
