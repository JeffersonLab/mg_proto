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
#include <omp.h>
#include "MG_config.h"

using namespace MG;
using namespace MGTesting;
using namespace QDP;


#ifdef KOKKOS_HAVE_CUDA
        constexpr static int V = 16;
#else
        constexpr static int V = 8;
#endif
#if 0
TEST(TestKokkos, TestSpinProject)
{
	IndexArray latdims={{32,32,32,32}};
	int iters = 1000;
 
	initQDPXXLattice(latdims);

	LatticeInfo info(latdims,4,3,NodeInfo());
	LatticeInfo hinfo(latdims,2,3,NodeInfo());

	KokkosCBFineSpinor<MGComplex<REAL32>,4> kokkos_in(info,EVEN);
	KokkosCBFineSpinor<MGComplex<REAL32>,2> kokkos_hspinor_out(hinfo,EVEN);

	{
		LatticeFermionF qdp_in;
		gaussian(qdp_in);
		QDPLatticeFermionToKokkosCBSpinor(qdp_in, kokkos_in);
	}

	double rfo = 1.0;
	double bytes_in = (latdims[0]/2)*latdims[1]*latdims[2]*latdims[3]*4*3*2*sizeof(REAL32)*iters;
	double bytes_out =(1.0+rfo)*(latdims[0]/2)*latdims[1]*latdims[2]*latdims[3]*2*3*2*sizeof(REAL32)*iters;
	double mem_moved_in_GB = (bytes_in+bytes_out)/1.0e9;

	{
	  MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",0,-1);
	  
	  double start_time = omp_get_wtime();
	  for(int i=0; i < iters; ++i) {
	    KokkosProjectLattice<MGComplex<REAL32>,0,-1>(kokkos_in,kokkos_hspinor_out);
	  }
	  double end_time = omp_get_wtime();
 	  double effective_bw = mem_moved_in_GB / (end_time-start_time);
	  MasterLog(INFO, "Effective Bandwidth = %16.8e GB/sec\n", effective_bw);
	}
	{
	  MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",1,-1);
	  
	  double start_time = omp_get_wtime();
	  for(int i=0; i < iters; ++i) {
	    KokkosProjectLattice<MGComplex<REAL32>,1,-1>(kokkos_in,kokkos_hspinor_out);
	  }
	  double end_time = omp_get_wtime();
 	  double effective_bw = mem_moved_in_GB / (end_time-start_time);
	  MasterLog(INFO, "Effective Bandwidth = %16.8e GB/sec\n", effective_bw);
	}
	{
	  MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",2,-1);
	  
	  double start_time = omp_get_wtime();
	  for(int i=0; i < iters; ++i) {
	    KokkosProjectLattice<MGComplex<REAL32>,2,-1>(kokkos_in,kokkos_hspinor_out);
	  }
	  double end_time = omp_get_wtime();
 	  double effective_bw = mem_moved_in_GB / (end_time-start_time);
	  MasterLog(INFO, "Effective Bandwidth = %16.8e GB/sec\n", effective_bw);
	}
	{
	  MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",3,-1);
	  
	  double start_time = omp_get_wtime();
	  for(int i=0; i < iters; ++i) {
	    KokkosProjectLattice<MGComplex<REAL32>,3,-1>(kokkos_in,kokkos_hspinor_out);
	  }
	  double end_time = omp_get_wtime();
 	  double effective_bw = mem_moved_in_GB / (end_time-start_time);
	  MasterLog(INFO, "Effective Bandwidth = %16.8e GB/sec\n", effective_bw);
	}

	{
	  MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",0,1);
	  
	  double start_time = omp_get_wtime();
	  for(int i=0; i < iters; ++i) {
	    KokkosProjectLattice<MGComplex<REAL32>,0,1>(kokkos_in,kokkos_hspinor_out);
	  }
	  double end_time = omp_get_wtime();
 	  double effective_bw = mem_moved_in_GB / (end_time-start_time);
	  MasterLog(INFO, "Effective Bandwidth = %16.8e GB/sec\n", effective_bw);
	}
	{
	  MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",1,1);
	  
	  double start_time = omp_get_wtime();
	  for(int i=0; i < iters; ++i) {
	    KokkosProjectLattice<MGComplex<REAL32>,1,1>(kokkos_in,kokkos_hspinor_out);
	  }
	  double end_time = omp_get_wtime();
 	  double effective_bw = mem_moved_in_GB / (end_time-start_time);
	  MasterLog(INFO, "Effective Bandwidth = %16.8e GB/sec\n", effective_bw);
	}
	{
	  MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",2,1);
	  
	  double start_time = omp_get_wtime();
	  for(int i=0; i < iters; ++i) {
	    KokkosProjectLattice<MGComplex<REAL32>,2,1>(kokkos_in,kokkos_hspinor_out);
	  }
	  double end_time = omp_get_wtime();
 	  double effective_bw = mem_moved_in_GB / (end_time-start_time);
	  MasterLog(INFO, "Effective Bandwidth = %16.8e GB/sec\n", effective_bw);
	}
	{
	  MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",3,1);
	  
	  double start_time = omp_get_wtime();
	  for(int i=0; i < iters; ++i) {
	    KokkosProjectLattice<MGComplex<REAL32>,3,1>(kokkos_in,kokkos_hspinor_out);
	  }
	  double end_time = omp_get_wtime();
 	  double effective_bw = mem_moved_in_GB / (end_time-start_time);
	  MasterLog(INFO, "Effective Bandwidth = %16.8e GB/sec\n", effective_bw);
	}

}
#endif

#if 0
TEST(TestKokkos, TestSpinRecons)
{
	IndexArray latdims={{24,24,24,24}};
	initQDPXXLattice(latdims);

	LatticeInfo info(latdims,4,3,NodeInfo());
	LatticeInfo hinfo(latdims,2,3,NodeInfo());

	LatticeHalfFermion qdp_in;
	LatticeFermion     qdp_out;
	LatticeFermion     kokkos_out;


	gaussian(qdp_in);

	KokkosCBFineSpinor<REAL,2> kokkos_hspinor_in(hinfo,EVEN);
	KokkosCBFineSpinor<REAL,4> kokkos_spinor_out(info,EVEN);

	QDPLatticeHalfFermionToKokkosCBSpinor2(qdp_in, kokkos_hspinor_in);

	for(int s=-1; s <= +1; s+=2 ) {
		for(int dir=0; dir < 4; ++dir ) {
			MasterLog(INFO, "Spin Recons Test: dir = %d sign = %d", dir, s);

			if ( s == -1 ) {
				switch(dir) {
				case 0:
					qdp_out[rb[0]] = spinReconstructDir0Minus(qdp_in);
					break;
				case 1:
					qdp_out[rb[0]] = spinReconstructDir1Minus(qdp_in);
					break;
				case 2:
					qdp_out[rb[0]] = spinReconstructDir2Minus(qdp_in);
					break;
				case 3:
					qdp_out[rb[0]] = spinReconstructDir3Minus(qdp_in);
					break;
				default:
					MasterLog(ERROR, "Bad direction in SpinReconstruction Test");
					break;
				}
			}
			else {
				switch(dir) {
				case 0:
					qdp_out[rb[0]] = spinReconstructDir0Plus(qdp_in);
					break;
				case 1:
					qdp_out[rb[0]] = spinReconstructDir1Plus(qdp_in);
					break;
				case 2:
					qdp_out[rb[0]] = spinReconstructDir2Plus(qdp_in);
					break;
				case 3:
					qdp_out[rb[0]] = spinReconstructDir3Plus(qdp_in);
					break;
				default:
					MasterLog(ERROR, "Bad direction in SpinReconstruction Test");
					break;
				}

			}


			qdp_out[rb[1]] = zero;

			KokkosReconsLattice(kokkos_hspinor_in,dir,s,kokkos_spinor_out);
			KokkosCBSpinorToQDPLatticeFermion(kokkos_spinor_out,kokkos_out);

			qdp_out[rb[0]] -= kokkos_out;

			double norm_diff = toDouble(sqrt(norm2(qdp_out)));
			MasterLog(INFO, "norm_diff = %lf", norm_diff);
			ASSERT_LT( norm_diff, 1.0e-5);
		} // dir
 	} // s
}
#endif

#if 0
TEST(TestKokkos, TestMultHalfSpinor)
{
	IndexArray latdims={{16,16,16,16}};
	initQDPXXLattice(latdims);
	multi1d<LatticeColorMatrix> gauge_in(n_dim);
	for(int mu=0; mu < n_dim; ++mu) {
		gaussian(gauge_in[mu]);
		reunit(gauge_in[mu]);
	}

	LatticeHalfFermion psi_in;
	gaussian(psi_in);

	LatticeInfo info(latdims,4,3,NodeInfo());
	LatticeInfo hinfo(latdims,2,3,NodeInfo());

	KokkosCBFineSpinor<REAL,2> kokkos_hspinor_in(hinfo,EVEN);
	KokkosCBFineSpinor<REAL,2> kokkos_hspinor_out(hinfo,EVEN);
	KokkosCBFineGaugeField<REAL>  kokkos_gauge_e(info, EVEN);


	// Import Gauge Field
	QDPGaugeFieldToKokkosCBGaugeField(gauge_in, kokkos_gauge_e);

	LatticeHalfFermion psi_out, kokkos_out;
	MasterLog(INFO, "Testing MV");
	{
		psi_out[rb[0]] = gauge_in[0]*psi_in;
		psi_out[rb[1]] = zero;


		// Import Gauge Field
		QDPGaugeFieldToKokkosCBGaugeField(gauge_in, kokkos_gauge_e);
		// Import input halfspinor
		QDPLatticeHalfFermionToKokkosCBSpinor2(psi_in, kokkos_hspinor_in);

		KokkosMVLattice(kokkos_gauge_e, kokkos_hspinor_in, 0, kokkos_hspinor_out);

		// Export result HalfFermion
		KokkosCBSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_out,kokkos_out);
		psi_out[rb[0]] -= kokkos_out;
		double norm_diff = toDouble(sqrt(norm2(psi_out)));

		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);

	}

	// ********* TEST ADJOINT ****************
	{

		psi_out[rb[0]] = adj(gauge_in[0])*psi_in;
		psi_out[rb[1]] = zero;

		// Import input halfspinor -- Gauge still imported

		QDPLatticeHalfFermionToKokkosCBSpinor2(psi_in, kokkos_hspinor_in);
		KokkosHVLattice(kokkos_gauge_e, kokkos_hspinor_in, 0, kokkos_hspinor_out);

		// Export result HalfFermion
		KokkosCBSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_out,kokkos_out);
		psi_out[rb[0]] -= kokkos_out;

		double norm_diff = toDouble(sqrt(norm2(psi_out)));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);

	}
}
#endif

#if 0

TEST(TestKokkos, TestDslash)
{
  IndexArray latdims={{32,32,32,32}};
	int iters = 400;

	initQDPXXLattice(latdims);
	LatticeInfo info(latdims,4,3,NodeInfo());
	KokkosFineGaugeField<MGComplex<REAL32>>  kokkos_gauge(info);

	{
	  multi1d<LatticeColorMatrixF> gauge_in(n_dim);
	  for(int mu=0; mu < n_dim; ++mu) {
	    gaussian(gauge_in[mu]);
	    reunit(gauge_in[mu]);
	  }

	  // Import gauge field
	  QDPGaugeFieldToKokkosGaugeField(gauge_in, kokkos_gauge);
	  // QDP Gauge field ought to go away here

	}


	KokkosCBFineSpinor<MGComplex<REAL32>,4> kokkos_spinor_in(info,EVEN);
	KokkosCBFineSpinor<MGComplex<REAL32>,4> kokkos_spinor_out(info,ODD);
	{
	  LatticeFermionF psi_in;
	  gaussian(psi_in);

	  // Import Spinor
	  QDPLatticeFermionToKokkosCBSpinor(psi_in, kokkos_spinor_in);
	  // QDP++ LatticeFermionF should go away here.
	}

	//for(int sites_per_team=8; sites_per_team < 8192; sites_per_team *=2) {
	int sites_per_team=32;

	KokkosDslash<MGComplex<REAL32>,MGComplex<REAL32>, MGComplex<REAL32>> D(info,sites_per_team);


	for(int rep=0; rep < 3; ++rep) {
	   for(int isign=-1; isign < 2; isign+=2) {
	  
	    MasterLog(INFO, "Timing Dslash: isign == %d", isign);
	    double start_time = omp_get_wtime();
	    for(int i=0; i < iters; ++i) {
	      D(kokkos_spinor_in,kokkos_gauge,kokkos_spinor_out,isign);
	    }
	    double end_time = omp_get_wtime();
	    double time_taken = end_time - start_time;
	    
	    double rfo = 1.0;
	    double num_sites = static_cast<double>((latdims[0]/2)*latdims[1]*latdims[2]*latdims[3]);
	    double bytes_in = static_cast<double>((8*4*3*2*sizeof(REAL32)+8*3*3*2*sizeof(REAL32))*num_sites*iters);
	    double bytes_out = (1.0+rfo)*static_cast<double>(4*3*2*sizeof(REAL32)*num_sites*iters);
	    double flops = static_cast<double>(1320.0*num_sites*iters);
	    
	    MasterLog(INFO,"sites_per_team=%d Performance: %lf GFLOPS", sites_per_team,flops/(time_taken*1.0e9));
	    MasterLog(INFO,"sites_per_team=%d Effective BW: %lf GB/sec", sites_per_team,(bytes_in+bytes_out)/(time_taken*1.0e9));
	    
	    
	    
	   } // isign
	} 
        // } // -- sites_per_team
}
#endif

#if 0
TEST(TestKokkos, TestDslashVec)
{
  IndexArray latdims={{16,16,16,32}};

#ifdef MG_USE_AVX512
	int iters = 1000;
#else 
	int iters = 100;
#endif

	initQDPXXLattice(latdims);
	LatticeInfo info(latdims,4,3,NodeInfo());
	KokkosFineGaugeField<MGComplex<REAL32>>  kokkos_gauge(info);

	{
	  multi1d<LatticeColorMatrixF> gauge_in(n_dim);
	  for(int mu=0; mu < n_dim; ++mu) {
	    gaussian(gauge_in[mu]);
	    reunit(gauge_in[mu]);
	  }

	  // Import gauge field
	  QDPGaugeFieldToKokkosGaugeField(gauge_in, kokkos_gauge);
	  // QDP Gauge field ought to go away here

	}


	KokkosCBFineSpinor<SIMDComplex<REAL32,V>,4> kokkos_spinor_in(info,EVEN);
	KokkosCBFineSpinor<SIMDComplex<REAL32,V>,4> kokkos_spinor_out(info,ODD);
	{
	  multi1d<LatticeFermionF> psi_in(V);

	  for(int v=0; v < V; ++v) {
		  gaussian(psi_in[v]);
	  }
	  // Import Spinor
	  QDPLatticeFermionToKokkosCBSpinor(psi_in, kokkos_spinor_in);
	  // QDP++ LatticeFermionF should go away here.
	}


	for(int per_team=1; per_team < 512; per_team *= 2) {

	  KokkosDslash<MGComplex<REAL32>,SIMDComplex<REAL32,V>,ThreadSIMDComplex<REAL32,V>> D(info,per_team);
	for(int rep=0; rep < 2; ++rep) {
	  for(int isign=-1; isign < 2; isign+=2) {
	    MasterLog(INFO, "Sites per Team=%d Timing Dslash: isign == %d", per_team, isign);
	    double start_time = omp_get_wtime();
	    for(int i=0; i < iters; ++i) {
	      D(kokkos_spinor_in,kokkos_gauge,kokkos_spinor_out,isign);
	    }
	    double end_time = omp_get_wtime();
	    double time_taken = end_time - start_time;

	    double rfo = 1.0;
	    double num_sites = static_cast<double>((latdims[0]/2)*latdims[1]*latdims[2]*latdims[3]);
	    double bytes_in = static_cast<double>((8*4*3*2*sizeof(REAL32)*V+8*3*3*2*sizeof(REAL32))*num_sites*iters);
	    double bytes_out = (1.0+rfo)*static_cast<double>(V*4*3*2*sizeof(REAL32)*num_sites*iters);
	    double flops = static_cast<double>(1320.0*V*num_sites*iters);

	    MasterLog(INFO,"Sites Per Team=%d Performance: %lf GFLOPS", per_team, flops/(time_taken*1.0e9));
	    MasterLog(INFO,"Sites Per Team=%d Effective BW: %lf GB/sec", per_team, (bytes_in+bytes_out)/(time_taken*1.0e9));



	  }
	}
	}

}
#endif


#if 1
TEST(TestKokkos, TestDslashVecLonger)
{
  IndexArray latdims={{16,16,16,32}};

#ifdef MG_USE_AVX512
	int iters = 1000;
#else 
	int iters = 900;
#endif


	initQDPXXLattice(latdims);
	LatticeInfo info(latdims,4,3,NodeInfo());
	KokkosFineGaugeField<MGComplex<REAL32>>  kokkos_gauge(info);

	{
	  multi1d<LatticeColorMatrixF> gauge_in(n_dim);
	  for(int mu=0; mu < n_dim; ++mu) {
	    gaussian(gauge_in[mu]);
	    reunit(gauge_in[mu]);
	  }

	  // Import gauge field
	  QDPGaugeFieldToKokkosGaugeField(gauge_in, kokkos_gauge);
	  // QDP Gauge field ought to go away here

	}


	KokkosCBFineSpinor<SIMDComplex<REAL32,V>,4> kokkos_spinor_in(info,EVEN);
	KokkosCBFineSpinor<SIMDComplex<REAL32,V>,4> kokkos_spinor_out(info,ODD);
	{
	  multi1d<LatticeFermionF> psi_in(V);

	  for(int v=0; v < V; ++v) {
		  gaussian(psi_in[v]);
	  }
	  // Import Spinor
	  QDPLatticeFermionToKokkosCBSpinor(psi_in, kokkos_spinor_in);
	  // QDP++ LatticeFermionF should go away here.
	}

        int per_team = 2;
	KokkosDslash<MGComplex<REAL32>,SIMDComplex<REAL32,V>,ThreadSIMDComplex<REAL32,V>> D(info,per_team);

	for(int rep=0; rep < 2; ++rep) {
	  // for(int isign=-1; isign < 2; isign+=2) {
	    int isign=1;
	    MasterLog(INFO, "Sites per Team=%d Timing Dslash: isign == %d", per_team, isign);
	    double start_time = omp_get_wtime();
	    for(int i=0; i < iters; ++i) {
	      D(kokkos_spinor_in,kokkos_gauge,kokkos_spinor_out,isign);
	    }
	    double end_time = omp_get_wtime();
	    double time_taken = end_time - start_time;

	    double rfo = 1.0;
	    double num_sites = static_cast<double>((latdims[0]/2)*latdims[1]*latdims[2]*latdims[3]);
	    double bytes_in = static_cast<double>((8*4*3*2*sizeof(REAL32)*V+8*3*3*2*sizeof(REAL32))*num_sites*iters);
	    double bytes_out = (1.0+rfo)*static_cast<double>(V*4*3*2*sizeof(REAL32)*num_sites*iters);
	    double flops = static_cast<double>(1320.0*V*num_sites*iters);

	    MasterLog(INFO,"Sites Per Team=%d Performance: %lf GFLOPS", per_team, flops/(time_taken*1.0e9));
	    MasterLog(INFO,"Sites Per Team=%d Effective BW: %lf GB/sec", per_team, (bytes_in+bytes_out)/(time_taken*1.0e9));

	  // }
	}

}
#endif

int main(int argc, char *argv[]) 
{
	return ::MGTesting::TestMain(&argc, argv);
}

