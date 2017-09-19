#include "gtest/gtest.h"
#include "../test_env.h"
#include "../mock_nodeinfo.h"
#include "../qdpxx/qdpxx_utils.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/fine_qdpxx/dslashm_w.h"
#include "./kokkos_vectype.h"
#include "./kokkos_vnode.h"
#include "./kokkos_vtypes.h"
#include "./kokkos_defaults.h"
#include "./kokkos_qdp_vutils.h"
#include "./kokkos_vspinproj.h"
#include "./kokkos_vmatvec.h"
#include "./kokkos_vdslash.h"
#include <omp.h>
#include "MG_config.h"

using namespace MG;
using namespace MGTesting;
using namespace QDP;


#ifdef KOKKOS_HAVE_CUDA
static constexpr int V = 1;
#else
static constexpr int V = 8;
#endif
using namespace QDP;
using namespace MG;
using namespace MGTesting;

#if 0
TEST(TestKokkos, TestDslash)
{
	IndexArray latdims={{32,32,32,32}};
	int iters = 100;

	initQDPXXLattice(latdims);
	multi1d<LatticeColorMatrix> gauge_in(n_dim);
	for(int mu=0; mu < n_dim; ++mu) {
		gaussian(gauge_in[mu]);
		reunit(gauge_in[mu]);
	}

	LatticeFermion psi_in=zero;
	gaussian(psi_in);

	LatticeInfo info(latdims,4,3,NodeInfo());

	using VN = VNode<MGComplex<REAL32>,V>;
	using SpinorType = KokkosCBFineVSpinor<MGComplex<REAL32>,VN,4>;
	using GaugeType = KokkosFineVGaugeField<MGComplex<REAL32>,VN>;

	SpinorType  kokkos_spinor_even(info,EVEN);
	SpinorType  kokkos_spinor_odd(info,ODD);
	GaugeType  kokkos_gauge(info);


	// Import Gauge Field
	QDPGaugeFieldToKokkosVGaugeField(gauge_in, kokkos_gauge);

	// Import spinor
	QDPLatticeFermionToKokkosCBVSpinor(psi_in, kokkos_spinor_even);

    for(int per_team=1; per_team < 2048; per_team *=2 ) {

		KokkosVDslash<VN,
		MGComplex<REAL32>,
		MGComplex<REAL32>,
		ThreadSIMDComplex<REAL32,VN::VecLen>,
		ThreadSIMDComplex<REAL32,VN::VecLen> > D(kokkos_spinor_even.GetInfo(),per_team);

		MasterLog(INFO, "per_team=%d", per_team);

		for(int isign=-1; isign < 2; isign+=2) {
			// Time it.
			double start_time = omp_get_wtime();
			for(int i=0; i < iters; ++i) {
				D(kokkos_spinor_even,kokkos_gauge,kokkos_spinor_odd,isign);
			}
			double end_time = omp_get_wtime();
			double time_taken = end_time - start_time;

			double rfo = 1.0;
			double num_sites = static_cast<double>((latdims[0]/2)*latdims[1]*latdims[2]*latdims[3]);
			double bytes_in = static_cast<double>((8*4*3*2*sizeof(REAL32)+8*3*3*2*sizeof(REAL32))*num_sites*iters);
			double bytes_out = static_cast<double>(4*3*2*sizeof(REAL32)*num_sites*iters);
			double rfo_bytes_out = (1.0 + rfo)*bytes_out;
			double flops = static_cast<double>(1320.0*num_sites*iters);

			MasterLog(INFO,"Sites Per Team=%d isign=%d Performance: %lf GFLOPS", per_team, isign, flops/(time_taken*1.0e9));
			MasterLog(INFO,"Sites Per Team=%d isign=%d Effective BW (RFO=0): %lf GB/sec", per_team, isign, (bytes_in+bytes_out)/(time_taken*1.0e9));
			MasterLog(INFO,"Sites Per Team=%d isign=%d Effective BW (RFO=1): %lf GB/sec", per_team, isign, (bytes_in+rfo_bytes_out)/(time_taken*1.0e9));



		} // isign
	} // per team
}
#endif

TEST(TestKokkos, TestDslashTime)
{
	IndexArray latdims={{32,32,32,32}};
	int iters=400;

	initQDPXXLattice(latdims);
	multi1d<LatticeColorMatrix> gauge_in(n_dim);
	for(int mu=0; mu < n_dim; ++mu) {
		gaussian(gauge_in[mu]);
		reunit(gauge_in[mu]);
	}

	LatticeFermion psi_in=zero;
	gaussian(psi_in);

	LatticeInfo info(latdims,4,3,NodeInfo());

	using VN = VNode<MGComplex<REAL32>,V>;
	using SpinorType = KokkosCBFineVSpinor<MGComplex<REAL32>,VN,4>;
	using FullGaugeType = KokkosFineVGaugeField<MGComplex<REAL32>,VN>;
	using GaugeType = KokkosCBFineVGaugeFieldDoubleCopy<MGComplex<REAL32>,VN>;

	SpinorType  kokkos_spinor_even(info,EVEN);
	SpinorType  kokkos_spinor_odd(info,ODD);
	FullGaugeType  kokkos_gauge(info);



	// Import Gauge Field
	QDPGaugeFieldToKokkosVGaugeField(gauge_in, kokkos_gauge);


	// Double Store Gauge field. This benchmark is always even cb.
	GaugeType  gauge_even(info,EVEN);


	// Import gets the rear neighbors, and permutes them if needed
	gauge_even.import( kokkos_gauge(EVEN), kokkos_gauge(ODD));

	// Import spinor
	QDPLatticeFermionToKokkosCBVSpinor(psi_in, kokkos_spinor_even);

	// for(int per_team=1; per_team < 256; per_team *=2 ) {

	KokkosVDslash<VN,
	MGComplex<REAL32>,
	MGComplex<REAL32>,
	SIMDComplex<REAL32,VN::VecLen>,
		      SIMDComplex<REAL32,VN::VecLen> > D(kokkos_spinor_even.GetInfo());

	IndexArray cb_latdims = kokkos_spinor_even.GetInfo().GetCBLatticeDimensions();
	double num_sites = static_cast<double>(V*cb_latdims[0]*cb_latdims[1]*cb_latdims[2]*cb_latdims[3]);

#if 0
	int titers=100;
	double best_flops = 0;
	IndexArray best_blocks={1,1,1,1};
	for(int t=cb_latdims[3]; t >= 1; t /= 2) {
		for(int z=cb_latdims[2]; z >= 1; z /= 2) {
			for(int y=cb_latdims[1]; y >= 1; y/=2 ) {
				for(int x = cb_latdims[0]; x >= 1; x/= 2 ) {
					int isign=1;
					int num_blocks = 1;
					num_blocks *= cb_latdims[0]/x;
					num_blocks *= cb_latdims[1]/y;
					num_blocks *= cb_latdims[2]/z;
					num_blocks *= cb_latdims[3]/t;
					if ( num_blocks >= 128) {
						double start_time = omp_get_wtime();
						for(int i=0; i < titers; ++i) {
						  D(kokkos_spinor_even,gauge_even,kokkos_spinor_odd,isign,{x,y,z,t});
						}
						double end_time = omp_get_wtime();
						double time_taken = end_time - start_time;
						double flops = static_cast<double>(1320.0*num_sites*titers);
						double floprate = flops/(time_taken*1.0e9);
						MasterLog(INFO,"Tuning: (Bx,By,Bz,Bt)=(%d,%d,%d,%d) GFLOPS=%lf", x,y,z,t,floprate);
						if (floprate > best_flops){
							best_flops = floprate;
							best_blocks[0]=x;
							best_blocks[1]=y;
							best_blocks[2]=z;
							best_blocks[3]=t;
						}
					}


				}
			}
		}
	}
#else
	IndexArray best_blocks={4,2,2,16};
#endif
	MasterLog(INFO, "Main timing: (Bx,By,Bz,Bt)=(%d,%d,%d,%d)",
				best_blocks[0],best_blocks[1],best_blocks[2],best_blocks[3]);

	for(int rep=0; rep < 10; ++rep ) {
		int isign = 1;
		//for(int isign=-1; isign < 2; isign+=2) {
			// Time it.
			double start_time = omp_get_wtime();
			for(int i=0; i < iters; ++i) {
				D(kokkos_spinor_even,gauge_even,kokkos_spinor_odd,isign, best_blocks);
			}
			double end_time = omp_get_wtime();
			double time_taken = end_time - start_time;

			double rfo = 1.0;
			double num_sites = static_cast<double>((latdims[0]/2)*latdims[1]*latdims[2]*latdims[3]);
			double bytes_in = static_cast<double>((8*4*3*2*sizeof(REAL32)+8*3*3*2*sizeof(REAL32))*num_sites*iters);
			double bytes_out = static_cast<double>(4*3*2*sizeof(REAL32)*num_sites*iters);
			double rfo_bytes_out = (1.0 + rfo)*bytes_out;
			double flops = static_cast<double>(1320.0*num_sites*iters);

			MasterLog(INFO,"isign=%d Performance: %lf GFLOPS", isign, flops/(time_taken*1.0e9));
			MasterLog(INFO,"isign=%d Effective BW (RFO=0): %lf GB/sec",isign, (bytes_in+bytes_out)/(time_taken*1.0e9));
			MasterLog(INFO,"isign=%d Effective BW (RFO=1): %lf GB/sec",  isign, (bytes_in+rfo_bytes_out)/(time_taken*1.0e9));



		// } // isign
		MasterLog(INFO,"");
	} // rep
}


int main(int argc, char *argv[]) 
{
	return ::MGTesting::TestMain(&argc, argv);
}

