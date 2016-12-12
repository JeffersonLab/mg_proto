#include "gtest/gtest.h"
#include "aggregate_qdpxx.h"
#include "../test_env.h"
#include "../mock_nodeinfo.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "qdpxx_helpers.h"
#include "reunit.h"
#include "transf.h"
#include "clover_fermact_params_w.h"
#include "clover_term_qdp_w.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_op.h"
#include "lattice/coarse/coarse_l1_blas.h"
#include "dslashm_w.h"
#include <complex>

using namespace MG;
using namespace MGTesting;
using namespace QDP;

//
//  We want to test:  D_c v_c = ( R D_f P )
//
// This relationship should hold true always, both
// when R and P aggregate over sites or blocks of sites.
// We can use the existing functionality without blocking
// over sites to test functionality, and to test the interface.

TEST(TestCoarseQDPXXBlock, TestFakeCoarseClov)
{
	IndexArray latdims={{4,4,4,4}};   // Fine lattice. Make it 4x4x4x4 so we can block it

	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	multi1d<LatticeColorMatrix> u(Nd);

	// Initialize the gauge field
	QDPIO::cout << "Initializing Gauge with Gaussian Noise" << std::endl;
	for(int mu=0; mu < Nd; ++mu) {
		gaussian(u[mu]);
		reunit(u[mu]);
	}

	// Initialize the Clover Op
	QDPIO::cout << "Initializing The Clover Term" << std::endl;

	CloverFermActParams clparam;
	AnisoParam_t aniso;

	// Aniso prarams
	aniso.anisoP=true;
	aniso.xi_0 = 1.5;
	aniso.nu = 0.95;
	aniso.t_dir = 3;

	// Set up the Clover params
	clparam.anisoParam = aniso;

	// Some mass
	clparam.Mass = Real(0.1);

	// Some random clover coeffs
	clparam.clovCoeffR=Real(1.35);
	clparam.clovCoeffT=Real(0.8);
	QDPCloverTerm clov_qdp;
	clov_qdp.create(u,clparam);

	QDPIO::cout << "Initializing Random Null-Vectors" << std::endl;

	multi1d<LatticeFermion> vecs(6);
	for(int k=0; k < 6; ++k) {
		gaussian(vecs[k]);
	}

	// Someone once said doing this twice is good
	QDPIO::cout << "Orthonormalizing Nullvecs" << std::endl;
	orthonormalizeAggregatesQDPXX(vecs);
	orthonormalizeAggregatesQDPXX(vecs);

	QDPIO::cout << "Coarsening Clover to create D_c" << std::endl;
	LatticeInfo info(latdims, 2, 6, NodeInfo());
	CoarseClover c_clov(info);
	clovTripleProductSiteQDPXX(clov_qdp, vecs, c_clov);


	// Now create a LatticeFermion and apply both the QDP++ and the Coarse Clover
	LatticeFermion v_f;
	gaussian(v_f);

	// Coarsen v_f to R(v_f) give us coarse RHS for tests
	CoarseSpinor v_c(info);
	restrictSpinorQDPXXFineToCoarse(vecs, v_f, v_c);

	// Output
	CoarseSpinor out(info);
	CoarseSpinor fake_out(info);

	// Now evaluate  D_c v_c
	int n_smt = 1;
	CoarseDiracOp D(info,n_smt);

	// Apply Coarsened Clover
#pragma omp parallel
	{
		int tid = omp_get_thread_num();

		D.CloverApply(out, c_clov, v_c,0,tid);
		D.CloverApply(out, c_clov, v_c,1,tid);
	}

	// Now apply the fake operator:
	LatticeFermion P_v_c = zero;
	prolongateSpinorCoarseToQDPXXFine(vecs, v_c, P_v_c); // NB: This is not the same as v_f, but rather P R v_f

	// Now apply the Clover Term to form D_f P
	LatticeFermion D_f_out = zero;

	for(int cb=0; cb < 2; ++cb) {
		clov_qdp.apply(D_f_out, P_v_c, 0, cb);
	}

	// Now restrict back:
	restrictSpinorQDPXXFineToCoarse(vecs, D_f_out, fake_out);

	// We should now compare out, with fake_out. For this we need an xmy
	double norm_diff = sqrt(xmyNorm2Coarse(fake_out,out));
	double norm_diff_per_site = norm_diff / (double)fake_out.GetInfo().GetNumSites();

	MasterLog(INFO, "Diff Norm = %16.8e", norm_diff);
	MasterLog(INFO, "Diff Norm per site = %16.8e", norm_diff_per_site);
}


TEST(TestCoarseQDPXXBlock, TestFakeCoarseDslash)
{
	IndexArray latdims={{4,4,4,4}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	multi1d<LatticeColorMatrix> u(Nd);

	QDPIO::cout << "Generating Random Gauge with Gaussian Noise" << std::endl;
	for(int mu=0; mu < Nd; ++mu) {
		gaussian(u[mu]);
		reunit(u[mu]);
	}


	// Random Basis vectors
	multi1d<LatticeFermion> vecs(6);
	for(int k=0; k < 6; ++k) {
		gaussian(vecs[k]);
	}

	// Someone once said doing this twice is good
	orthonormalizeAggregatesQDPXX(vecs);
	orthonormalizeAggregatesQDPXX(vecs);


	// Next step should be to copy this into the fields needed for gauge and clover ops
	LatticeInfo info(latdims, 2, 6, NodeInfo());
	CoarseGauge u_coarse(info);

	// Generate the triple products directly into the u_coarse
	for(int mu=0; mu < 8; ++mu) {
		QDPIO::cout << " Attempting Triple Product in direction: " << mu << std::endl;
		dslashTripleProductSiteDirQDPXX(mu, u, vecs, u_coarse);
	}

	int n_smt = 1;
	CoarseDiracOp D_op_coarse(info, n_smt);

	// Now create a LatticeFermion and apply both the QDP++ and the Coarse Clover
	LatticeFermion v_f;
	gaussian(v_f);

	// Coarsen v_f to R(v_f) give us coarse RHS for tests
	CoarseSpinor v_c(info);
	restrictSpinorQDPXXFineToCoarse(vecs, v_f, v_c);

	// Output
	CoarseSpinor out(info);
	CoarseSpinor fake_out(info);

	// Apply Coarse Op Dslash in Threads
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		D_op_coarse.Dslash(out, u_coarse, v_c, 0, tid);
		D_op_coarse.Dslash(out, u_coarse, v_c, 1, tid);
	}


	// Now apply the fake operator:
	LatticeFermion P_v_c = zero;
	prolongateSpinorCoarseToQDPXXFine(vecs, v_c, P_v_c); // NB: This is not the same as v_f, but rather P R v_f

	// Now apply the Clover Term to form D_f P
	LatticeFermion D_f_out = zero;

	// Apply Dslash to both CBs, isign=1
	// Result in m_psiu
	for(int cb=0; cb < 2; ++cb) {
		dslash(D_f_out, u, P_v_c, 1, cb);
	}


	// Now restrict back: fake_out = R D_f P  v_c
	restrictSpinorQDPXXFineToCoarse(vecs, D_f_out, fake_out);

	// We should now compare out, with fake_out. For this we need an xmy
	double norm_diff = sqrt(xmyNorm2Coarse(fake_out,out));
	double norm_diff_per_site = norm_diff / (double)fake_out.GetInfo().GetNumSites();

	MasterLog(INFO, "Diff Norm = %16.8e", norm_diff);
	MasterLog(INFO, "Diff Norm per site = %16.8e", norm_diff_per_site);

}


int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

