#include "gtest/gtest.h"
#include "../test_env.h"
#include "../mock_nodeinfo.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_op.h"
#include "lattice/coarse/coarse_l1_blas.h"
#include "utils/print_utils.h"
#include "qdpxx_helpers.h"
#include "aggregate_qdpxx.h"
#include "aggregate_block_qdpxx.h"

#include "reunit.h"
#include "transf.h"

using namespace MG;
using namespace MGTesting;
using namespace QDP;

TEST(TestCoarseCoarse, TestCoarseDslashDir)
{
	IndexArray latdims={{2,2,2,2}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	multi1d<LatticeColorMatrix> u(Nd);

	QDPIO::cout << "Generating Random Gauge with Gaussian Noise" << std::endl;
	for(int mu=0; mu < Nd; ++mu) {
		//u[mu] = 1;
		gaussian(u[mu]);
		reunit(u[mu]);
	}

	multi1d<LatticeFermion> in_vecs(Nc*Ns/2);     // In terms of vectors
	multi1d<LatticePropagator> dslash_links(8); // In terms of propagators

	MasterLog(INFO,"Generating Eye\n");
	LatticePropagator eye=1;


	// Pack 'Eye' vectors into to the in_vecs;
	for(int spin=0; spin < Ns/2; ++spin) {
		for(int color =0; color < Nc; ++color) {
			LatticeFermion upper = zero;
			LatticeFermion lower = zero;

			PropToFerm(eye, lower, color, spin);
			PropToFerm(eye, upper, color, spin+Ns/2);
			in_vecs[color + Nc*spin] = upper + lower;
		}
	}


	// Next step should be to copy this into the fields needed for gauge and clover ops
	LatticeInfo info(latdims, 2, 6, NodeInfo());
	CoarseGauge u_coarse(info);


	// Generate the triple products directly into the u_coarse
	for(int mu=0; mu < 8; ++mu) {
		QDPIO::cout << " Attempting Triple Product in direction: " << mu << std::endl;
		dslashTripleProductSiteDirQDPXX(mu, u, in_vecs, u_coarse);
	}


	MasterLog(INFO,"Coarse Gauge Field initialized\n");
	// Create A coarse operator
	int n_smt = 1;
	CoarseDiracOp D_op_coarse(info, n_smt);

	for(int mu=0; mu < 8; ++mu) {
		LatticeFermion psi, d_psi, coarse_d_psi;
		gaussian(psi);

		// Apply Dslash Dir
		DslashDirQDPXX(d_psi, u, psi, mu);

		// CoarsSpinors
		CoarseSpinor coarse_s_in(info);
		CoarseSpinor coarse_s_out(info);

		// Import psi
		QDPSpinorToCoarseSpinor(psi, coarse_s_in);
		for(int cb=0; cb < n_checkerboard; ++cb) {
#pragma omp parallel
			{
				int tid=omp_get_thread_num();
				D_op_coarse.DslashDir(coarse_s_out, u_coarse, coarse_s_in, cb, mu, tid);
			}
		}

		CoarseSpinorToQDPSpinor(coarse_s_out, coarse_d_psi);
		// Find the difference between regular dslash and 'coarse' dslash
		LatticeFermion diff = d_psi - coarse_d_psi;

		QDPIO::cout << "Direction=" << mu << std::endl;
		QDPIO::cout << "Norm Diff[0] = " << sqrt(norm2(diff, rb[0])) << std::endl;
		QDPIO::cout << "Norm Diff[1] = " << sqrt(norm2(diff, rb[1])) 	<< std::endl;
		QDPIO::cout << "Norm Diff = " << sqrt(norm2(diff)) << std::endl;
		QDPIO::cout << "Rel. Norm Diff[0] = " << sqrt(norm2(diff, rb[0])/norm2(psi,rb[0])) << std::endl;
		QDPIO::cout << "Rel. Norm Diff[1] = " << sqrt(norm2(diff, rb[1])/norm2(psi,rb[1])) << std::endl;
		QDPIO::cout << "Rel. Norm Diff = " << sqrt(norm2(diff)/norm2(psi)) << std::endl;

		ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[0])) ), 0, 1.e-5 );
		ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[1])) ), 0, 1.e-5 );
		ASSERT_NEAR( toDouble( sqrt(norm2(diff)) ) , 0, 1.5e-5 );
		ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[0])/norm2(psi,rb[0])) ), 0, 1.e-5 );
		ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[1])/norm2(psi,rb[1])) ), 0, 1.e-5 );
		ASSERT_NEAR( toDouble( sqrt(norm2(diff)/norm2(psi)) ), 0, 1.e-5 );

	}

}

TEST(TestCoarseCoarse, TestCoarseDslashDir2)
{
	IndexArray latdims={{4,4,4,4}};
	IndexArray blockdims={{2,2,2,2}};

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
	QDPIO::cout << "Orthonormalizing Nullvecs" << std::endl;

	// 1) Create the blocklist
	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	CreateBlockList(my_blocks,blocked_lattice_dims,latdims,blockdims);

	// Do the proper block orthogonalize -- I do it twice... Why not
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);


	// Next step should be to copy this into the fields needed for gauge and clover ops
	LatticeInfo info(blocked_lattice_dims, 2, 6, NodeInfo());
	CoarseGauge u_coarse(info);

	// Generate the triple products directly into the u_coarse
	for(int mu=0; mu < 8; ++mu) {
		QDPIO::cout << " Attempting Triple Product in direction: " << mu << std::endl;
		dslashTripleProductDirQDPXX(my_blocks, mu, u, vecs, u_coarse);
	}

	int n_smt = 1;
	CoarseDiracOp D_op_coarse(info, n_smt);

	LatticeFermion v_f;
	gaussian(v_f);

	// Coarsen v_f to R(v_f) give us coarse RHS for tests
	CoarseSpinor v_c(info);
	QDPIO::cout << "Restricting v_f -> v_c over blocks" << std::endl;
	restrictSpinorQDPXXFineToCoarse(my_blocks, vecs, v_f, v_c);

	// Output
	CoarseSpinor coarse_d_out(info);

	// Apply Coarse Op Dslash in Threads
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		for(int cb=0; cb < n_checkerboard; ++cb) {
			D_op_coarse.Dslash(coarse_d_out, u_coarse, v_c, cb, LINOP_OP, tid);
		}
	}

	// Now apply direction by direction
	CoarseSpinor dslash_dir_out(info);
	CoarseSpinor dslash_dir_sum(info);

	ZeroVec(dslash_dir_sum);
	for(int dir=0; dir < 8; ++dir) {
		// Apply Coarse Op Dslash in Threads
#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			for(int cb=0; cb < n_checkerboard; ++cb) {
				D_op_coarse.DslashDir(dslash_dir_out, u_coarse, v_c, cb, dir, tid);
			}
		}

		float alpha = 1;
		// Accumulate
		AxpyVec(alpha, dslash_dir_out, dslash_dir_sum);
	}

	double d_out_norm = sqrt(Norm2Vec(coarse_d_out));
	double diffnorm = sqrt(XmyNorm2Vec(dslash_dir_sum,coarse_d_out));

	MasterLog(INFO, "|| direct - accumulated directions || = %16.8e", diffnorm);
	ASSERT_NEAR( diffnorm, 0, 5.e-6 );
	MasterLog(INFO, "|| direct - accumulated directions || / || direct || = %16.8e", diffnorm/d_out_norm);
	ASSERT_NEAR( (diffnorm/d_out_norm), 0, 3.0e-7 );


}


int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

