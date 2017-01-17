#include "gtest/gtest.h"
#include "../test_env.h"
#include "../mock_nodeinfo.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_op.h"
#include "lattice/coarse/coarse_l1_blas.h"
#include "lattice/coarse/block.h"
#include "utils/print_utils.h"

#include "qdpxx_helpers.h"
#include "aggregate_qdpxx.h"
#include "aggregate_block_qdpxx.h"

#include "reunit.h"
#include "transf.h"

#include <cassert>
#include <memory>
#include <vector>

using namespace MG;
using namespace MGTesting;
using namespace QDP;

namespace MGTesting
{
	// This function will fill N_color vectors with an eye in both the
	// lower and upper chiralities.
	// Currently designed to work with only two Chiralities
	void GenerateEye( std::vector<std::shared_ptr<CoarseSpinor> >& vecs )
	{
		// Need to have at least 1 vector to get the info.
		int  num_vecs = vecs.size();
		if( num_vecs == 0 ) {
			MasterLog(ERROR, "Attempt to generate eye with 0 vectors");
		}

		const LatticeInfo& info = vecs[0]->GetInfo();


		const int num_colors = info.GetNumColors();
		const int num_spins = info.GetNumSpins();
		const int num_colorspin = num_colors*num_spins;

		if( (num_spins != 2) && (num_spins != 4) ) {
			MasterLog(ERROR, "Only 2 or 4 spin components are supporeted for generating eye");
		}

		// The number of colorspins per chiral block
		// For 2 spins, this is just the number of colors
		// For 4 spins, this will be 2x the number of colors -- ie 2 spins per chirality
		int n_per_chiral = num_colors;

		if( num_spins == 4 ) {
			n_per_chiral *= 2;
		}

		// Now to generate the eye, we need a unit matrix that is (n_per_chiral * n_per_chiral)
		// So we had better have n_per_chiral vectors
		if( num_vecs != n_per_chiral ) {
			MasterLog(ERROR, "With num_spins=%d and num_colors=%d one needs %d vectors for eye, but %d given",num_spins, num_colors, n_per_chiral, num_vecs);
		}

		int num_cbsites = info.GetNumCBSites();

		// Loop through the vecs
		for(int vec=0; vec < num_vecs; ++vec ) {
			CoarseSpinor& v = *(vecs[vec]); // Grab the spinor to work with.

		// OK ready to go
#pragma omp parallel for collapse(2)
			for(int cb=0; cb < n_checkerboard; ++cb) {
				for(int cbsite=0; cbsite < num_cbsites; ++cbsite) {

					// Site data is now a flat pointer to num_colors * num_spins
					// which should be equal to   n_per_chiral * 2
					// with 2 chiralities
					float *site_data = v.GetSiteDataPtr(cb,cbsite);

					// Zero the site data: 2*n_per_chiral components=
					for(int component = 0; component < num_colorspin; ++component ) {
							site_data[ RE + n_complex*component ] = 0;
							site_data[ IM + n_complex*component ] = 0;
					}

					// Set the vec-th data to 1. Ie Re=1, Imag stays 0
					// Upper Chirality
					site_data[RE + n_complex*vec] = 1;

					// Lower Chirality
					site_data[RE + n_complex*vec + n_complex*n_per_chiral ] = 1;

				}
			}
		}
	}

}

// Test the Generate Eye function
// Generate Eye for 6 colors 2 spins
// Compare with eye-generated with QDP++ Propagator
TEST(TestCoarseCoarse, TestGenerateEye)
{
	IndexArray latdims={{2,2,2,2}};
	initQDPXXLattice(latdims);

	LatticeInfo info(latdims, 2,6, NodeInfo());

	// Nothing stored here yet
	std::vector< std::shared_ptr< CoarseSpinor > > vecs;

	// Make shared should initialize the shared pointer and allocate my coarse spinor
	// with info
	for(int v=0; v < 6; ++v) {
		vecs.push_back( std::make_shared<CoarseSpinor>(info) );
	}

	// Now generate the eye.
	GenerateEye( vecs );

	// Now generate the QDP++ eye
	LatticePropagator qdp_eye = 1;

	// Check that these guys are the same
	for(int spin=0; spin < Ns/2; ++spin) {
		for(int color =0; color < Nc; ++color) {

			LatticeFermion upper = zero;
			LatticeFermion lower = zero;

			PropToFerm(qdp_eye, lower, color, spin);
			PropToFerm(qdp_eye, upper, color, spin+Ns/2);

			LatticeFermion prop_vector = upper+lower;


			LatticeFermion from_coarse;
			CoarseSpinorToQDPSpinor(*(vecs[color + Nc*spin]), from_coarse);

			LatticeFermion diff = prop_vector - from_coarse;
			double norm_diff = toDouble(sqrt(norm2(diff)));
			double norm_diff_rel = toDouble(norm2(diff)/sqrt(norm2(prop_vector)));

			MasterLog(INFO, "Check Eye: spin=%d color=%d norm_diff=%16.8e norm_diff_rel=%16.8e",
					spin,color,norm_diff,norm_diff_rel);

			ASSERT_NEAR( norm_diff_rel,0,1.0e-5);

		}
	}

}

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
	IndexArray node_orig=NodeInfo().NodeCoords();
	for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

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
	CreateBlockList(my_blocks,blocked_lattice_dims,latdims,blockdims,node_orig);

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

