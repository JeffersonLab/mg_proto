#include "gtest/gtest.h"
#include "../test_env.h"
#include "../mock_nodeinfo.h"
#include "qdpxx_utils.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_op.h"
#include "lattice/coarse/coarse_l1_blas.h"
#include "lattice/coarse/block.h"
#include "utils/print_utils.h"
#include "lattice/fine_qdpxx/wilson_clover_linear_operator.h"
#include "lattice/coarse/coarse_wilson_clover_linear_operator.h"
#include "lattice/fine_qdpxx/qdpxx_helpers.h"
#include "lattice/fine_qdpxx/aggregate_qdpxx.h"

#include "lattice/fine_qdpxx/aggregate_block_qdpxx.h"
#include "lattice/coarse/aggregate_block_coarse.h"
#include "lattice/fine_qdpxx/transf.h"
#include "lattice/fine_qdpxx/dslashm_w.h"

#include <cassert>
#include <memory>
#include <vector>
#include <cmath>
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
	LatticeInfo info(latdims, 2,6, NodeInfo());
	initQDPXXLattice(latdims);

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
	LatticeInfo info(latdims, 2,6, NodeInfo());

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
	multi1d<LatticePropagator> dslash_links(9); // In terms of propagators

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
	initQDPXXLattice(latdims);

	IndexArray blockdims={{2,2,2,2}};
	IndexArray node_orig=NodeInfo().NodeCoords();
	for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

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
	IndexArray blocked_lattice_origin;
	CreateBlockList(my_blocks,blocked_lattice_dims,blocked_lattice_origin,latdims,blockdims,node_orig);

	// Do the proper block orthogonalize -- I do it twice... Why not
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);


	// Next step should be to copy this into the fields needed for gauge and clover ops
	LatticeInfo info(blocked_lattice_origin, blocked_lattice_dims, 2, 6, NodeInfo());
	CoarseGauge u_coarse(info);
	ZeroGauge(u_coarse);

	// This will generate the 8 gauge links, and a central clover like link
	// But does not coarsen the original clover
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
	// This will apply the central 'self induced clover term' too.
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		for(int cb=0; cb < n_checkerboard; ++cb) {
			D_op_coarse.unprecOp(coarse_d_out, u_coarse, v_c, cb, LINOP_OP, tid);
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

	// Now I need to add in the clover term too.
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		for(int cb=0; cb < n_checkerboard; ++cb) {
			D_op_coarse.CloverApply(dslash_dir_out,u_coarse,v_c,cb,LINOP_OP,tid);
		}
	}

	// Accumulate
	float alpha = 1;
	AxpyVec(alpha, dslash_dir_out, dslash_dir_sum);

	// Now subtract away the result of the operator, from a self accumulated.
	double d_out_norm = sqrt(Norm2Vec(coarse_d_out));
	double diffnorm = sqrt(XmyNorm2Vec(dslash_dir_sum,coarse_d_out));

	MasterLog(INFO, "|| direct - accumulated directions || = %16.8e", diffnorm);
	ASSERT_NEAR( diffnorm, 0, 5.e-6 );
	MasterLog(INFO, "|| direct - accumulated directions || / || direct || = %16.8e", diffnorm/d_out_norm);
	ASSERT_NEAR( (diffnorm/d_out_norm), 0, 3.0e-7 );


}


// Create a blocked dirac op from fine op: This creates self coupling
//   so D_coarse = X + sum_mu Y_mu    X is the self coupling
//
//   then make a D_coarse_coarse by coarsening Y
//   so  D_coarse_coarse = Y' + sum_mu Y'_mu
//
// Since we use 'eye' we should have Y'_mu = Y_mu
// Since the second blocking is trivial, Y' should be zero.
//
// NB: D_coarse - D_coarse_coarse = X
//
//     So we can check by y = D_coarse x,  y' = D_coarse_coarse_x
//     then tmp = D_coarse.CloverApply(x)  =>   tmp = X x
//      y'+= tmp
//     then we should have y'=y
TEST(TestCoarseCoarse, TestCoarseTripleProductDslashEyeTrivial)
{
	IndexArray latdims={{4,4,4,4}};
	initQDPXXLattice(latdims);

	IndexArray blockdims={{2,2,2,2}};
	IndexArray blockdims2={{1,1,1,1}}; // Trivial blocking from coarse-to-coarse

	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	IndexArray node_orig=NodeInfo().NodeCoords();
	for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

	// Create fine gauge field
	multi1d<LatticeColorMatrix> u(Nd);

	QDPIO::cout << "Generating Random Gauge with Gaussian Noise" << std::endl;
	for(int mu=0; mu < Nd; ++mu) {
		gaussian(u[mu]);
		reunit(u[mu]);
	}

	// Create Fine Basis 1
	multi1d<LatticeFermion> vecs(6);
	for(int k=0; k < 6; ++k) {
		gaussian(vecs[k]);
	}

	// Someone once said doing this twice is good
	QDPIO::cout << "Orthonormalizing Nullvecs" << std::endl;

	// 1) Create the blocklist
	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	IndexArray blocked_lattice_origin;
	CreateBlockList(my_blocks,blocked_lattice_dims,blocked_lattice_origin,latdims,blockdims,node_orig);

	// Do the proper block orthogonalize -- I do it twice... Why not
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);


	// Next step should be to copy this into the fields needed for gauge and clover ops
	LatticeInfo info(blocked_lattice_origin, blocked_lattice_dims, 2, 6, NodeInfo());
	CoarseGauge u_coarse(info);
	ZeroGauge(u_coarse);

	// Generate the triple products directly into the u_coarse
	// This generates a central term: X
	for(int mu=0; mu < 8; ++mu) {
		QDPIO::cout << " Attempting Triple Product in direction: " << mu << std::endl;
		dslashTripleProductDirQDPXX(my_blocks, mu, u, vecs, u_coarse);
	}

	// Now we have u_coarse

	// We will need a coarse Dirac_op
	int n_smt = 1;
	CoarseDiracOp D_op_coarse(info, n_smt);


	// We will need some basis vectors. I will use 6 again because we are trying to map an identity
	std::vector< std::shared_ptr<CoarseSpinor> > coarse_basis_vecs(6);
	for(int i=0; i < 6; ++i) {
		coarse_basis_vecs[i] = std::make_shared<CoarseSpinor>(info);
		ZeroVec( *(coarse_basis_vecs[i]) );
	}

	// Now Generate Eye:
	GenerateEye( coarse_basis_vecs );
	for(int i=0; i < 6; ++i) {
		double normvec=Norm2Vec( *(coarse_basis_vecs[i]) );
		double normvec_per_chiral = normvec/(2* info.GetNumSites() );

		// MasterLog(INFO, "norm vec[%d]=%16.8e", i,normvec);
		MasterLog(INFO, "norm vec[%d] per aggregate = %16.8e", i, normvec_per_chiral);
	}

	// Generate the blocking
	std::vector<Block> coarse_blocks;
	IndexArray blocked_coarse_lattice_dims;
	IndexArray blocked_coarse_lattice_origin;
	CreateBlockList(coarse_blocks, blocked_coarse_lattice_dims, blocked_coarse_lattice_origin, blocked_lattice_dims, blockdims2, node_orig);

	MasterLog(INFO, "Coarse Blocks has size=%d", coarse_blocks.size());
	ASSERT_EQ( info.GetNumSites(), coarse_blocks.size() );

	// Check site list is the same
	for(int block_cb=0; block_cb < n_checkerboard; ++block_cb) {
		for(int block_cbsite=0; block_cbsite < info.GetNumCBSites(); ++block_cbsite) {

			int block = block_cbsite + block_cb*info.GetNumCBSites();

			auto sitelist = coarse_blocks[block].getCBSiteList();

			// I'd like this blocklist to be identical to the other blocklist;
			for(int blocksite = 0; blocksite < sitelist.size(); ++blocksite ) {
				CBSite& cbsite = sitelist[blocksite];
//				MasterLog(INFO, "Block cb=%d block_cbsite=%d cb=%d site=%d",
//						block_cb, block_cbsite, cbsite.cb, cbsite.site);
				ASSERT_EQ(block_cb, cbsite.cb);
				ASSERT_EQ(block_cbsite, cbsite.site);
			}

		}
	}

	// Check orthonormality of each block:
	for(IndexType block_idx=0; block_idx < coarse_blocks.size(); ++block_idx) {
		const Block& the_block = coarse_blocks[ block_idx ];
		for(IndexType vector=0; vector < static_cast<IndexType>(coarse_basis_vecs.size()); ++vector) {

			for(IndexType vec_prev=0; vec_prev < vector; ++vec_prev) {

				for(IndexType chiral=0; chiral < 2; ++chiral ) {
					std::complex<double> iprod = innerProductBlockAggr(*(coarse_basis_vecs[vec_prev]),*(coarse_basis_vecs[vector]), the_block, chiral);
					ASSERT_NEAR( std::norm(iprod), 0, 1.0e-6);
				}
			}
			for(IndexType chiral=0; chiral < 2; ++chiral ) {
				double norm2_b= norm2BlockAggr(*(coarse_basis_vecs[vector]), the_block, chiral);
				ASSERT_NEAR( norm2_b, 1, 1.0e-6);
			}

		}
	}


	LatticeInfo coarse_coarse_info(blocked_coarse_lattice_origin,blocked_coarse_lattice_dims,2,6,NodeInfo());
	CoarseGauge u_coarse_coarse(coarse_coarse_info);
	ZeroGauge(u_coarse_coarse);
	for( int mu=0; mu < 8; ++mu) {
		QDPIO::cout << "Attempting Coarse Triple Product in direction: " << mu << std::endl;
		dslashTripleProductDir(D_op_coarse, coarse_blocks, mu, u_coarse, coarse_basis_vecs, u_coarse_coarse);
	}

	int num_colorspin=u_coarse.GetNumColorSpin();
	MasterLog(INFO, "Checking 8 links for equality, and central link for 0");
	for(int cb=0; cb < n_checkerboard; ++cb) {
		for(int site = 0; site < info.GetNumCBSites(); ++site) {
			for(int mu=0; mu < 8 ; ++mu) {

				const float* link_orig = u_coarse.GetSiteDirDataPtr(cb,site,mu);
				const float* link_new = u_coarse_coarse.GetSiteDirDataPtr(cb,site,mu);


				for(int j=0; j < num_colorspin*num_colorspin; ++j) {
					double diff_re = std::fabs( link_orig[RE+n_complex*j] - link_new[RE+n_complex*j]);
					double diff_im = std::fabs( link_orig[IM+n_complex*j] - link_new[IM+n_complex*j]);

					if ( ( diff_re > 5.0e-6) || (diff_im > 5.0e-6) )  {
					MasterLog(INFO, "dir=%d cb=%d site=%d idx=%d old=(%16.8e,%16.8e) new=(%16.8e, %16.8e) diff=(%16.8e,%16.8e)",
							mu,cb,site,j, link_orig[RE+n_complex*j], link_orig[IM+n_complex*j],
							link_new[RE+n_complex*j], link_new[IM + n_complex*j],
							diff_re,diff_im);
					}
					ASSERT_NEAR( diff_re, 0, 5.0e-6);
					ASSERT_NEAR( diff_im, 0, 5.0e-6);
				}



			}

			// Check new coupling term is 0
			const float *central_new = u_coarse_coarse.GetSiteDiagDataPtr(cb,site);
			for(int j=0; j < num_colorspin*num_colorspin; ++j) {
				ASSERT_NEAR( central_new[RE+n_complex*j], 0, 5.0e-6);
				ASSERT_NEAR( central_new[IM+n_complex*j], 0, 5.0e-6);
			}
		}

	}

	// Now want to test applying CoarseCoarseOp

	// Make a new dirac op.
	CoarseDiracOp D_op_coarse_coarse(coarse_coarse_info,1);

	// First of all these two dirac ops should now be the same.
	// The Info's should all be the same
	AssertCompatible( info, coarse_coarse_info );

	// Second Let us make some test vectors:
	CoarseSpinor psi(info);
	Gaussian(psi);
	CoarseSpinor D_c_psi(info);
	CoarseSpinor D_cc_psi(info);
	CoarseSpinor X_psi(info);

	ZeroVec(D_c_psi);
	ZeroVec(D_cc_psi);
	ZeroVec(X_psi);

	for(int op = LINOP_OP; op <= LINOP_DAGGER; ++op) {

		for(int cb=0; cb < n_checkerboard; ++cb) {
#pragma omp parallel
			{
				int tid = omp_get_thread_num();
				D_op_coarse.unprecOp(D_c_psi, u_coarse, psi, cb,op, tid );
				D_op_coarse.CloverApply(X_psi,u_coarse,psi,cb,op,tid);
			}

#pragma omp parallel
			{
				int tid = omp_get_thread_num();
				D_op_coarse_coarse.unprecOp(D_cc_psi, u_coarse_coarse, psi, cb, op, tid);
			}
		} // cb

		float alpha=1;
		// Accumulate: ass X psi onto the CoarseCoarse operator
		AxpyVec(alpha, X_psi, D_cc_psi);

		double norm_psi = sqrt(Norm2Vec(psi));
		double norm_Dop = sqrt(Norm2Vec(D_c_psi));
		double norm_Dc_op = sqrt(Norm2Vec(D_cc_psi));

		MasterLog( INFO, " || psi_in || = %16.8e  || D_op_psi_in || = %16.8e || D_op_coarse ||=%16.8e", norm_psi, norm_Dop, norm_Dc_op);

		double norm_diff = sqrt(XmyNorm2Vec(D_cc_psi, D_c_psi));

		MasterLog(INFO, "Op=%d diff=%16.8e relative_diff=%16.8e",op,norm_diff, norm_diff/norm_Dop);
		ASSERT_NEAR( norm_diff, 0, 6.0e-6);
	}

}

TEST(TestCoarseCoarse, TestCoarseTripleProductCloverEyeTrivial)
{
	IndexArray latdims={{4,4,4,4}};
	IndexArray blockdims={{2,2,2,2}};
	IndexArray blockdims2={{1,1,1,1}}; // Trivial blocking from coarse-to-coarse



	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	IndexArray node_orig=NodeInfo().NodeCoords();
	for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

	// Create fine gauge field
	multi1d<LatticeColorMatrix> u(Nd);

	QDPIO::cout << "Generating Random Gauge with Gaussian Noise" << std::endl;
	for(int mu=0; mu < Nd; ++mu) {
		gaussian(u[mu]);
		reunit(u[mu]);
	}

	// Create Fine Basis 1
	multi1d<LatticeFermion> vecs(6);
	for(int k=0; k < 6; ++k) {
		gaussian(vecs[k]);
	}

	// Someone once said doing this twice is good
	QDPIO::cout << "Orthonormalizing Nullvecs" << std::endl;

	// 1) Create the blocklist
	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	IndexArray blocked_lattice_origin;
	CreateBlockList(my_blocks,blocked_lattice_dims, blocked_lattice_origin, latdims,blockdims,node_orig);

	// Do the proper block orthogonalize -- I do it twice... Why not
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);


	// Now need to make a clover op
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

	// Next step should be to copy this into the fields needed for gauge and clover ops
	LatticeInfo info(blocked_lattice_origin,blocked_lattice_dims, 2, 6, NodeInfo());
	CoarseGauge coarse_clover(info);
	ZeroGauge(coarse_clover);
	QDPIO::cout << " Attempting Clover Tripe Product"<<  std::endl;

	// This coarsens into coarse_clover[8]
	clovTripleProductQDPXX(my_blocks,clov_qdp, vecs, coarse_clover);



	int n_smt = 1;
	CoarseDiracOp D_op_coarse(info, n_smt);

	QDPIO::cout << "Generating Coarse basis vectors" << std::endl;
	// We will need some basis vectors. I will use 6 again because we are trying to map an identity
	std::vector< std::shared_ptr<CoarseSpinor> > coarse_basis_vecs(6);
	for(int i=0; i < 6; ++i) {
		coarse_basis_vecs[i] = std::make_shared<CoarseSpinor>(info);
		ZeroVec( *(coarse_basis_vecs[i]) );
	}

	QDPIO::cout << "Generating Coarse Eye" << std::endl;
	// Now Generate Eye:
	GenerateEye( coarse_basis_vecs );


	// Generate the blocking
	QDPIO::cout << "Generating Coarse-Coarse Blocking" << std::endl;
	std::vector<Block> coarse_blocks;
	IndexArray blocked_coarse_lattice_dims;
	IndexArray blocked_coarse_lattice_origin;
	CreateBlockList(coarse_blocks, blocked_coarse_lattice_dims, blocked_coarse_lattice_origin, blocked_lattice_dims, blockdims2, node_orig);



	LatticeInfo coarse_coarse_info(blocked_coarse_lattice_origin, blocked_coarse_lattice_dims,2,6,NodeInfo());
	CoarseGauge coarse_coarse_clover(coarse_coarse_info);
	ZeroGauge(coarse_coarse_clover);

	QDPIO::cout << "Attempting Coarse-Coarse Triple Product in direction: " << std::endl;
	clovTripleProduct(D_op_coarse,
			coarse_blocks,
			coarse_clover,
			coarse_basis_vecs, coarse_coarse_clover);


	QDPIO::cout << "Comparing Coarse Clover with new CoarseCoarse CLover...";
	for(int cb=0; cb < n_checkerboard; ++cb) {
		for(int site = 0; site < info.GetNumCBSites(); ++site) {

				const float* clov_orig = coarse_clover.GetSiteDiagDataPtr(cb,site);
				const float* clov_new = coarse_coarse_clover.GetSiteDiagDataPtr(cb,site);

				int num_colorspin=info.GetNumColorSpins();
				for(int j=0; j < num_colorspin*num_colorspin; ++j) {
					float old_re = clov_orig[RE + n_complex*j];
					float old_im = clov_orig[IM + n_complex*j];

					float new_re = clov_new[RE + n_complex*j];
					float new_im = clov_new[IM + n_complex*j];

					ASSERT_NEAR( old_re, new_re, 5.0e-6);
					ASSERT_NEAR( old_im, new_im, 5.0e-6);
				} // j in clover
		} // site
	} // cb
	QDPIO::cout << "OK" << std::endl;

	// Make a new dirac op.
	CoarseDiracOp D_op_coarse_coarse(coarse_coarse_info,n_smt);

	// First of all these two dirac ops should now be the same.
	// The Info's should all be the same
	AssertCompatible( info, coarse_coarse_info );

	// Second Let us make some test vectors:
	CoarseSpinor psi(info);
	Gaussian(psi);
	CoarseSpinor Clov_c_psi(info);
	CoarseSpinor Clov_cc_psi(info);

	ZeroVec(Clov_c_psi);
	ZeroVec(Clov_cc_psi);

	// Now check the operators ApplyClover against each other.
	// They should be equal.
	for(int op = LINOP_OP; op <= LINOP_DAGGER; ++op) {
		QDPIO::cout << "Op=" << ((op == LINOP_OP) ? "LINOP " : "DAGGER" ) << std::endl;
		QDPIO::cout << "   Aoplying Coarse Clover" << std::endl;
		for(int cb=0; cb < n_checkerboard; ++cb) {
#pragma omp parallel
			{
				int tid = omp_get_thread_num();
				D_op_coarse.CloverApply(Clov_c_psi, coarse_clover, psi, cb,op, tid );
			}
		}

		QDPIO::cout << "   Aoplying Coarse Coarse Clover" << std::endl;
		for(int cb=0; cb < n_checkerboard; ++cb) {
#pragma omp parallel
			{
				int tid = omp_get_thread_num();
				D_op_coarse_coarse.CloverApply(Clov_cc_psi, coarse_coarse_clover, psi, cb, op, tid);
			}
		} // cb

		double norm_coarse_clov = sqrt(Norm2Vec(Clov_c_psi));
		double norm_coarse_coarse_clov = sqrt(Norm2Vec(Clov_cc_psi));
		double norm_diff = sqrt(XmyNorm2Vec(Clov_cc_psi, Clov_c_psi));

		MasterLog(INFO, "Op=%d || Clov_c_psi || =%16.8e || Clov_cc_psi || = %16.8e diff=%16.8e relative_diff=%16.8e",op,norm_coarse_clov,norm_coarse_coarse_clov, norm_diff, norm_diff/norm_coarse_clov);
		ASSERT_NEAR( norm_diff, 0, 1.0e-6);
	}
}

TEST(TestCoarseCoarse, TestCoarseProlongRestrictTrivial)
{
	IndexArray latdims={{4,4,4,4}};
	IndexArray blockdims={{1,1,1,1}}; // Trivial blocking from coarse-to-coarse

	IndexArray node_orig=NodeInfo().NodeCoords();
	for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

	const int N_colors = 6;
	LatticeInfo info(latdims, 2, N_colors, NodeInfo());

	MasterLog(INFO, "Generating Basis Vecs");
	std::vector<std::shared_ptr<CoarseSpinor>> vecs(6);
	for(int vec=0;vec<N_colors;++vec) {
		vecs[vec] = std::make_shared<CoarseSpinor>(info);
		Gaussian( *(vecs[vec]) );
	}

	// Someone once said doing this twice is good
	QDPIO::cout << "Creating Blocklist and Orthonormalizing Vecs" << std::endl;

	// 1) Create the blocklist -- this is a trivial blocking
	//    The effect of a restriction is a unitary rotation
	//    essentially.

	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	IndexArray blocked_lattice_origin;
	CreateBlockList(my_blocks,blocked_lattice_dims, blocked_lattice_origin,latdims,blockdims,node_orig);

	// Do the proper block orthogonalize -- I do it twice... Why not
	orthonormalizeBlockAggregates(vecs, my_blocks);
	orthonormalizeBlockAggregates(vecs, my_blocks);

	// Now I should be able to restrict and prolongate back
	// These are essentially inverse operations as long as
	// the number of vecs is equal to the number of colors

	// Next step should be to copy this into the fields needed for gauge and clover ops
	LatticeInfo coarse_info(blocked_lattice_origin, blocked_lattice_dims, 2, N_colors, NodeInfo());

	CoarseSpinor psi( info );
	CoarseSpinor R_psi( coarse_info );
	CoarseSpinor PR_psi( info );

	Gaussian( psi );
	restrictSpinor(my_blocks,vecs, psi, R_psi );
	prolongateSpinor(my_blocks, vecs, R_psi, PR_psi);

	double norm_psi = sqrt( Norm2Vec(psi) );
	double norm_RPpsi = sqrt( Norm2Vec(PR_psi) );
	double norm_diff = sqrt(XmyNorm2Vec(PR_psi,psi));
	MasterLog(INFO, " || (P R - 1) psi || = %16.8e", norm_diff);
	MasterLog(INFO, " || (P R - 1) psi ||/site = %16.8e", norm_diff / info.GetNumSites());
	ASSERT_LT( norm_diff, 8.0e-6);

}

TEST(TestCoarseCoarse, TestCoarseRestrictProlong)
{
	IndexArray latdims={{8,8,8,8}};
	IndexArray blockdims={{2,2,2,2}}; // Trivial blocking from coarse-to-coarse

	IndexArray node_orig=NodeInfo().NodeCoords();
	for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

	const int N_colors = 6;
	LatticeInfo info(latdims, 2, N_colors, NodeInfo());

	MasterLog(INFO, "Generating Basis Vecs");
	std::vector<std::shared_ptr<CoarseSpinor>> vecs(6);
	for(int vec=0;vec<N_colors;++vec) {
		vecs[vec] = std::make_shared<CoarseSpinor>(info);
		Gaussian( *(vecs[vec]) );
	}

	// Someone once said doing this twice is good
	QDPIO::cout << "Creating Blocklist and Orthonormalizing Vecs" << std::endl;

	// 1) Create the blocklist -- this is a trivial blocking
	//    The effect of a restriction is a unitary rotation
	//    essentially.

	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	IndexArray blocked_lattice_origin;
	CreateBlockList(my_blocks,blocked_lattice_dims, blocked_lattice_origin,latdims,blockdims,node_orig);

	// Do the proper block orthogonalize -- I do it twice... Why not
	orthonormalizeBlockAggregates(vecs, my_blocks);
	orthonormalizeBlockAggregates(vecs, my_blocks);

	// Now I should be able to restrict and prolongate back
	// These are essentially inverse operations as long as
	// the number of vecs is equal to the number of colors

	// Next step should be to copy this into the fields needed for gauge and clover ops
	LatticeInfo coarse_info(blocked_lattice_origin,blocked_lattice_dims, 2, N_colors, NodeInfo());

	CoarseSpinor psi( coarse_info );
	CoarseSpinor P_psi( info );
	CoarseSpinor RP_psi( coarse_info );

	Gaussian( psi );
	prolongateSpinor(my_blocks,vecs, psi, P_psi);
	restrictSpinor(my_blocks,vecs, P_psi, RP_psi );


	double norm_psi = sqrt( Norm2Vec(psi) );
	double norm_RPpsi = sqrt( Norm2Vec(RP_psi) );
	double norm_diff = sqrt(XmyNorm2Vec(RP_psi,psi));
	MasterLog(INFO, " || (R P - 1) psi || = %16.8e", norm_diff);
	MasterLog(INFO, " || (R P - 1) psi ||/site = %16.8e", norm_diff / info.GetNumSites());
	ASSERT_LT( norm_diff, 2.0e-5);

}


TEST(TestCoarseCoarse, TestCoarseCoarseDslashClov)
{
	IndexArray latdims={{8,8,8,8}};
	IndexArray blockdims={{2,2,2,2}};
	IndexArray blockdims2={{2,2,2,2}}; // Trivial blocking from coarse-to-coarse

	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	IndexArray node_orig=NodeInfo().NodeCoords();
	for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

	// Create fine gauge field
	multi1d<LatticeColorMatrix> u(Nd);
	QDPIO::cout << "Generating Random Gauge with Gaussian Noise" << std::endl;
	for(int mu=0; mu < Nd; ++mu) {
		gaussian(u[mu]);
		reunit(u[mu]);
	}

	QDPWilsonCloverLinearOperator D_fine(0.1, 1.35, 1, u );


	// Create Fine Basis 1 -- choose not 6 vectors
	int N_color_1 = 8;
	multi1d<LatticeFermion> vecs(N_color_1);
	for(int k=0; k < N_color_1; ++k) {
		gaussian(vecs[k]);
	}

	// Someone once said doing this twice is good
	QDPIO::cout << "Orthonormalizing Nullvecs" << std::endl;

	// 1) Create the blocklist
	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	IndexArray blocked_lattice_origin;
	CreateBlockList(my_blocks,blocked_lattice_dims,blocked_lattice_origin,latdims,blockdims,node_orig);

	// Do the proper block orthogonalize -- I do it twice... Why not
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);

	QDPIO::cout << "Creating Level 1 Coarse Gauge Field " << std::endl;
	// Next step should be to copy this into the fields needed for gauge and clover ops
	LatticeInfo info(blocked_lattice_origin,blocked_lattice_dims, 2, N_color_1, NodeInfo());
	std::shared_ptr<CoarseGauge> u_coarse = std::make_shared<CoarseGauge>(info);
	ZeroGauge(*u_coarse);

	D_fine.generateCoarse(my_blocks,vecs,*u_coarse);
	CoarseWilsonCloverLinearOperator D_coarse(u_coarse, 1);


	{
		CoarseSpinor  psi(info);
		Gaussian(psi);
		CoarseSpinor  Dc_psi(info);
		ZeroVec(Dc_psi);


		QDPIO::cout << "Applying Coarse Op (inc self genenerated + clover term" << std::endl;
		D_coarse(Dc_psi,psi,LINOP_OP);


		QDPIO::cout << "Applying Fake Op" << std::endl;
		LatticeFermion P_psi = zero;
		LatticeFermion DP_psi = zero;
		CoarseSpinor RDP_psi(info);
		ZeroVec(RDP_psi);

		prolongateSpinorCoarseToQDPXXFine( my_blocks, vecs, psi, P_psi);
		D_fine(DP_psi,P_psi,LINOP_OP);
		restrictSpinorQDPXXFineToCoarse( my_blocks, vecs,DP_psi,RDP_psi);

		double norm_RDP_psi = sqrt(Norm2Vec(RDP_psi));
		double norm_Dc_psi = sqrt(Norm2Vec(Dc_psi));
		QDPIO::cout << " || RDP psi || = " << norm_RDP_psi << std::endl;
		QDPIO::cout << " || Dc psi  || = " << norm_Dc_psi << std::endl;

		double norm_diff = sqrt(XmyNorm2Vec( RDP_psi, Dc_psi ));

		double norm_diff_per_site = norm_diff / info.GetNumSites();
		QDPIO::cout << "|| (RDP - D_c) R_psi || =  " << norm_diff << std::endl;
		QDPIO::cout << "|| (RDP - D_c) R_psi || / site =  " << norm_diff_per_site << std::endl;
		ASSERT_LT( norm_diff_per_site, 3.0e-6);
	}
    // Time to make coarse coarse

    int N_color_2 =16;
	MasterLog(INFO, "Generating L2 Basis Vecs");
	std::vector<std::shared_ptr<CoarseSpinor>> vecs_l2(N_color_2);
	for(int vec=0;vec<N_color_2;++vec) {
		MasterLog(INFO, "   Doing vec=%d\n", vec);
		vecs_l2[vec] = std::make_shared<CoarseSpinor>(info);
		Gaussian( *(vecs_l2[vec]) );
	}

	// 1) Create the blocklist
	std::vector<Block> my_blocks_l2;
	IndexArray blocked_lattice_dims_l2;
	IndexArray blocked_lattice_orig_12;
	CreateBlockList(my_blocks_l2,blocked_lattice_dims_l2,blocked_lattice_orig_12, blocked_lattice_dims,blockdims2,node_orig);

	// Do the proper block orthogonalize -- I do it twice... Why not
	orthonormalizeBlockAggregates(vecs_l2, my_blocks_l2);
	orthonormalizeBlockAggregates(vecs_l2, my_blocks_l2);

	LatticeInfo info_l2(blocked_lattice_orig_12, blocked_lattice_dims_l2, 2, N_color_2, NodeInfo());
	std::shared_ptr<CoarseGauge> u_coarse_coarse=std::make_shared<CoarseGauge>(info_l2);
	D_coarse.generateCoarse(my_blocks_l2, vecs_l2, *u_coarse_coarse);


	CoarseWilsonCloverLinearOperator D_coarse_coarse(u_coarse_coarse,2);
	CoarseSpinor  psi_2(info_l2); // Coarse Coarse
	CoarseSpinor  out_2(info_l2); // Coarse Coarse

	CoarseSpinor  Ppsi_2(info);   // Coarse
	CoarseSpinor  DPpsi_2(info);  // Coarse
	CoarseSpinor  RDPpsi_2(info_l2); // Coarse Coarse

	// Fill Psi2 with random.
	Gaussian( psi_2 );
	//ZeroVec(out_2 );
	D_coarse_coarse(out_2,psi_2,LINOP_OP);

	//ZeroVec(Ppsi_2);
	//ZeroVec(DPpsi_2);
	//ZeroVec(RDPpsi_2);

	prolongateSpinor(my_blocks_l2, vecs_l2, psi_2, Ppsi_2);
	D_coarse(DPpsi_2, Ppsi_2, LINOP_OP);
	restrictSpinor(my_blocks_l2, vecs_l2, DPpsi_2, RDPpsi_2);

	double norm2_Dcc_psi=Norm2Vec(out_2);
	double norm2_RDcP_psi=Norm2Vec(RDPpsi_2);
	MasterLog(INFO, "|| D_cc psi || = %16.8e", sqrt(norm2_Dcc_psi));
	MasterLog(INFO, "|| R D_c P psi || = %16.8e", sqrt(norm2_RDcP_psi));

	double normDiff = sqrt(XmyNorm2Vec(RDPpsi_2,out_2));
	MasterLog(INFO, "|| (D_cc - R D_c P) psi = %16.8e\n", normDiff);

	double normDiffPerSite = normDiff/(info_l2.GetNumSites());
	MasterLog(INFO, "|| (D_cc - R D_c P) psi = %16.8e\n", normDiffPerSite);

	ASSERT_LE( normDiffPerSite, 5.0e-6);

}


int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

