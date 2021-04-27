#include <omp.h>
#include "gtest/gtest.h"
#include "../test_env.h"
#include "../mock_nodeinfo.h"
#include "qdpxx_utils.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/geometry_utils.h"

#include "lattice/coarse/coarse_op.h"
#include "lattice/coarse/coarse_l1_blas.h"
#include "lattice/coarse/block.h"

#include "lattice/fine_qdpxx/qdpxx_helpers.h"
#include "lattice/fine_qdpxx/transf.h"
#include "lattice/fine_qdpxx/clover_fermact_params_w.h"
#include "lattice/fine_qdpxx/clover_term_qdp_w.h"
#include "lattice/fine_qdpxx/dslashm_w.h"
#include <complex>

// Site stuff
#include "lattice/fine_qdpxx/aggregate_qdpxx.h"

// Block Stuff
#include "lattice/fine_qdpxx/aggregate_block_qdpxx.h"

using namespace MG;
using namespace MGTesting;
using namespace QDP;


#include <iostream>



TEST(TestCoarseQDPXXBlock, TestBlockOrthogonalize)
{
	IndexArray latdims={{4,4,4,4}};   // Fine lattice. Make it 4x4x4x4 so we can block it
	IndexArray blockdims = {{2,2,2,2}};

	initQDPXXLattice(latdims);

	IndexArray node_orig=NodeInfo().NodeCoords();
	for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	// 1) Create the blocklist
	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	IndexArray blocked_lattice_orig;
	CreateBlockList(my_blocks,blocked_lattice_dims,blocked_lattice_orig, latdims,blockdims,node_orig);

	// 2) Create the test vectors
	multi1d<LatticeFermion> vecs(6);
	for(int vec=0; vec < 6; ++vec) {
			gaussian(vecs[vec]);
	}

	// Orthonormalize -- do it twice just to make sure
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);

	for(int block_idx=0; block_idx < my_blocks.size(); ++block_idx) {

		// Check orthonormality for each block
		const Block& block = my_blocks[block_idx];

		// Check orthonormality separately for each aggregate
		for(int aggr=0; aggr < 2; ++aggr ) {

			// Check normalization:
			for(int curr_vec = 0; curr_vec < 6; ++curr_vec) {


				for(int test_vec = 0; test_vec < 6; ++test_vec ) {

					if( test_vec != curr_vec ) {
						MasterLog(DEBUG, "Checking inner product of pair (%d,%d), block=%d aggr=%d", curr_vec,test_vec, block_idx,aggr);
						std::complex<double> iprod = innerProductBlockAggrQDPXX(vecs[test_vec],vecs[curr_vec],block, aggr);
						ASSERT_NEAR( real(iprod), 0, 1.0e-15);
						ASSERT_NEAR( imag(iprod), 0, 1.0e-15);

					}
					else {

						std::complex<double> iprod = innerProductBlockAggrQDPXX(vecs[test_vec],vecs[curr_vec], block, aggr);
						ASSERT_NEAR( real(iprod), 1, 1.0e-15);
						ASSERT_NEAR( imag(iprod), 0, 1.0e-15);

						MasterLog(DEBUG, "Checking norm2 of vector %d block=%d aggr=%d", curr_vec, block_idx,aggr);
						double norm = sqrt(norm2BlockAggrQDPXX(vecs[curr_vec],block,aggr));
						ASSERT_NEAR(norm, 1, 1.0e-15);

					}
				}
			}
		}
	}


}

#if 0
TEST(TestCoarseQDPXXBlock, TestBlockOrthogonalize8)
{
	IndexArray latdims={{8,8,8,8}};   // Fine lattice. Make it 4x4x4x4 so we can block it
	IndexArray blockdims = {{2,2,2,2}};

	initQDPXXLattice(latdims);

	IndexArray node_orig=NodeInfo().NodeCoords();
	for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	// 1) Create the blocklist
	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	CreateBlockList(my_blocks,blocked_lattice_dims,latdims,blockdims,node_orig);

	// 2) Create the test vectors
	multi1d<LatticeFermion> vecs(6);
	for(int vec=0; vec < 6; ++vec) {
			gaussian(vecs[vec]);
	}

	// Orthonormalize -- do it twice just to make sure
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);

	for(int block_idx=0; block_idx < my_blocks.size(); ++block_idx) {

		// Check orthonormality for each block
		const Block& block = my_blocks[block_idx];

		// Check orthonormality separately for each aggregate
		for(int aggr=0; aggr < 2; ++aggr ) {

			// Check normalization:
			for(int curr_vec = 0; curr_vec < 6; ++curr_vec) {


				for(int test_vec = 0; test_vec < 6; ++test_vec ) {

					if( test_vec != curr_vec ) {
						MasterLog(DEBUG, "Checking inner product of pair (%d,%d), block=%d aggr=%d", curr_vec,test_vec, block_idx,aggr);
						std::complex<double> iprod = innerProductBlockAggrQDPXX(vecs[test_vec],vecs[curr_vec],block, aggr);
						ASSERT_NEAR( real(iprod), 0, 1.0e-15);
						ASSERT_NEAR( imag(iprod), 0, 1.0e-15);

					}
					else {

						std::complex<double> iprod = innerProductBlockAggrQDPXX(vecs[test_vec],vecs[curr_vec], block, aggr);
						ASSERT_NEAR( real(iprod), 1, 1.0e-15);
						ASSERT_NEAR( imag(iprod), 0, 1.0e-15);

						MasterLog(DEBUG, "Checking norm2 of vector %d block=%d aggr=%d", curr_vec, block_idx,aggr);
						double norm = sqrt(norm2BlockAggrQDPXX(vecs[curr_vec],block,aggr));
						ASSERT_NEAR(norm, 1, 1.0e-15);

					}
				}
			}
		}
	}


}
#endif


// THis tests fails if the number of theads is more than the number of sites?
TEST(TestCoarseQDPXXBlock,TestOrthonormal2)
{
	IndexArray latdims={{2,2,2,2}};   // Fine lattice. Make it 4x4x4x4 so we can block it
	IndexArray blockdims = {{1,1,1,1}}; // Trivial blocking -- can check against site variant

	initQDPXXLattice(latdims);

	IndexArray node_orig=NodeInfo().NodeCoords();
	for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	// 1) Create the blocklist
	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	IndexArray blocked_lattice_orig;
	CreateBlockList(my_blocks,blocked_lattice_dims, blocked_lattice_orig, latdims,blockdims, node_orig);

	// 2) Create the test vectors
	multi1d<LatticeFermion> vecs(6);
	multi1d<LatticeFermion> compare_vecs(6);
	for(int vec=0; vec < 6; ++vec) {
			gaussian(vecs[vec]);
			compare_vecs[vec] = vecs[vec];
	}

	// Do the proper block orthogonalize
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);

	// Do the site oritented one...
	orthonormalizeAggregatesQDPXX(compare_vecs);
	orthonormalizeAggregatesQDPXX(compare_vecs);

	// Site oriented orthonormalization should agree with the
	// Block one for this case.
	for(IndexType vec=0; vec < 6; ++vec) {
		Double ndiff = norm2(compare_vecs[vec] -vecs[vec]);

		QDPIO::cout << "Ndiff["<<vec<<"] = " << ndiff << std::endl;
		ASSERT_NEAR( toDouble(ndiff), 0, 1.0e-28);
	}
}


TEST(TestCoarseQDPXXBlock, TestRestrictorTrivial)
{
	IndexArray latdims={{2,2,2,2}};   // Fine lattice. Make it 4x4x4x4 so we can block it
	IndexArray blockdims = {{1,1,1,1}}; // Trivial blocking -- can check against site variant

	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	IndexArray node_orig=NodeInfo().NodeCoords();
		for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];
	// 1) Create the blocklist
	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	IndexArray blocked_lattice_orig;
	CreateBlockList(my_blocks,blocked_lattice_dims, blocked_lattice_orig, latdims,blockdims, node_orig);

	// 2) Create the test vectors
	multi1d<LatticeFermion> vecs(6);
	multi1d<LatticeFermion> compare_vecs(6);
	for(int vec=0; vec < 6; ++vec) {
			gaussian(vecs[vec]);
			compare_vecs[vec] = vecs[vec];
	}

	// Do the proper block orthogonalize
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);

	// Do the site oritented one...
	orthonormalizeAggregatesQDPXX(compare_vecs);
	orthonormalizeAggregatesQDPXX(compare_vecs);

	for(IndexType vec=0; vec < 6; ++vec) {
		Double ndiff = norm2(compare_vecs[vec] -vecs[vec]);
		QDPIO::cout << "Ndiff["<<vec<<"] = " << ndiff << std::endl;
	}


    LatticeInfo info(blocked_lattice_orig, blocked_lattice_dims, 2, 6, NodeInfo());
	CoarseSpinor coarse_block(info);
	CoarseSpinor coarse_site(info);


	LatticeFermion fine_in;

	gaussian(fine_in);

	// Restrict -- this should be just like packing
	restrictSpinorQDPXXFineToCoarse(compare_vecs,fine_in,coarse_site);
	restrictSpinorQDPXXFineToCoarse(my_blocks,vecs,fine_in,coarse_block);


	for(int cb=0; cb < n_checkerboard; ++cb) {
		for(int cbsite=0; cbsite < info.GetNumCBSites(); ++cbsite) {

			float* coarse_site_cursite = coarse_site.GetSiteDataPtr(0,cb,cbsite);
			float* coarse_block_cursite = coarse_block.GetSiteDataPtr(0,cb,cbsite);
			// Loop over the components - contiguous NumColorspin x n_complex
			for(int comp = 0; comp < n_complex*coarse_block.GetNumColorSpin(); ++comp) {
				QDPIO::cout << "cb="<< cb << " site=" <<  cbsite << " component = " << comp
							<< " coarse_site=" << coarse_site_cursite[comp] << " coarse_block=" << coarse_block_cursite[comp] << std::endl;

				ASSERT_NEAR( coarse_site_cursite[comp], coarse_block_cursite[comp], 1.0e-6);

			}
		}
	}

}


TEST(TestCoarseQDPXXBlock, TestProlongatorTrivial)
{
	IndexArray latdims={{2,2,2,2}};   // Fine lattice. Make it 4x4x4x4 so we can block it
	IndexArray blockdims = {{1,1,1,1}}; // Trivial blocking -- can check against site variant

	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	IndexArray node_orig=NodeInfo().NodeCoords();
		for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];
	// 1) Create the blocklist
	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	IndexArray blocked_lattice_orig;
	CreateBlockList(my_blocks,blocked_lattice_dims,blocked_lattice_orig,latdims,blockdims, node_orig);

	// 2) Create the test vectors
	multi1d<LatticeFermion> vecs(6);
	for(int vec=0; vec < 6; ++vec) {
			gaussian(vecs[vec]);
	}

	// Do the proper block orthogonalize
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);

    LatticeInfo info(blocked_lattice_orig,blocked_lattice_dims, 2, 6, NodeInfo());
	CoarseSpinor coarse_block(info);

	LatticeFermion fine_in;
	gaussian(fine_in);

	restrictSpinorQDPXXFineToCoarse(my_blocks,vecs,fine_in,coarse_block);

	// Now prolongate back both the site version and the non site version

	LatticeFermion fine_out1;
	LatticeFermion fine_out2;
	gaussian(fine_out1); // Fill with junk to make sure everything gets overwritten by prolongator
	gaussian(fine_out2); // Fill with junk to make sure everything gets overwritten by prolongator

	prolongateSpinorCoarseToQDPXXFine(vecs,coarse_block, fine_out1);
	prolongateSpinorCoarseToQDPXXFine(my_blocks, vecs, coarse_block, fine_out2);

	LatticeFermion diff = fine_out2 - fine_out1;

	for(int site=0; site < Layout::vol(); ++site) {
		for(int spin=0; spin < Ns; ++spin) {
			for(int color=0; color < Nc; ++color) {
				ASSERT_NEAR(
											fine_out1.elem(site).elem(spin).elem(color).real(),
											fine_out2.elem(site).elem(spin).elem(color).real(), 1.0e-6);

				ASSERT_NEAR(
											fine_out1.elem(site).elem(spin).elem(color).imag(),
											fine_out2.elem(site).elem(spin).elem(color).imag(), 1.0e-6);


			}
		}
	}

}

TEST(TestCoarseQDPXXBlock, TestCoarseQDPXXDslashTrivial)
{
	IndexArray latdims={{2,2,2,2}};
	IndexArray blockdims={{1,1,1,1}};

	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	IndexArray node_orig=NodeInfo().NodeCoords();
	for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

	multi1d<LatticeColorMatrix> u(Nd);

	QDPIO::cout << "Generating Random Gauge with Gaussian Noise" << std::endl;
	for(int mu=0; mu < Nd; ++mu) {
		//u[mu] = 1;
		gaussian(u[mu]);
		reunit(u[mu]);
	}


	// Random Basis vectors
	multi1d<LatticeFermion> vecs(6);
	for(int k=0; k < 6; ++k) {
		gaussian(vecs[k]);
	}

	// 1) Create the blocklist
		std::vector<Block> my_blocks;
		IndexArray blocked_lattice_dims;
		IndexArray blocked_lattice_orig;
		CreateBlockList(my_blocks,blocked_lattice_dims,blocked_lattice_orig,latdims,blockdims, node_orig);

		// Do the proper block orthogonalize
			orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
			orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);

	// Next step should be to copy this into the fields needed for gauge and clover ops
	LatticeInfo info(blocked_lattice_orig,blocked_lattice_dims, 2, 6, NodeInfo());
	CoarseGauge u_coarse(info);
	ZeroGauge(u_coarse);
	// Generate the triple products directly into the u_coarse
	for(int mu=0; mu < 8; ++mu) {
		QDPIO::cout << " Attempting Triple Product in direction: " << mu << std::endl;
		dslashTripleProductDirQDPXX(my_blocks, mu, u, vecs, u_coarse);
	}


	LatticeFermion psi, d_psi, m_psi;

	gaussian(psi);

	for(int op=LINOP_OP; op <=LINOP_DAGGER; op++) {
	m_psi = zero;

	int isign = ( op == LINOP_OP ) ? 1 : -1;

	// Fine version:  m_psi_f =  D_f  psi_f
	// Apply Dslash to both CBs, isign=1
	// Result in m_psiu
	for(int cb=0; cb < 2; ++cb) {


		dslash(m_psi, u, psi, isign, cb);
	}

	// CoarsSpinors
	CoarseSpinor coarse_s_in(info);
	CoarseSpinor coarse_s_out(info);

	restrictSpinorQDPXXFineToCoarse(my_blocks, vecs, psi, coarse_s_in);


	// Create A coarse operator
	int n_smt = 1;
	CoarseDiracOp D_op_coarse(info, n_smt);

	// Apply Coarse Op Dslash in Threads
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		D_op_coarse.unprecOp(coarse_s_out, u_coarse, coarse_s_in, 0, op, tid);
		D_op_coarse.unprecOp(coarse_s_out, u_coarse, coarse_s_in, 1, op, tid);
	}

	// Export Coa            rse spinor to QDP++ spinors.
	LatticeFermion coarse_d_psi = zero;

	// Prolongate to form coarse_d_psi = P D_c R psi_f
	prolongateSpinorCoarseToQDPXXFine(my_blocks,vecs, coarse_s_out, coarse_d_psi);

	// Check   D_f psi_f = P D_c R psi_f
	LatticeFermion diff = m_psi - coarse_d_psi;

	QDPIO::cout << "OP=" << op << std::endl;
	QDPIO::cout << "Norm Diff[0] = " << sqrt(norm2(diff, rb[0])) << std::endl;
	QDPIO::cout << "Norm Diff[1] = " << sqrt(norm2(diff, rb[1])) << std::endl;
	QDPIO::cout << "Norm Diff = " << sqrt(norm2(diff)) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[0] = " << sqrt(norm2(diff, rb[0])/norm2(psi,rb[0])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[1] = " << sqrt(norm2(diff, rb[1])/norm2(psi,rb[1])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff = " << sqrt(norm2(diff)/norm2(psi)) << std::endl;

	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[0])) ), 0, 5.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[1])) ), 0, 5.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff)) ) , 0, 5.e-5);
	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[0])/norm2(psi,rb[0])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[1])/norm2(psi,rb[1])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff)/norm2(psi)) ), 0, 1.e-5 );
	} // op
}

TEST(TestCoarseQDPXXBlock, TestCoarseQDPXXClovTrivial)
{
	IndexArray latdims={{2,2,2,2}};
	IndexArray blockdims={{1,1,1,1}};

	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	IndexArray node_orig=NodeInfo().NodeCoords();
		for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];
	multi1d<LatticeColorMatrix> u(Nd);

	QDPIO::cout << "Generating Random Gauge with Gaussian Noise" << std::endl;
	for(int mu=0; mu < Nd; ++mu) {
		//		u[mu] = 1;
		gaussian(u[mu]);
		reunit(u[mu]);
	}

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

	multi1d<LatticeFermion> vecs(6);
	for(int k=0; k < 6; ++k) {
		gaussian(vecs[k]);
	}

	// 1) Create the blocklist
	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	IndexArray blocked_lattice_orig;
	CreateBlockList(my_blocks,blocked_lattice_dims, blocked_lattice_orig, latdims,blockdims, node_orig);

	// Do the proper block orthogonalize
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);

	QDPIO::cout << "Coarsening Clover" << std::endl;

	LatticeInfo info(blocked_lattice_orig, blocked_lattice_dims, 2, 6, NodeInfo());
	CoarseGauge c_clov(info);
	ZeroGauge(c_clov);

	clovTripleProductQDPXX(my_blocks, clov_qdp, vecs, c_clov);

	// Now create a LatticeFermion and apply both the QDP++ and the Coarse Clover
	LatticeFermion orig;
	gaussian(orig);

	for(int op=LINOP_OP; op <= LINOP_DAGGER; ++op) {


	LatticeFermion orig_res=zero;
	int isign = ( op == LINOP_OP ) ? 1 : -1 ;

	// Apply QDP++ clover
	for(int cb=0; cb < 2; ++cb) {
		clov_qdp.apply(orig_res, orig, isign, cb);
	}

	// Convert original spinor to a coarse spinor
	CoarseSpinor s_in(info);

	// Restrict using orthonormal basis
	restrictSpinorQDPXXFineToCoarse(my_blocks, vecs, orig, s_in);

	// Output
	CoarseSpinor s_out(info);

	int n_smt = 1;
	CoarseDiracOp D(info,n_smt);

	// Apply Coarsened Clover
#pragma omp parallel
	{
		int tid = omp_get_thread_num();

		D.CloverApply(s_out, c_clov, s_in,0,op,tid);
		D.CloverApply(s_out, c_clov, s_in,1,op,tid);
	}

	LatticeFermion coarse_res;
	prolongateSpinorCoarseToQDPXXFine(my_blocks, vecs, s_out, coarse_res);


	LatticeFermion diff = orig_res - coarse_res;

	QDPIO::cout << "OP=" << op << std::endl;
	QDPIO::cout << "Norm Diff[0] = " << sqrt(norm2(diff, rb[0])) << std::endl;
	QDPIO::cout << "Norm Diff[1] = " << sqrt(norm2(diff, rb[1])) 	<< std::endl;
	QDPIO::cout << "Norm Diff = " << sqrt(norm2(diff)) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[0] = " << sqrt(norm2(diff, rb[0])/norm2(orig,rb[0])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[1] = " << sqrt(norm2(diff, rb[1])/norm2(orig,rb[1])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff = " << sqrt(norm2(diff)/norm2(orig)) << std::endl;

	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[0])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[1])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff)) ) , 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[0])/norm2(orig,rb[0])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[1])/norm2(orig,rb[1])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff)/norm2(orig)) ), 0, 1.e-5 );

	} // op

}


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
	IndexArray blockdims = {{2,2,2,2}};

	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	IndexArray node_orig=NodeInfo().NodeCoords();
		for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];
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

	// 1) Create the blocklist
	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	IndexArray blocked_lattice_orig;
	CreateBlockList(my_blocks,blocked_lattice_dims,blocked_lattice_orig,latdims,blockdims, node_orig);

	// Do the proper block orthogonalize
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);

	QDPIO::cout << "Coarsening Clover with Triple Product to create D_c" << std::endl;
	LatticeInfo info(blocked_lattice_orig,blocked_lattice_dims, 2, 6, NodeInfo());
	CoarseGauge c_clov(info);
	ZeroGauge(c_clov);

	clovTripleProductQDPXX(my_blocks, clov_qdp, vecs, c_clov);

	for(int op = LINOP_OP; op <= LINOP_DAGGER; ++op ) {

		int isign= (op == LINOP_OP ) ? +1 : -1;

	// Now create a LatticeFermion and apply both the QDP++ and the Coarse Clover
	LatticeFermion v_f;
	gaussian(v_f);

	// Coarsen v_f to R(v_f) give us coarse RHS for tests
	CoarseSpinor v_c(info);

	QDPIO::cout << "Creating v_c = R v_f, with v_f gaussian" << std::endl;
	restrictSpinorQDPXXFineToCoarse(my_blocks, vecs, v_f, v_c);

	// Output
	CoarseSpinor out(info);
	CoarseSpinor fake_out(info);

	QDPIO::cout << "Applying Clov_c: out = D_c v_c" <<std::endl;
	// Now evaluate  D_c v_c
	int n_smt = 1;
	CoarseDiracOp D(info,n_smt);

	// Apply Coarsened Clover
#pragma omp parallel
	{
		int tid = omp_get_thread_num();

		D.CloverApply(out, c_clov, v_c,0,op,tid);
		D.CloverApply(out, c_clov, v_c,1,op,tid);
	}

	// Now apply the fake operator:
	LatticeFermion P_v_c = zero;
	QDPIO::cout << "Prolongating; P v_c " << std::endl;
	prolongateSpinorCoarseToQDPXXFine(my_blocks, vecs, v_c, P_v_c); // NB: This is not the same as v_f, but rather P R v_f

	QDPIO::cout << "Applying: Clov_f P v_c" << std::endl;
	// Now apply the Clover Term to form D_f P
	LatticeFermion D_f_out = zero;
	for(int cb=0; cb < n_checkerboard; ++cb) {
		clov_qdp.apply(D_f_out, P_v_c, isign, cb);
	}

	// Now restrict back:
	QDPIO::cout << "Restricting: out = R Clov_f P v_c" << std::endl;
	restrictSpinorQDPXXFineToCoarse(my_blocks, vecs, D_f_out, fake_out);

	QDPIO::cout << "Checking Clov_c v_c = R Clov_f P v_c. " << std::endl;
	// We should now compare out, with fake_out. For this we need an xmy
	double norm_diff = sqrt(XmyNorm2Vec(fake_out,out)[0]);
	double norm_diff_per_site = norm_diff / (double)fake_out.GetInfo().GetNumSites();

	MasterLog(INFO, "OP = %d", op);
	MasterLog(INFO, "Diff Norm = %16.8e", norm_diff);
	ASSERT_NEAR( norm_diff, 0, 2.e-5 );
	MasterLog(INFO, "Diff Norm per site = %16.8e", norm_diff_per_site);
	ASSERT_NEAR( norm_diff_per_site,0,1.e-6);
	}
}


// Test against pencil and paper
TEST(TestCoarseQDPXXBlock, TestTripleProductT1)
{
	IndexArray latdims={{2,2,2,4}};
	IndexArray blockdims={{1,1,1,1}};
	IndexArray node_orig=NodeInfo().NodeCoords();
	for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];
	initQDPXXLattice(latdims);

	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	QDPIO::cout << "Generating Unit Gauge" << std::endl;

	// Taking Unit Gauge here to remove color from the inner products.
	// Just check spin structure for now.
	multi1d<LatticeColorMatrix> u(Nd);
	for(int mu=0; mu < Nd; ++mu) u[mu] = 1;

	// Now I want to create 6 vectors
	multi1d<LatticeFermion> vecs(6);

	for(int k=0; k < 6; ++k) {
		vecs[k] = zero;
	}

	// Fill out the vector that the spinors for t=0 are  [ (1 0 0), (2 0 0), (3,0,0), (4,0,0) ] on all spatial sites
	//                                          t=1 are  [ (5,0,0), (6,0,0), (7,0,0), (8,0,0) ] on all spatial sites
	//                                          t=2 are  [ (9,0,0), (10,0,0), (11,0,0), (12,0,0) ] on all spatial sites
	//                                          t=3 are  [ (13,0,0), (14,0,0), (15,0,0), (16,0,0) ] on all spatial sites
	//
	// I.e. the number on generic t, spin is 1 + 4*t + spin


	for(int site=0; site < QDP::all.numSiteTable(); ++site) {
		// Get the t_coordinate of the site
		multi1d<int> coord = Layout::siteCoords(Layout::nodeNumber(), site);
		for(int spin=0; spin < 4; ++spin) {
			int idx = 1 + 4*coord[3] + spin;
			for(int k=0; k < 6; ++k) {
				vecs[k].elem(QDP::all.start() + site).elem(spin).elem(0).real() = idx;
			}
		}
	}

	// Check the vectors
	for(int t=0; t < 2; ++t) {
		multi1d<int> coord(4);
		coord[0]=0; coord[1]=0; coord[2]=0; coord[3]=t;
		int site=Layout::linearSiteIndex(coord);
		QDPIO::cout << "Coord=(0,0,0,"<<t<<") site =" << site <<  "  vec= [ " ;
		for(int spin=0; spin < 4; ++spin) {
			for(int color=0; color < 3; ++color ) {
				QDPIO::cout <<" ( " << vecs[0].elem(site).elem(spin).elem(color).real() << " , " <<
							vecs[0].elem(site).elem(spin).elem(color).imag()  <<" ) ";

			}
		}
		QDPIO::cout << " ] " <<  std::endl;
	}
	// 1) Create the blocklist
	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	IndexArray blocked_lattice_orig;
	CreateBlockList(my_blocks,blocked_lattice_dims,blocked_lattice_orig,latdims,blockdims, node_orig);

	// Next step should be to copy this into the fields needed for gauge and clover ops
	LatticeInfo info(blocked_lattice_orig,blocked_lattice_dims, 2, 6, NodeInfo());
	CoarseGauge u_coarse(info);
	ZeroGauge(u_coarse);
	QDPIO::cout << " Attempting Triple Product in  T+ direction (6): "<< std::endl;
	dslashTripleProductDirQDPXX(my_blocks, 6, u, vecs, u_coarse);

	// Now check the gauge links on site 0 (x,y,z,t)=0,0,0,0
	// matrix should be
	//     A   B
	//     C   D
	// where A is a 6x6 matrix  with all elements -6 and B is a matrix with all elements 4
	for(int cb=0; cb < n_checkerboard; ++cb) {
		for(int cbsite=0; cbsite < info.GetNumCBSites(); ++cbsite) {



			// turn cb, cbsite into coordinate:
			IndexArray coords;
			CBIndexToCoords(cbsite,cb, blocked_lattice_dims, node_orig, coords);

			int compareA = 0;
			int compareB = 0;
			int compareC = 0;
			int compareD = 0;

			int t=coords[3];
			switch(t)  {
			case 0:
			{
				compareA = 17;
				compareB = -23;
				compareC = -39;
				compareD = 53;
			}
			break;
			case 1:
			{
				compareA = 105;
				compareB = -127;
				compareC = -143;
				compareD = 173;
			}
			break;
			case 2:
			{
				compareA = 257;
				compareB = -295;
				compareC = -311;
				compareD = 357;
			}
			break;
			case 3:
			{
				compareA = 41;
				compareB = -95;
				compareC = -47;
				compareD = 109;

			}
			break;
			default:
				break;
			};
			int num_color = info.GetNumColors();
			int num_colorspin = 2*num_color;

			// chiral row = 0; chiral col = 0;
			float* u_coarse_data = u_coarse.GetSiteDirDataPtr(cb,cbsite,6);
			for(int col=0; col < num_color; ++col) {
			  for(int row=0; row < num_color; ++row) {

			    // A Block (
					ASSERT_FLOAT_EQ( u_coarse_data[ RE + n_complex*(row+ num_colorspin*col) ], (float)(compareA) );
					ASSERT_FLOAT_EQ( u_coarse_data[ IM + n_complex*(row+ num_colorspin*col) ],  (float)(0));
				}
			}

	    for(int col=0; col < num_color; ++col) {

	      for(int row=0; row < num_color; ++row) {

					ASSERT_FLOAT_EQ( u_coarse_data[ RE + n_complex*(row + num_colorspin*(col + num_color)) ], (float)(compareB) );
					ASSERT_FLOAT_EQ( u_coarse_data[ IM + n_complex*(row + num_colorspin*(col + num_color)) ],  (float)(0));

				}
			}

	    for(int col=0; col < num_color; ++col) {

	      for(int row=0; row < num_color; ++row) {

					ASSERT_FLOAT_EQ( u_coarse_data[RE + n_complex*(row+num_color + num_colorspin*col)  ], (float)(compareC) );
					ASSERT_FLOAT_EQ( u_coarse_data[ IM + n_complex*(row+num_color + num_colorspin*col) ],  (float)(0));
				}
			}

	    for(int col=0; col < num_color; ++col) {

	      for(int row=0; row < num_color; ++row) {

					ASSERT_FLOAT_EQ( u_coarse_data[ RE + n_complex*(row+num_color+ num_colorspin*(col+ num_color)) ], (float)(compareD) );
					ASSERT_FLOAT_EQ( u_coarse_data[ IM + n_complex*(row+num_color + num_colorspin*(col + num_color)) ],  (float)(0));
				}
			}

		}
	}

}


#if 0
// Test against pencil and paper
TEST(TestCoarseQDPXXBlock, TestTripleProductT2)
{
	IndexArray latdims={{2,2,2,4}};
	IndexArray blockdims={{1,1,1,2}};
	IndexArray node_orig=NodeInfo().NodeCoords();
	for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];
	initQDPXXLattice(latdims);

	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	QDPIO::cout << "Generating Unit Gauge" << std::endl;

	// Taking Unit Gauge here to remove color from the inner products.
	// Just check spin structure for now.
	multi1d<LatticeColorMatrix> u(Nd);
	for(int mu=0; mu < Nd; ++mu) u[mu] = 1;

	// Now I want to create 6 vectors
	multi1d<LatticeFermion> vecs(6);

	for(int k=0; k < 6; ++k) {
		vecs[k] = zero;
	}

	// Fill out the vector that the spinors for t=0 are  [ (1 0 0), (2 0 0), (3,0,0), (4,0,0) ] on all spatial sites
	//                                          t=1 are  [ (5,0,0), (6,0,0), (7,0,0), (8,0,0) ] on all spatial sites
	//                                          t=2 are  [ (9,0,0), (10,0,0), (11,0,0), (12,0,0) ] on all spatial sites
	//                                          t=3 are  [ (13,0,0), (14,0,0), (15,0,0), (16,0,0) ] on all spatial sites
	//
	// I.e. the number on generic t, spin is 1 + 4*t + spin


	for(int site=0; site < all.numSiteTable(); ++site) {
		// Get the t_coordinate of the site
		multi1d<int> coord = Layout::siteCoords(Layout::nodeNumber(), site);
		for(int spin=0; spin < 4; ++spin) {
			int idx = 1 + 4*coord[3] + spin;
			for(int k=0; k < 6; ++k) {
				vecs[k].elem(all.start() + site).elem(spin).elem(0).real() = idx;
			}
		}
	}

	// Check the vectors
	for(int t=0; t < 2; ++t) {
		multi1d<int> coord(4);
		coord[0]=0; coord[1]=0; coord[2]=0; coord[3]=t;
		int site=Layout::linearSiteIndex(coord);
		QDPIO::cout << "Coord=(0,0,0,"<<t<<") site =" << site <<  "  vec= [ " ;
		for(int spin=0; spin < 4; ++spin) {
			for(int color=0; color < 3; ++color ) {
				QDPIO::cout <<" ( " << vecs[0].elem(site).elem(spin).elem(color).real() << " , " <<
							vecs[0].elem(site).elem(spin).elem(color).imag()  <<" ) ";

			}
		}
		QDPIO::cout << " ] " <<  std::endl;
	}
	// 1) Create the blocklist
	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	CreateBlockList(my_blocks,blocked_lattice_dims,latdims,blockdims, node_orig);

	// Next step should be to copy this into the fields needed for gauge and clover ops
	LatticeInfo info(blocked_lattice_dims, 2, 6, NodeInfo());
	CoarseGauge u_coarse(info);
	ZeroGauge(u_coarse);

	QDPIO::cout << " Attempting Triple Product in  T+ direction (6): "<< std::endl;
	dslashTripleProductDirQDPXX(my_blocks, 6, u, vecs, u_coarse);

	// Now check the gauge links on site 0 (x,y,z,t)=0,0,0,0
	// matrix should be
	//     A   B
	//     C   D
	// where A is a 6x6 matrix  with all elements -6 and B is a matrix with all elements 4
	for(int cb=0; cb < n_checkerboard; ++cb) {
		for(int cbsite=0; cbsite < info.GetNumCBSites(); ++cbsite) {



			// turn cb, cbsite into coordinate:
			IndexArray coords;
			CBIndexToCoords(cbsite,cb, blocked_lattice_dims, coords);

			int compareA = 0;
			int compareB = 0;
			int compareC = 0;
			int compareD = 0;

			int t=coords[3];
			switch(t)  {
			case 0:
			{
				compareA = 122;
				compareB = -150;
				compareC = -182;
				compareD = 226;
			}
			break;

			case 1:
			{
				compareA = 298;
				compareB = -390;
				compareC = -358;
				compareD = 466;
			}
			break;
			default:
				break;
			};
			int num_color = info.GetNumColors();
			int num_colorspin = 2*num_color;

			// chiral row = 0; chiral col = 0;
			float* u_coarse_data = u_coarse.GetSiteDirDataPtr(cb,cbsite,6);

			for(int row=0; row < num_color; ++row) {
				for(int col=0; col < num_color; ++col) {
					ASSERT_FLOAT_EQ( u_coarse_data[ RE + n_complex*(col+ num_colorspin*row) ], (float)(compareA) );
					ASSERT_FLOAT_EQ( u_coarse_data[ IM + n_complex*(col+ num_colorspin*row) ],  (float)(0));
				}
			}


			for(int row=0; row < num_color; ++row) {
				for(int col=0; col < num_color; ++col) {

					ASSERT_FLOAT_EQ( u_coarse_data[ RE + n_complex*(col+ num_colorspin*(row + num_color)) ], (float)(compareC) );
					ASSERT_FLOAT_EQ( u_coarse_data[  IM + n_complex*(col+ num_colorspin*(row + num_color)) ],  (float)(0));

				}
			}

			for(int row=0; row < num_color; ++row) {
				for(int col=0; col < num_color; ++col) {

					ASSERT_FLOAT_EQ( u_coarse_data[RE + n_complex*(col+num_color + num_colorspin*row)  ], (float)(compareB) );
					ASSERT_FLOAT_EQ( u_coarse_data[ IM + n_complex*(col+num_color + num_colorspin*row) ],  (float)(0));
				}
			}

			for(int row=0; row < num_color; ++row) {
				for(int col=0; col < num_color; ++col) {

					ASSERT_FLOAT_EQ( u_coarse_data[ RE + n_complex*(col+num_color+ num_colorspin*(row + num_color)) ], (float)(compareD) );
					ASSERT_FLOAT_EQ( u_coarse_data[ IM + n_complex*(col+num_color + num_colorspin*(row + num_color)) ],  (float)(0));
				}
			}

		}
	}

	{
		QDPIO::cout << "Testing Restriction" << std::endl;
		// Now test restriction
		LatticeFermion pre_R = zero;
		for(int site=0; site < QDP::all.numSiteTable(); ++site) {
			for(int spin=0; spin < 4; ++spin) {
				for(int color=0; color < 3; ++color ) {
					pre_R.elem(site).elem(spin).elem(color).real() = 1;
				}
			}
		}

		CoarseSpinor R_pre_R(info);

		restrictSpinorQDPXXFineToCoarse(my_blocks, vecs, pre_R, R_pre_R);

		// Now block zero comes frome t=0 + t=1 sites
		// Because all the elements of R_pre_R are 1 this just sums the elements in the spins of the vectors
		// which are [ 1 + 2 ]+[ 5+ 6 ] for coarse t=0; chirality=0 = 14
		//           [ 3 + 4 ]+[ 7 + 8 ] for coarse t=0; chirality=1 = 22

		// for t=1 we will have
		//  ch=0   [ 9 + 10  ] + [13 + 14 ] = 46
		//  ch=1   [ 11 + 12 ] + [15 + 16]  = 23 + 31 = 54
		for(int cb=0; cb < n_checkerboard; ++cb) {
			for(int cbsite=0; cbsite < info.GetNumCBSites();++cbsite) {
				float* vec_data = R_pre_R.GetSiteDataPtr(cb,cbsite);
				IndexArray coords;
				CBIndexToCoords(cbsite,cb,blocked_lattice_dims,coords);
				int expected_ch0 = 0;
				int expected_ch1 = 0;
				if ( coords[3] == 0  )  {
					expected_ch0 = 14;
					expected_ch1 = 22;
				}
				if( coords[3] == 1 ) {
					expected_ch0 = 46;
					expected_ch1 = 54;
				}
				for(int cs = 0; cs < R_pre_R.GetNumColorSpin(); ++cs ) {
					if ( cs < R_pre_R.GetNumColor() ) {
						ASSERT_FLOAT_EQ( (float)(expected_ch0), vec_data[RE + n_complex*cs]);
						ASSERT_FLOAT_EQ( (float)(0), vec_data[IM + n_complex*cs]);
					}
					else {
						ASSERT_FLOAT_EQ( (float)(expected_ch1), vec_data[RE + n_complex*cs]);
						ASSERT_FLOAT_EQ( (float)(0), vec_data[IM + n_complex*cs]);
					}
				}

			}
		}

		QDPIO::cout << "RESTRICTION ASSERTIONS PASSED *********" << std::endl;
	}


	// Now test Prolongation
	{
		QDPIO::cout << "Testing Prolongation : " << std::endl;
		CoarseSpinor pre_P(info);
		for(int cb=0; cb < n_checkerboard; ++cb) {
			for(int cbsite=0; cbsite < info.GetNumCBSites(); ++cbsite ) {
				float* vec_data = pre_P.GetSiteDataPtr( cb, cbsite);
				for(int cs=0; cs < pre_P.GetNumColorSpin(); ++cs) {
					vec_data[ RE + n_complex*cs ] = 1;
					vec_data[ IM + n_complex*cs ] = 0;
				}
			}
		}

		LatticeFermion P_pre_P;
		prolongateSpinorCoarseToQDPXXFine(my_blocks, vecs, pre_P, P_pre_P);

		for(int cb=0; cb < n_checkerboard; ++cb) {
			for(int cbsite=0; cbsite < rb[cb].numSiteTable();++cbsite) {
				int idx = cbsite + cb * rb[cb].numSiteTable();

				multi1d<int> qdp_coords = Layout::siteCoords(Layout::nodeNumber(), idx);
				int t = qdp_coords[3];
				int start = 4*t + 1;
				int expected[4] = { 6*start, 6*(start + 1), 6*(start + 2), 6*(start + 3) };

				for(int spin=0; spin < 4; ++spin ) {
					for(int color=0; color < 3; ++color ) {
						float realpart = P_pre_P.elem(idx).elem(spin).elem(color).real();
						float imagpart = P_pre_P.elem(idx).elem(spin).elem(color).imag();
						if( color == 0 ) {
							float expect = (float)(expected[spin]);

							ASSERT_FLOAT_EQ( (float)(expect), realpart );
							ASSERT_FLOAT_EQ( (float)(0), imagpart );
						}
						else {
							// Other colors
							ASSERT_FLOAT_EQ( (float)(0), realpart );
							ASSERT_FLOAT_EQ( (float)(0), imagpart );
						}
					}
				}

			}
		}
		QDPIO::cout << "PROLONGATION Tests passed  ******* " << std::endl;
	}



	// Now test CoarseDslash
	int n_smt = 1;
	CoarseDiracOp D_op_coarse(info, n_smt);
	CoarseSpinor in(info);
	CoarseSpinor out(info);

	ZeroVec(in);
	ZeroVec(out);

	// On each site with a (0,0,0,1) coordinate on the coarse lattice create the vector [1,1,1,1,1 ...,1]
	for(int cb=0; cb < n_checkerboard; ++cb) {
		for(int cbsite=0; cbsite < info.GetNumCBSites(); ++cbsite) {
			IndexArray coords;
			CBIndexToCoords(cbsite,cb, blocked_lattice_dims, coords);
			if( coords[0]==0 && coords[1] == 0 && coords[2] == 0 && coords[3] == 1) {
				float *in_data = in.GetSiteDataPtr(cb,cbsite);

				for(int cs=0; cs < in.GetNumColorSpin(); cs++) {
					in_data[ RE + n_complex*cs ] = 1;
					in_data[ IM + n_complex*cs ] = 0;
				}
			}
		}
	}

#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		D_op_coarse.Dslash(out, u_coarse, in, 0, LINOP_OP, tid);
		D_op_coarse.Dslash(out, u_coarse, in, 1, LINOP_OP, tid);
	}

	// OK only site (0,0,0,0) should be nonzero
	// and its value should be 6*(122-150)=-168 for upper chirality and 6*(-182+226) =264 for the lower chirality
	for(int cb=0; cb < n_checkerboard; ++cb) {
			for(int cbsite=0; cbsite < info.GetNumCBSites(); ++cbsite) {
				IndexArray coords;
				CBIndexToCoords(cbsite,cb, blocked_lattice_dims, coords);

				float *out_data = out.GetSiteDataPtr(cb,cbsite);


				if( coords[0]==0 && coords[1] == 0 && coords[2] == 0 && coords[3] == 0) {
					for(int cs=0; cs < out.GetNumColor(); cs++) {
						ASSERT_FLOAT_EQ( out_data[ RE + n_complex*cs ], (float)(-168)) ;
						ASSERT_FLOAT_EQ( out_data[ IM + n_complex*cs ], (float)(0));
						ASSERT_FLOAT_EQ( out_data[ RE + n_complex*(cs + out.GetNumColor())], (float)(264)) ;
						ASSERT_FLOAT_EQ( out_data[ IM + n_complex*(cs + out.GetNumColor())], (float)(0));
					}
				}
				else {
					for(int cs=0; cs < out.GetNumColorSpin(); cs++) {
						ASSERT_FLOAT_EQ( out_data[ RE + n_complex*cs ], (float)(0)) ;
						ASSERT_FLOAT_EQ( out_data[ IM + n_complex*cs ], (float)(0));
					}
				}
			}
		}


	LatticeFermion P_in, DP_in;
	P_in = zero;
	DP_in= zero;
	CoarseSpinor RDP_out(info);
	ZeroVec(RDP_out);

	QDPIO::cout << "Resetting in " << std::endl;

	for(int cb=0; cb < n_checkerboard; ++cb) {
			for(int cbsite=0; cbsite < info.GetNumCBSites(); ++cbsite ) {
				float* in_data =in.GetSiteDataPtr( cb, cbsite);
				for(int cs=0; cs < in.GetNumColorSpin(); ++cs) {
					in_data[ RE + n_complex*cs ] = 1;
					in_data[ IM + n_complex*cs ] = 0;
				}
			}
		}

	QDPIO::cout << "Prolongating P_in" << std::endl;

	prolongateSpinorCoarseToQDPXXFine(my_blocks, vecs, in, P_in); // NB: This is not the same as v_f, but rather P R v_f

	QDPIO::cout << "Checking Prolongated P_in" << std::endl;

	// Now we should have that P_in has spin components as per test before i.e.
	for(int cb=0; cb < n_checkerboard; ++cb) {
		for(int cbsite=0; cbsite < rb[cb].numSiteTable();++cbsite) {
			int idx = cbsite + cb * rb[cb].numSiteTable();

			multi1d<int> qdp_coords = Layout::siteCoords(Layout::nodeNumber(), idx);
			int t = qdp_coords[3];
			int start = 4*t + 1;
			int expected[4] = { 6*start, 6*(start + 1), 6*(start + 2), 6*(start + 3) };

			for(int spin=0; spin < 4; ++spin ) {
				for(int color=0; color < 3; ++color ) {
					float realpart = P_in.elem(idx).elem(spin).elem(color).real();
					float imagpart = P_in.elem(idx).elem(spin).elem(color).imag();
					if( color == 0 ) {
						float expect = (float)(expected[spin]);

						ASSERT_FLOAT_EQ( (float)(expect), realpart );
						ASSERT_FLOAT_EQ( (float)(0), imagpart );
					}
					else {
						// Other colors
						ASSERT_FLOAT_EQ( (float)(0), realpart );
						ASSERT_FLOAT_EQ( (float)(0), imagpart );
					}
				}
			}

		}
	}

	QDPIO::cout << "Hitting P_in with DslashDir(6)" << std::endl;


	DslashDirQDPXX(DP_in, u, P_in, 6);

	QDPIO::cout << "Checking DP_in" << std::endl;
	// Now we should have that P_in has spin components as per test before i.e.
	for(int cb=0; cb < n_checkerboard; ++cb) {
		for(int cbsite=0; cbsite < rb[cb].numSiteTable();++cbsite) {
			int idx = cbsite + cb * rb[cb].numSiteTable();

			for(int spin=0; spin < 4; ++spin ) {

				for(int color=0; color < 3; ++color ) {
					float realpart = DP_in.elem(idx).elem(spin).elem(color).real();
					float imagpart = DP_in.elem(idx).elem(spin).elem(color).imag();
					if( color == 0 ) {
					//	QDPIO::cout << "cb="<<cb<<" cbsite="<<cbsite<<" spin=" << spin << " color=" << color<< " (" << realpart << " , " << imagpart <<" ) " << std::endl;
						int expect = -12;
						if( spin >= 2 ) {
							expect = 12;
						}

						ASSERT_FLOAT_EQ( (float)(expect), realpart );
						ASSERT_FLOAT_EQ( (float)(0), imagpart );
					}
					else {
						// Other colors
						ASSERT_FLOAT_EQ( (float)(0), realpart );
						ASSERT_FLOAT_EQ( (float)(0), imagpart );
					}
				}
			}



		}
	}
	QDPIO::cout << "DP Checking completed" << std::endl;
	QDPIO::cout << "Restricting back" << std::endl;

	restrictSpinorQDPXXFineToCoarse(my_blocks, vecs, DP_in, RDP_out);
	QDPIO::cout << "Checking Restriction" << std::endl;
	for(int cb=0; cb < n_checkerboard; ++cb) {
		for(int cbsite=0; cbsite < info.GetNumCBSites(); ++cbsite) {
			IndexArray coords;
			CBIndexToCoords(cbsite,cb, blocked_lattice_dims, coords);
			float *out_data = RDP_out.GetSiteDataPtr(cb,cbsite);
			float expected_up = -168;
			float expected_down = 264;

			switch( coords[3] ) {
			case 0:
			{
				expected_up = -168;
				expected_down = 264;
			}
			break;
			case 1:
			{
				expected_up = -552;
				expected_down = 648;
			}
			break;
			default:
				abort();
				break;
			};

			for(int cs=0; cs < out.GetNumColor(); cs++) {
				ASSERT_FLOAT_EQ( out_data[ RE + n_complex*cs ], expected_up) ;
				ASSERT_FLOAT_EQ( out_data[ IM + n_complex*cs ], (float)(0));
				ASSERT_FLOAT_EQ( out_data[ RE + n_complex*(cs + out.GetNumColor())], expected_down) ;
				ASSERT_FLOAT_EQ( out_data[ IM + n_complex*(cs + out.GetNumColor())], (float)(0));

			}
		}
	}
	QDPIO::cout << "Restriction checks PASSED ***" << std::endl;

}
#endif


TEST(TestCoarseQDPXXBlock, TestCheckerboardSiteOrder)
{
	IndexArray latdims={{2,2,2,2}};
	LatticeInfo info(latdims,2,6,NodeInfo());

	initQDPXXLattice(latdims);

	int num_cbsites=info.GetNumCBSites();

	for(int cb=0; cb < n_checkerboard; ++cb ) {
		for(int cbsite=0; cbsite < num_cbsites; ++cbsite) {
			int idx = cbsite + cb*num_cbsites;

			IndexArray my_coords;
			CBIndexToCoords(cbsite,cb,latdims, info.GetLatticeOrigin(), my_coords);

			multi1d<int> qdp_coords(Nd);
			qdp_coords = Layout::siteCoords( Layout::nodeNumber(), idx);

			QDPIO::cout << "cb="<<cb<<" cbsite="<< cbsite << " My_coords=( " << my_coords[0] << "," << my_coords[1]<<","<<my_coords[2] <<"," << my_coords[3] << ")    ";
			QDPIO::cout << " QDP_coords=( "<< qdp_coords[0] << "," << qdp_coords[1]<<","<< qdp_coords[2] <<"," << qdp_coords[3] << ")    "
					<< "  my_idx=" << idx << "  qdp::rb[cb].siteTable()[cbsite]=" << rb[cb].siteTable()[cbsite] << std::endl;
		}
	}
}

TEST(TestCoarseQDPXXBlock, TestRestrictProlong)
{
	IndexArray latdims={{8,8,8,8}};
	IndexArray blockdims={{2,2,2,2}};

	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	IndexArray node_orig=NodeInfo().NodeCoords();
		for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

	multi1d<LatticeColorMatrix> u(Nd);

	QDPIO::cout << "Generating Random Gauge with Gaussian Noise" << std::endl;
	for(int mu=0; mu < 4; ++mu) {
//		gaussian(u[mu]);
//		reunit(u[mu]);
		u[mu]=1;

	}

	// Random Basis vectors
	multi1d<LatticeFermion> vecs(6);
	for(int k=0; k < 6; ++k) {
		gaussian(vecs[k]);
	}

	// 1) Create the blocklist
	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	IndexArray blocked_lattice_orig;
	CreateBlockList(my_blocks,blocked_lattice_dims,blocked_lattice_orig,latdims,blockdims,node_orig);

	// Do the proper block orthogonalize -- I do it twice... Why not
	// This should stay real;
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);

	for(int vec=0; vec < 6; ++vec ) {
		for(int block = 0; block < my_blocks.size(); ++block) {
			for(int chiral=0; chiral < 2; ++chiral ) {
				double nvec = norm2BlockAggrQDPXX(vecs[vec],my_blocks[block],chiral);
				ASSERT_NEAR( nvec, (double)1, 1.0e-12);
			}
		}
	}

	LatticeInfo info(blocked_lattice_orig,blocked_lattice_dims, 2, 6, NodeInfo());
	CoarseSpinor in(info);
	Gaussian(in);
	CoarseSpinor out(info);
	ZeroVec(out);

	LatticeFermion intermediary=zero;
	prolongateSpinorCoarseToQDPXXFine(my_blocks, vecs, in, intermediary);
	restrictSpinorQDPXXFineToCoarse(my_blocks, vecs, intermediary, out);
	for(int ccb=0; ccb < 2; ++ccb) {
		for(int ccbsite=0; ccbsite < info.GetNumCBSites(); ++ccbsite) {
			float *site_data = in.GetSiteDataPtr(0,ccb,ccbsite);
			float *site_data2 = out.GetSiteDataPtr(0,ccb,ccbsite);
			for(int cspin=0; cspin < out.GetNumColorSpin(); ++cspin) {
//				QDPIO::cout << " in[cb="<< ccb<<"][csite=" << ccbsite <<"][cspin=" << cspin <<"]=("
//						<< site_data[RE+cspin*n_complex] << ", " << site_data[IM + cspin*n_complex] << ")       "
//						<< " out[cb="<< ccb<<"][csite=" << ccbsite <<"][cspin=" << cspin <<"]=("
//												<< site_data2[RE+cspin*n_complex] << ", " << site_data2[IM + cspin*n_complex] << ")" << std::endl;

				ASSERT_NEAR( site_data[RE+cspin*n_complex],site_data2[RE+cspin*n_complex], 1.0e-5 );
				ASSERT_NEAR( site_data[IM+cspin*n_complex],site_data2[IM+cspin*n_complex], 1.0e-5 );


			}
			//QDPIO::cout << std::endl;
		}
	}

}

void zeroImagPart(LatticeFermion& ferm)
{
	for(int site=0; site < Layout::sitesOnNode(); ++site ) {
		for(int spin=0; spin < 4; ++spin)  {
			for(int color=0; color < 3; ++color ) {
				if( color == 0 ) {
					ferm.elem(site).elem(spin).elem(color).imag() = 0; // Keeping it real!
				}
				else {
					ferm.elem(site).elem(spin).elem(color).real() = 0; // Keeping it real!
					ferm.elem(site).elem(spin).elem(color).imag() = 0; // Keeping it real!
				}
			}
		}
	}
}

void zeroImagPart(CoarseSpinor& spinor)
{
	const LatticeInfo& info = spinor.GetInfo();
	const int num_cbsites = info.GetNumCBSites();
	for(int cb=0; cb < n_checkerboard;++cb) {
		for(int cbsite=0; cbsite < num_cbsites; ++cbsite) {
			float* s_data = spinor.GetSiteDataPtr(0,cb,cbsite);
			for(int cspin=0; cspin < spinor.GetNumColorSpin(); cspin++) {
				s_data[IM + n_complex*cspin ] = 0;
			}
		}
	}
}
void Fill(CoarseSpinor& spinor, const float re, const float im)
{
	const LatticeInfo& info = spinor.GetInfo();
	const int num_cbsites = info.GetNumCBSites();
	for(int cb=0; cb < n_checkerboard;++cb) {
		for(int cbsite=0; cbsite < num_cbsites; ++cbsite) {
			float* s_data = spinor.GetSiteDataPtr(0,cb,cbsite);
			for(int cspin=0; cspin < spinor.GetNumColorSpin(); cspin++) {
				s_data[RE + n_complex*cspin ] = re - 0.2*cspin;
				s_data[IM + n_complex*cspin ] = im + 0.05*cspin;
			}
		}
	}
}

TEST(TestCoarseQDPXXBlock, TestCoarseDslashNeighbors)
{
	IndexArray qdp_latdims={{4,4,4,4}};
	IndexArray latdims={{2,2,2,4}};

	initQDPXXLattice(latdims);

	LatticeInfo info( latdims, 2, 6, NodeInfo());
	CoarseGauge u(info);

	int num_cbsites = info.GetNumCBSites();
	int num_colors = u.GetNumColor();
	int num_colorspin = u.GetNumColorSpin();

	CoarseSpinor in(info);
	CoarseSpinor out(info);
	int n_smt = 1;
	CoarseDiracOp D_op_coarse(info, n_smt);
	ZeroVec(in);
	for(int cb=0; cb < n_checkerboard; ++cb) {
		for(int cbsites=0; cbsites < num_cbsites; ++cbsites) {
			int qdp_site = rb[cb].siteTable()[cbsites];
			multi1d<int> coords=Layout::siteCoords(Layout::nodeNumber(), qdp_site);
			IndexArray coords2;
			CBIndexToCoords(cbsites,cb,latdims,info.GetLatticeOrigin(),coords2);
			ASSERT_EQ( coords[0], coords2[0] );
			ASSERT_EQ( coords[1], coords2[1] );
			ASSERT_EQ( coords[2], coords2[2] );
			ASSERT_EQ( coords[3], coords2[3] );
			float* v_data = in.GetSiteDataPtr(0,cb,cbsites);
			v_data[0] = (float)coords[0];
			v_data[1] = (float)coords[1];
			v_data[2] = (float)coords[2];
			v_data[3] = (float)coords[3];
			v_data[4] = (float)cb;
			v_data[5] = (float)cbsites;

		}
	}
#if 1

	// Test neighbors in all 8 directions
	for(int mu=0; mu < 8; ++mu ) {
		QDPIO::cout << "Testing Dslash Dir=" << mu << std::endl;
		// Zero Output Vector
		ZeroVec(out);
		ZeroGauge(u);
		for(int cb=0; cb < n_checkerboard;++cb) {
			for(int cbsites=0; cbsites < num_cbsites; ++cbsites) {
				float *u_data = u.GetSiteDirDataPtr(cb,cbsites,mu);
				for(int diag=0; diag < num_colorspin; ++diag ) {
					u_data[ RE + n_complex*(diag + diag*num_colorspin)] = 1;
				}

			} //cbsites
		} // cb

	#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			D_op_coarse.unprecOp(out, u, in, 0, LINOP_OP, tid);
			D_op_coarse.unprecOp(out, u, in, 1, LINOP_OP, tid);
		}

		for(int cb=0; cb < n_checkerboard; ++cb) {
			for(int cbsites=0; cbsites < num_cbsites; ++cbsites) {
				const float* out_data = out.GetSiteDataPtr(0,cb,cbsites);
				IndexArray my_coords={{0,0,0,0}};
				CBIndexToCoords(cbsites,cb, latdims,info.GetLatticeOrigin(),my_coords); // Get My Coords
				IndexArray expected = my_coords;
				int direct=mu/2;
				int addend = (mu % 2 == 0 ) ? +1 : -1; // Even numbers: forward neigh, odd numbers backward neight
				expected[direct] += addend;

				// Wraparound
				if (expected[direct] < 0) expected[direct] = latdims[direct]-1;
				if (expected[direct] >= latdims[direct]) expected[direct]=0;


				std::cout << "cb = " << cb << " cbsite=" << cbsites << " coord=(" << my_coords[0] << ", " << my_coords[1] << ", "
						<< my_coords[2] << ", " << my_coords[3] << ")   mu=" << mu << " dir=" << direct << " add=" << addend
						<< " expected=(" << expected[0] << ", " << expected[1] << ", " << expected[2] << ", " << expected[3] << ")"
						<< "   got=(" << out_data[0] << ", " << out_data[1]
						<< ", " << out_data[2] << ", " << out_data[3]<<")"
						<< "   out_cb="<< out_data[4]<< " out_site=" << out_data[5] <<std::endl;

				float fexpected[4] = { (float)expected[0],
						(float)expected[1],
						(float)expected[2],
						(float)expected[3] };

				ASSERT_NEAR( fexpected[0] , out_data[0], 1.0e-10 );
				ASSERT_NEAR( fexpected[1] , out_data[1], 1.0e-10 );
				ASSERT_NEAR( fexpected[2] , out_data[2], 1.0e-10 );
				ASSERT_NEAR( fexpected[3] , out_data[3], 1.0e-10 );
			}
		}
	} // mu

#endif
}


TEST(TestCoarseQDPXXBlock, TestFakeCoarseDslashUnitGaugeDir6)
{
	IndexArray latdims={{2,2,2,4}};
	IndexArray blockdims={{1,1,1,2}};

	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	IndexArray node_orig=NodeInfo().NodeCoords();
		for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

	multi1d<LatticeColorMatrix> u(Nd);

	QDPIO::cout << "Generating Random Gauge with Gaussian Noise" << std::endl;
	for(int mu=0; mu < Nd; ++mu) {

		u[mu] = 1;

	}


	// Random Basis vectors
	multi1d<LatticeFermion> vecs(6);

	MasterLog(INFO,"Generating Eye\n");
	{
		LatticePropagator eye=1;


		// Pack 'Eye' vectors into to the in_vecs;
		for(int spin=0; spin < Ns/2; ++spin) {
			for(int color =0; color < Nc; ++color) {
				LatticeFermion upper = zero;
				LatticeFermion lower = zero;

				PropToFerm(eye, lower, color, spin);
				PropToFerm(eye, upper, color, spin+Ns/2);
				vecs[color + Nc*spin] = upper + lower;
			}
		}
	}
	// Someone once said doing this twice is good
	QDPIO::cout << "Orthonormalizing Nullvecs" << std::endl;

	// 1) Create the blocklist
	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	IndexArray blocked_lattice_orig;
	CreateBlockList(my_blocks,blocked_lattice_dims,blocked_lattice_orig,latdims,blockdims,node_orig);

	// Do the proper block orthogonalize -- I do it twice... Why not
	// This should stay real;
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);


	// Next step should be to copy this into the fields needed for gauge and clover ops
	LatticeInfo info(blocked_lattice_orig,blocked_lattice_dims, 2, 6, NodeInfo());
	int num_coarse_cbsites = info.GetNumCBSites();

	CoarseGauge u_coarse(info);

		// Just do time
	//for(int mu=6; mu < 8; ++mu) {
	QDPIO::cout << " Attempting Triple Product in direction: " << 6 << std::endl;
	ZeroGauge(u_coarse);
	dslashTripleProductDirQDPXX(my_blocks, 6, u, vecs, u_coarse);

	int n_smt = 1;
	CoarseDiracOp D_op_coarse(info, n_smt);

#if 0
	for(int op=LINOP_OP; op <= LINOP_DAGGER; ++op ) {

		int isign = ( op == LINOP_OP ) ? +1 : -1;
#endif

		int isign = +1;
		int op = LINOP_OP;


	// Coarsen v_f to R(v_f) give us coarse RHS for tests
	CoarseSpinor v_c(info);
	ZeroVec(v_c);
	int num_coarse_colorspin = v_c.GetNumColorSpin();
	for(int coarse_cb=0; coarse_cb < n_checkerboard; ++coarse_cb) {
		for(int coarse_cbsite=0; coarse_cbsite < num_coarse_cbsites;++coarse_cbsite) {
			float *vec_data =v_c.GetSiteDataPtr(0,coarse_cb,coarse_cbsite);
			int idx = coarse_cbsite + coarse_cb*num_coarse_cbsites;
			IndexArray coords;
			CBIndexToCoords(coarse_cbsite,coarse_cb,blocked_lattice_dims,info.GetLatticeOrigin(),coords);
			if( coords[0]==0 && coords[1]==0 && coords[2]==0 && coords[3] == 1) {
				vec_data[0]=1; // Single dirac spike at (0,0,0,1)
			}
		}
	}

	//Fill(v_c, 1, 0);
	// Output
	CoarseSpinor out(info);
	ZeroVec(out);

	CoarseSpinor fake_out(info);
	ZeroVec(fake_out);

	QDPIO::cout << "Applying: out = D_c v_c" << std::endl;
	// Apply Coarse Op Dslash in Threads
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		D_op_coarse.unprecOp(out, u_coarse, v_c, 0, op, tid);
		D_op_coarse.unprecOp(out, u_coarse, v_c, 1, op, tid);
	}


	QDPIO::cout << "Prolongating: P_v_c = P v_c" << std::endl;
	// Now apply the fake operator:
	LatticeFermion P_v_c = zero;
	prolongateSpinorCoarseToQDPXXFine(my_blocks, vecs, v_c, P_v_c); // NB: This is not the same as v_f, but rather P R v_f


	QDPIO::cout << "Applying: D_f P_v_c = D_f P v_c" << std::endl;
	// Now apply the Clover Term to form D_f P
	LatticeFermion D_f_out = zero;

	DslashDirQDPXX(D_f_out, u, P_v_c, 6);

	QDPIO::cout << "Restricting: D_f_out = R D_f P v_c" << std::endl;
	// Now restrict back: fake_out = R D_f P  v_c
	restrictSpinorQDPXXFineToCoarse(my_blocks, vecs, D_f_out, fake_out);

	QDPIO::cout << "Checking: R D_f P v_c == D_c v_c " <<std::endl;
	// We should now compare out, with fake_out. For this we need an xmy


	for(int ccb=0; ccb < 2; ++ccb) {
		for(int ccbsite=0; ccbsite < info.GetNumCBSites(); ++ccbsite) {
			float *fake_data = fake_out.GetSiteDataPtr(0,ccb,ccbsite);
			float *out_data = out.GetSiteDataPtr(0,ccb,ccbsite);
			for(int cspin=0; cspin < out.GetNumColorSpin(); ++cspin) {
				QDPIO::cout << " fake_out[cb="<< ccb<<"][csite=" << ccbsite <<"][cspin=" << cspin <<"]=("
						<< fake_data[RE+cspin*n_complex] << ", " << fake_data[IM + cspin*n_complex] << ")       "
						<< " out[cb="<< ccb<<"][csite=" << ccbsite <<"][cspin=" << cspin <<"]=("
												<< out_data[RE+cspin*n_complex] << ", " << out_data[IM + cspin*n_complex] << ")" << std::endl;
				ASSERT_FLOAT_EQ( fake_data[ RE + cspin*n_complex], out_data[ RE + cspin*n_complex]);
				ASSERT_FLOAT_EQ( fake_data[ IM + cspin*n_complex], out_data[ IM + cspin*n_complex]);

			}
			QDPIO::cout << std::endl;
		}
	}
}

TEST(TestCoarseQDPXXBlock, TestFakeCoarseDslash)
{
	IndexArray latdims={{8,8,8,8}};
	IndexArray blockdims={{2,2,2,2}};

	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	IndexArray node_orig=NodeInfo().NodeCoords();
	for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

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
	IndexArray blocked_lattice_orig;
	CreateBlockList(my_blocks,blocked_lattice_dims,blocked_lattice_orig,latdims,blockdims,node_orig);

	// Do the proper block orthogonalize -- I do it twice... Why not
	// This should stay real;
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);


	// Next step should be to copy this into the fields needed for gauge and clover ops
	LatticeInfo info(blocked_lattice_orig,blocked_lattice_dims, 2, 6, NodeInfo());
	int num_coarse_cbsites = info.GetNumCBSites();
	CoarseGauge u_coarse(info);

	ZeroGauge(u_coarse);
	for(int mu=0; mu < 8; ++mu) {
		QDPIO::cout << " Attempting Triple Product in direction: " << mu << std::endl;
		dslashTripleProductDirQDPXX(my_blocks, mu, u, vecs, u_coarse);
	}

	int n_smt = 1;
	CoarseDiracOp D_op_coarse(info, n_smt);

	for(int op=LINOP_OP; op <= LINOP_DAGGER; ++op ) {

		int isign = ( op == LINOP_OP ) ? +1 : -1;


		// Coarsen v_f to R(v_f) give us coarse RHS for tests
		CoarseSpinor v_c(info);
		Gaussian(v_c);

		//Fill(v_c, 1, 0);
		// Output
		CoarseSpinor out(info);
		ZeroVec(out);

		CoarseSpinor fake_out(info);
		ZeroVec(fake_out);

		QDPIO::cout << "Applying: out = D_c v_c" << std::endl;
		// Apply Coarse Op Dslash in Threads
#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			D_op_coarse.unprecOp(out, u_coarse, v_c, 0, op, tid);
			D_op_coarse.unprecOp(out, u_coarse, v_c, 1, op, tid);
		}


		QDPIO::cout << "Prolongating: P_v_c = P v_c" << std::endl;
		// Now apply the fake operator:
		LatticeFermion P_v_c = zero;
		prolongateSpinorCoarseToQDPXXFine(my_blocks, vecs, v_c, P_v_c); // NB: This is not the same as v_f, but rather P R v_f


		QDPIO::cout << "Applying: D_f P_v_c = D_f P v_c" << std::endl;
		// Now apply the Clover Term to form D_f P
		LatticeFermion D_f_out = zero;

		dslash(D_f_out,u,P_v_c,isign,0);
		dslash(D_f_out,u,P_v_c,isign,1);

		QDPIO::cout << "Restricting: D_f_out = R D_f P v_c" << std::endl;
		// Now restrict back: fake_out = R D_f P  v_c
		restrictSpinorQDPXXFineToCoarse(my_blocks, vecs, D_f_out, fake_out);

		QDPIO::cout << "Checking: R D_f P v_c == D_c v_c " <<std::endl;
		// We should now compare out, with fake_out. For this we need an xmy


		for(int ccb=0; ccb < 2; ++ccb) {
			for(int ccbsite=0; ccbsite < info.GetNumCBSites(); ++ccbsite) {
				float *fake_data = fake_out.GetSiteDataPtr(0,ccb,ccbsite);
				float *out_data = out.GetSiteDataPtr(0,ccb,ccbsite);
				for(int cspin=0; cspin < out.GetNumColorSpin(); ++cspin) {

					ASSERT_NEAR( fake_data[ RE + cspin*n_complex], out_data[ RE + cspin*n_complex], 5.0e-6);
					ASSERT_NEAR( fake_data[ IM + cspin*n_complex], out_data[ IM + cspin*n_complex], 5.0e-6);

				} // cspin
			} // ccbsite
		} // ccb
	} // op
}

int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

