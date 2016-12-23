#include "gtest/gtest.h"
#include "../test_env.h"
#include "../mock_nodeinfo.h"

#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/geometry_utils.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_op.h"
#include "lattice/coarse/coarse_l1_blas.h"


#include "qdpxx_helpers.h"
#include "reunit.h"
#include "transf.h"
#include "clover_fermact_params_w.h"
#include "clover_term_qdp_w.h"

#include "dslashm_w.h"
#include <complex>

// Site stuff
#include "aggregate_qdpxx.h"

// Block Stuff
#include "aggregate_block_qdpxx.h"

using namespace MG;
using namespace MGTesting;
using namespace QDP;






// Test I can create an uninitialized block
TEST(TestBlocking, TestBlockConstruction )
{
	Block b;
	ASSERT_FALSE( b.isCreated()  );
	ASSERT_EQ( b.getNumSites() , 0);
	auto uninitialized_site_list = b.getSiteList();
	ASSERT_EQ( uninitialized_site_list.size() , 0 );
	ASSERT_TRUE( uninitialized_site_list.empty() );
}

// Test I can create a single block which is the whole lattice
TEST(TestBlocking, TestBlockCreateTrivialSingleBlock )
{
	Block b;
	IndexArray block_dims = {{2,2,2,2}};
	IndexArray local_origin = {{0,0,0,0}};
	IndexArray local_lattice_dims = {{2,2,2,2}};

	b.create(local_lattice_dims,local_origin, block_dims);

	ASSERT_TRUE( b.isCreated() );
	ASSERT_EQ( b.getNumSites() , 16 );
	auto block_sites = b.getSiteList();

	// This is a loop in a single block.
	// this is not further checkerboarded at the moment...
	for( IndexType i=0; i < block_sites.size(); ++i) {
		// This is a trivial example expect block_site[i] to map to lattice site[i];
		ASSERT_EQ(i , block_sites[i] );
	}
}

// Test I can create a single subblock
TEST(TestBlocking, TestBlockCreateSingleBlock )
{
	Block b;
	IndexArray block_dims = {{2,2,2,2}};
	IndexArray local_origin = {{0,0,2,2}};
	IndexArray local_lattice_dims = {{4,4,4,4}};

	b.create(local_lattice_dims,local_origin, block_dims);

	ASSERT_TRUE( b.isCreated() );
	ASSERT_EQ( b.getNumSites() , 16 );
	auto block_sites = b.getSiteList();

	// Go through all the blocks
	for( int i=0; i < block_sites.size(); ++i) {

		// Convert block_idx to coordinates
		IndexArray block_coords = {{0,0,0,0}};
		IndexToCoords(i,block_dims,block_coords);

		// Now offset these coords by the local origin, to get coords in local lattice
		for(int mu=0; mu < n_dim;++mu ) block_coords[mu] += local_origin[mu];

		// Convert lattice Coordinates back to a lattice site index
		IndexType lattice_idx = CoordsToIndex(block_coords, local_lattice_dims);

		// the site_index thus computed should be part of the block site list
		ASSERT_EQ( block_sites[i], lattice_idx);
	}
}

TEST(TestBlocking, TestBlockCreateBadOriginDeath )
{

	Block b;
		IndexArray block_dims = {{2,2,2,2}};
		IndexArray local_origin = {{0,0,3,3}};
		IndexArray local_lattice_dims = {{4,4,4,4}};

	EXPECT_EXIT( b.create(local_lattice_dims,local_origin, block_dims) ,
			::testing::KilledBySignal(SIGABRT), "Assertion failed:*");


}

TEST(TestBlocking, TestBlockCreateBadOriginNegativeDeath )
{

	Block b;
		IndexArray block_dims = {{2,2,2,2}};
		IndexArray local_origin = {{0,0,-1,0}};
		IndexArray local_lattice_dims = {{4,4,4,4}};

		EXPECT_EXIT( b.create(local_lattice_dims,local_origin, block_dims) ,
				::testing::KilledBySignal(SIGABRT), "Assertion failed:*");

}

TEST(TestBlocking, TestCreateBlockList)
{
	using BlockList = std::vector<Block>;

	BlockList my_blocks;
	IndexArray local_lattice_dims = {{4,4,4,4}};
	IndexArray block_dims = {{2,2,2,2}};
	IndexArray blocked_lattice_dims;

	CreateBlockList(my_blocks,blocked_lattice_dims,local_lattice_dims,block_dims);

	// Get checkerboarded dims
	IndexArray blocked_lattice_cbdims(blocked_lattice_dims);
	blocked_lattice_cbdims[0]/=2;
	unsigned int num_cb_blocks = my_blocks.size()/n_checkerboard;

	ASSERT_EQ( my_blocks.size(), 16);
	ASSERT_EQ( blocked_lattice_dims[0],2);
	ASSERT_EQ( blocked_lattice_dims[1],2);
	ASSERT_EQ( blocked_lattice_dims[2],2);
	ASSERT_EQ( blocked_lattice_dims[3],2);

	// Check that for all blocks the sites are the same as if I'd just created them
	for(unsigned int block_cb=0; block_cb < n_checkerboard; ++block_cb) {
		for( unsigned int block_cbsite=0; block_cbsite < num_cb_blocks; ++block_cbsite ) {

			// Get an index into the list of blocks
			unsigned int block_idx = block_cbsite + block_cb*num_cb_blocks;

			// Get the coordinate of this block
			IndexArray block_coord;
			IndexToCoords(block_idx, blocked_lattice_dims, block_coord);

			// Convert coordinate to origin;
			IndexArray block_origin(block_coord);
			for(int mu=0; mu < n_dim; ++mu)  block_origin[mu] *= block_dims[mu];

			// Manually make the block to compare
			Block compare; compare.create(local_lattice_dims, block_origin, block_dims);

			// This is what CreateBlockList created
			Block& from_list = my_blocks[block_idx];

			// Now compare stuff
			ASSERT_TRUE( from_list.isCreated() );
			ASSERT_TRUE( compare.isCreated() );


			auto compare_sitelist = compare.getSiteList();
			auto from_list_sitelist = from_list.getSiteList();

#if 0
			std::cout << "Block: "<< block_idx << std::endl;
			std::cout << " \t sites from blocklist[block_idx] = { " <<std::endl;
			for(unsigned int i=0; i < from_list_sitelist.size(); ++i) {
				std::cout << from_list_sitelist[i] << " " ;
			}
			std::cout << "} " << std::endl;
			std::cout << " \t sites from compare sitelist     = { " <<std::endl;
			for(unsigned int i=0; i < from_list_sitelist.size(); ++i) {
				std::cout << from_list_sitelist[i] << " " ;
			}
			std::cout << "} " << std::endl;
#endif
			for(unsigned int i=0; i < from_list_sitelist.size(); ++i) {
				ASSERT_EQ( from_list_sitelist[i], compare_sitelist[i]);
			}
		}
	}
}

TEST(TestBlocking, TestBlockListCBOrdering)
{
	using BlockList = std::vector<Block>;

	BlockList my_blocks;
	IndexArray local_lattice_dims = {{2,2,2,2}};
	initQDPXXLattice(local_lattice_dims);
	IndexArray block_dims = {{1,1,1,1}};
	IndexArray blocked_lattice_dims;

	// Create the blocked list etc
	CreateBlockList(my_blocks,blocked_lattice_dims,local_lattice_dims,block_dims);

	ASSERT_EQ( my_blocks.size(), 16);
	ASSERT_EQ( blocked_lattice_dims[0],2);
	ASSERT_EQ( blocked_lattice_dims[1],2);
	ASSERT_EQ( blocked_lattice_dims[2],2);
	ASSERT_EQ( blocked_lattice_dims[3],2);

	// Get the number of sites in a checkerboard
	IndexType block_cbsize=my_blocks.size()/n_checkerboard;

	// loop through sites
	for(IndexType cb =  0; cb < n_checkerboard; ++cb ) {
		const int* site_table = rb[cb].siteTable().slice();

		for(IndexType cbsite = 0; cbsite < block_cbsize; ++cbsite) {
			// get the 'coarse site'
			int block_idx = cbsite + cb*block_cbsize;

			// get the list of sites in the block. There should be only
			// one site in the block
			auto blocklist = my_blocks[block_idx].getSiteList();
			ASSERT_EQ( blocklist.size(), 1);

#if 0
			QDPIO::cout << "cb=" << cb << " cbsite=" << cbsite << " block_idx=" << block_idx << " ";
			QDPIO::cout << " blocklist_size=" << blocklist.size() << " blocklist[0]=" << blocklist[0]
							<< " rb[cb][cbsite]=" << site_table[cbsite] << std::endl << std::flush;
#endif

			// That site should have the same index as my current blocksite
			// since each coarse site should be the same as the fine site.
			ASSERT_EQ( block_idx, blocklist[0]) ;
		}
	}

}


TEST(TestCoarseQDPXXBlock, TestBlockOrthogonalize)
{
	IndexArray latdims={{4,4,4,4}};   // Fine lattice. Make it 4x4x4x4 so we can block it
	IndexArray blockdims = {{2,2,2,2}};

	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	// 1) Create the blocklist
	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	CreateBlockList(my_blocks,blocked_lattice_dims,latdims,blockdims);

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

TEST(TestCoarseQDPXXBlock,TestOrthonormal2)
{
	IndexArray latdims={{2,2,2,2}};   // Fine lattice. Make it 4x4x4x4 so we can block it
	IndexArray blockdims = {{1,1,1,1}}; // Trivial blocking -- can check against site variant

	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	// 1) Create the blocklist
	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	CreateBlockList(my_blocks,blocked_lattice_dims,latdims,blockdims);

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
		ASSERT_DOUBLE_EQ( toDouble(ndiff), 0);
	}
}

TEST(TestCoarseQDPXXBlock, TestRestrictorTrivial)
{
	IndexArray latdims={{2,2,2,2}};   // Fine lattice. Make it 4x4x4x4 so we can block it
	IndexArray blockdims = {{1,1,1,1}}; // Trivial blocking -- can check against site variant

	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	// 1) Create the blocklist
	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	CreateBlockList(my_blocks,blocked_lattice_dims,latdims,blockdims);

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


    LatticeInfo info(blocked_lattice_dims, 2, 6, NodeInfo());
	CoarseSpinor coarse_block(info);
	CoarseSpinor coarse_site(info);


	LatticeFermion fine_in;

	gaussian(fine_in);

	// Restrict -- this should be just like packing
	restrictSpinorQDPXXFineToCoarse(compare_vecs,fine_in,coarse_site);
	restrictSpinorQDPXXFineToCoarse(my_blocks,vecs,fine_in,coarse_block);


	for(int cb=0; cb < n_checkerboard; ++cb) {
		for(int cbsite=0; cbsite < info.GetNumCBSites(); ++cbsite) {

			float* coarse_site_cursite = coarse_site.GetSiteDataPtr(cb,cbsite);
			float* coarse_block_cursite = coarse_block.GetSiteDataPtr(cb,cbsite);
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

	// 1) Create the blocklist
	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	CreateBlockList(my_blocks,blocked_lattice_dims,latdims,blockdims);

	// 2) Create the test vectors
	multi1d<LatticeFermion> vecs(6);
	for(int vec=0; vec < 6; ++vec) {
			gaussian(vecs[vec]);
	}

	// Do the proper block orthogonalize
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);

    LatticeInfo info(blocked_lattice_dims, 2, 6, NodeInfo());
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
		CreateBlockList(my_blocks,blocked_lattice_dims,latdims,blockdims);

		// Do the proper block orthogonalize
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


	LatticeFermion psi, d_psi, m_psi;

	gaussian(psi);

	m_psi = zero;


	// Fine version:  m_psi_f =  D_f  psi_f
	// Apply Dslash to both CBs, isign=1
	// Result in m_psiu
	for(int cb=0; cb < 2; ++cb) {
		dslash(m_psi, u, psi, 1, cb);
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
		D_op_coarse.Dslash(coarse_s_out, u_coarse, coarse_s_in, 0, tid);
		D_op_coarse.Dslash(coarse_s_out, u_coarse, coarse_s_in, 1, tid);
	}

	// Export Coa            rse spinor to QDP++ spinors.
	LatticeFermion coarse_d_psi = zero;

	// Prolongate to form coarse_d_psi = P D_c R psi_f
	prolongateSpinorCoarseToQDPXXFine(my_blocks,vecs, coarse_s_out, coarse_d_psi);

	// Check   D_f psi_f = P D_c R psi_f
	LatticeFermion diff = m_psi - coarse_d_psi;

	QDPIO::cout << "Norm Diff[0] = " << sqrt(norm2(diff, rb[0])) << std::endl;
	QDPIO::cout << "Norm Diff[1] = " << sqrt(norm2(diff, rb[1])) 	<< std::endl;
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
}

TEST(TestCoarseQDPXXBlock, TestCoarseQDPXXClovTrivial)
{
	IndexArray latdims={{2,2,2,2}};
	IndexArray blockdims={{1,1,1,1}};

	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

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
	CreateBlockList(my_blocks,blocked_lattice_dims,latdims,blockdims);

	// Do the proper block orthogonalize
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);

	QDPIO::cout << "Coarsening Clover" << std::endl;

	LatticeInfo info(blocked_lattice_dims, 2, 6, NodeInfo());
	CoarseClover c_clov(info);
	clovTripleProductQDPXX(my_blocks, clov_qdp, vecs, c_clov);

	// Now create a LatticeFermion and apply both the QDP++ and the Coarse Clover
	LatticeFermion orig;
	gaussian(orig);
	LatticeFermion orig_res=zero;

	// Apply QDP++ clover
	for(int cb=0; cb < 2; ++cb) {
		clov_qdp.apply(orig_res, orig, 0, cb);
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

		D.CloverApply(s_out, c_clov, s_in,0,tid);
		D.CloverApply(s_out, c_clov, s_in,1,tid);
	}

	LatticeFermion coarse_res;
	prolongateSpinorCoarseToQDPXXFine(my_blocks, vecs, s_out, coarse_res);


	LatticeFermion diff = orig_res - coarse_res;


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
	CreateBlockList(my_blocks,blocked_lattice_dims,latdims,blockdims);

	// Do the proper block orthogonalize
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);

	QDPIO::cout << "Coarsening Clover with Triple Product to create D_c" << std::endl;
	LatticeInfo info(blocked_lattice_dims, 2, 6, NodeInfo());
	CoarseClover c_clov(info);
	clovTripleProductQDPXX(my_blocks, clov_qdp, vecs, c_clov);



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

		D.CloverApply(out, c_clov, v_c,0,tid);
		D.CloverApply(out, c_clov, v_c,1,tid);
	}

	// Now apply the fake operator:
	LatticeFermion P_v_c = zero;
	QDPIO::cout << "Prolongating; P v_c " << std::endl;
	prolongateSpinorCoarseToQDPXXFine(my_blocks, vecs, v_c, P_v_c); // NB: This is not the same as v_f, but rather P R v_f

	QDPIO::cout << "Applying: Clov_f P v_c" << std::endl;
	// Now apply the Clover Term to form D_f P
	LatticeFermion D_f_out = zero;
	for(int cb=0; cb < n_checkerboard; ++cb) {
		clov_qdp.apply(D_f_out, P_v_c, 0, cb);
	}

	// Now restrict back:
	QDPIO::cout << "Restricting: out = R Clov_f P v_c" << std::endl;
	restrictSpinorQDPXXFineToCoarse(my_blocks, vecs, D_f_out, fake_out);

	QDPIO::cout << "Checking Clov_c v_c = R Clov_f P v_c. " << std::endl;
	// We should now compare out, with fake_out. For this we need an xmy
	double norm_diff = sqrt(xmyNorm2Coarse(fake_out,out));
	double norm_diff_per_site = norm_diff / (double)fake_out.GetInfo().GetNumSites();

	MasterLog(INFO, "Diff Norm = %16.8e", norm_diff);
	ASSERT_NEAR( norm_diff, 0, 2.e-5 );
	MasterLog(INFO, "Diff Norm per site = %16.8e", norm_diff_per_site);
	ASSERT_NEAR( norm_diff_per_site,0,1.e-6);

}


TEST(TestCoarseQDPXXBlock, TestFakeCoarseDslash)
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

	// Now create a LatticeFermion and apply both the QDP++ and the Coarse Clover
	LatticeFermion v_f;
	gaussian(v_f);

	// Coarsen v_f to R(v_f) give us coarse RHS for tests
	CoarseSpinor v_c(info);
	QDPIO::cout << "Restricting v_f -> v_c over blocks" << std::endl;
	restrictSpinorQDPXXFineToCoarse(my_blocks, vecs, v_f, v_c);

	// Output
	CoarseSpinor out(info);
	CoarseSpinor fake_out(info);

	QDPIO::cout << "Applying: out = D_c v_c" << std::endl;
	// Apply Coarse Op Dslash in Threads
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		D_op_coarse.Dslash(out, u_coarse, v_c, 0, tid);
		D_op_coarse.Dslash(out, u_coarse, v_c, 1, tid);
	}


	QDPIO::cout << "Prolongating: P_v_c = P v_c" << std::endl;
	// Now apply the fake operator:
	LatticeFermion P_v_c = zero;
	prolongateSpinorCoarseToQDPXXFine(my_blocks, vecs, v_c, P_v_c); // NB: This is not the same as v_f, but rather P R v_f


	QDPIO::cout << "Applying: D_f P_v_c = D_f P v_c" << std::endl;
	// Now apply the Clover Term to form D_f P
	LatticeFermion D_f_out = zero;

	// Apply Dslash to both CBs, isign=1
	// Result in m_psiu
	for(int cb=0; cb < n_checkerboard; ++cb) {
		dslash(D_f_out, u, P_v_c, 1, cb);
	}


	QDPIO::cout << "Restricting: D_f_out = R D_f P v_c" << std::endl;
	// Now restrict back: fake_out = R D_f P  v_c
	restrictSpinorQDPXXFineToCoarse(my_blocks, vecs, D_f_out, fake_out);

	QDPIO::cout << "Checking: R D_f P v_c == D_c v_c " <<std::endl;
	// We should now compare out, with fake_out. For this we need an xmy
	double norm_diff = sqrt(xmyNorm2Coarse(fake_out,out));
	double norm_diff_per_site = norm_diff / (double)fake_out.GetInfo().GetNumSites();

	MasterLog(INFO, "Diff Norm = %16.8e", norm_diff);
	ASSERT_NEAR( norm_diff, 0, 1.e-5 );
	MasterLog(INFO, "Diff Norm per site = %16.8e", norm_diff_per_site);
	ASSERT_NEAR( norm_diff_per_site,0,1.e-6);

}


int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

