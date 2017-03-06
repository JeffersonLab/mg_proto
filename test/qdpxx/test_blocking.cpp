#include "gtest/gtest.h"
#include "../test_env.h"
#include "../mock_nodeinfo.h"

#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/geometry_utils.h"

#include "lattice/coarse/coarse_op.h"
#include "lattice/coarse/coarse_l1_blas.h"
#include "lattice/coarse/block.h"

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


#include <iostream>



// Test I can create an uninitialized block
TEST(TestBlocking, TestBlockConstruction )
{
	Block b;
	ASSERT_FALSE( b.isCreated()  );
	ASSERT_EQ( b.getNumSites() , 0);
	auto uninitialized_site_list = b.getCBSiteList();
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
	IndexArray node_orig=NodeInfo().NodeCoords();
	for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=local_lattice_dims[mu];
	b.create(local_lattice_dims,local_origin, block_dims, node_orig);


	ASSERT_TRUE( b.isCreated() );
	ASSERT_EQ( b.getNumSites() , 16 );
	auto block_sites = b.getCBSiteList();

	int num_cb_sites = block_sites.size() / n_checkerboard;

#if 0
	// This test is not valid, since  the site list is not in CB order for now
	for( int cb=0; cb < n_checkerboard; ++cb ) {
		for( int i=0; i < num_cb_sites; ++i) {
			int block_idx = i + cb*num_cb_sites;
			ASSERT_EQ(cb, block_sites[block_idx].cb) ;
			ASSERT_EQ(i , block_sites[block_idx].site );
		}
	}
#endif
}

// Test I can create a single subblock
TEST(TestBlocking, TestBlockCreateSingleBlock )
{
	Block b;
	IndexArray block_dims = {{2,2,2,2}};
	IndexArray local_block_origin = {{0,0,2,2}};
	IndexArray local_lattice_origin = {{0,0,0,0}};
	IndexArray local_lattice_dims = {{4,4,4,4}};

	initQDPXXLattice(local_lattice_dims); // Give me access to site tables

	IndexArray node_orig=NodeInfo().NodeCoords();
	for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=local_lattice_dims[mu];

	b.create(local_lattice_dims,local_block_origin, block_dims,node_orig);

	ASSERT_TRUE( b.isCreated() );
	ASSERT_EQ( b.getNumSites() , 16 );


	auto block_cbsites = b.getCBSiteList();
	int num_block_sites=block_cbsites.size();

	// Go through all the sites in the block
	// These are not in a checkeboarded order.
	for( unsigned int i=0; i < num_block_sites; ++i) {

		// First convert index i, to linear coordinates
		IndexArray i_coords;

		// i_coords holds block_coords
		IndexToCoords(i,block_dims, i_coords);

		// Convert block to lattice coords by offsetting origin
		for(int mu=0;  mu < n_dim; ++ mu ) i_coords[mu] += local_block_origin[mu];

		// Convert coords to cb and site
		CBSite i_site;
		CoordsToCBIndex(i_coords, local_lattice_dims, local_lattice_origin, i_site.cb, i_site.site);

		// Should match the one from the block cbsites.

		ASSERT_EQ( i_site.cb, block_cbsites[i].cb);
		ASSERT_EQ( i_site.site, block_cbsites[i].site);


	}
}

TEST(TestBlocking, TestBlockNoInteriorTrivial)
{
	Block b;
	IndexArray block_dims={{1,1,1,1}};
	IndexArray local_origin={{0,0,0,0}};
	IndexArray local_lattice_dims={{2,2,2,2}};

	b.create(local_lattice_dims,local_origin, block_dims, {{0,0,0,0}});

	auto inner_sites=b.getInnerBodySiteList();
	ASSERT_EQ( inner_sites.size(), 0);
	for(int mu=0; mu < 8;++mu) {
		auto face_sites_this_dir = b.getFaceList(mu);
		ASSERT_EQ( face_sites_this_dir.size(),1);
		ASSERT_EQ(face_sites_this_dir[0].cb, 0);
		ASSERT_EQ(face_sites_this_dir[0].site, 0 );

	}
}

TEST(TestBlocking, TestBlockNoInterior2222)
{
	Block b;
	IndexArray block_dims={{2,2,2,2}};
	IndexArray local_origin={{0,0,0,0}};
	IndexArray local_lattice_dims={{2,2,2,2}};

	b.create(local_lattice_dims,local_origin, block_dims, {{0,0,0,0}});

	auto inner_sites=b.getInnerBodySiteList();
	ASSERT_EQ( inner_sites.size(), 0);

	for(int mu=0; mu < 8;++mu) {
		auto face_sites_this_dir = b.getFaceList(mu);
		ASSERT_EQ( face_sites_this_dir.size(),8);

		int expect; // this is the value we will expet coord[dir] to have;

		// Index into the cbsite coord arrays ie if mu=0,1 that is x_dir so dir = mu/2 = 0;
		int dir = mu/2;

		// if mu is even then forward face so expect dimsize-1
		if ( mu % 2 == 0 ) {
			// FORWARD
			expect = block_dims[dir]-1;
		}
		else {
			// BACKWARD
			expect = 0;
		}
		MasterLog(INFO, "Mu=%d dir=%d expect=%d \n", mu,dir,expect);
		for(int site=0; site < face_sites_this_dir.size(); ++site) {
			auto site_coords = face_sites_this_dir[site].coords;
			ASSERT_EQ( site_coords[dir], expect);
		}
	}
}

TEST(TestBlocking, TestBlockInterior4444)
{
	Block b;
	IndexArray block_dims={{4,4,4,4}};
	IndexArray local_origin={{0,0,0,0}};
	IndexArray local_lattice_dims={{4,4,4,4}};

	b.create(local_lattice_dims,local_origin, block_dims, {{0,0,0,0}});

	auto inner_sites=b.getInnerBodySiteList();
	ASSERT_EQ( inner_sites.size(), 16); // 2x2x2x2 interior

	for(int mu=0; mu < 8;++mu) {
		auto face_sites_this_dir = b.getFaceList(mu);

		ASSERT_EQ( face_sites_this_dir.size(),64); // 4x4x4 surface

		int expect; // this is the value we will expet coord[dir] to have;

		// Index into the cbsite coord arrays ie if mu=0,1 that is x_dir so dir = mu/2 = 0;
		int dir = mu/2;

		// if mu is even then forward face so expect dimsize-1
		if ( mu % 2 == 0 ) {
			// FORWARD
			expect = block_dims[dir]-1;
		}
		else {
			// BACKWARD
			expect = 0;
		}
		MasterLog(INFO, "Mu=%d dir=%d expect=%d \n", mu,dir,expect);
		for(int site=0; site < face_sites_this_dir.size(); ++site) {
			auto site_coords = face_sites_this_dir[site].coords;
			ASSERT_EQ( site_coords[dir], expect);
		}


		auto not_face_sites_this_dir = b.getNotFaceList(mu);
		int total_sites_in_block = b.getNumSites();
		int sum_sites = face_sites_this_dir.size() + not_face_sites_this_dir.size();
		ASSERT_EQ( total_sites_in_block, sum_sites );

		std::vector<bool> site_found(total_sites_in_block);
		for(int site=0; site < total_sites_in_block; ++site) {
			site_found[site]=false;
		}

		// Process sites in the face
		for(int site=0; site < face_sites_this_dir.size(); ++site) {
			int fine_cb = face_sites_this_dir[site].cb;
			int fine_cbsite = face_sites_this_dir[site].site;
			int test_site = rb[ fine_cb ].siteTable()[ fine_cbsite ];

			ASSERT_FALSE( site_found[ test_site ] ); // make sure this is not yet found
			site_found[test_site] = true;
		}

		for(int site=0; site < not_face_sites_this_dir.size(); ++site ) {
			int fine_cb = not_face_sites_this_dir[site].cb;
			int fine_cbsite = not_face_sites_this_dir[site].site;
			int test_site = rb[ fine_cb ].siteTable()[ fine_cbsite ];
			ASSERT_FALSE( site_found[ test_site ]);
			site_found[test_site ] = true;
		}

		// Now we processed all the face and not face sites
		// All sites should be processed.
		for(int site=0; site < total_sites_in_block; ++site ) {
			ASSERT_TRUE(site_found[site]);
		}

	}
}

TEST(TestBlocking, TestBlockCreateBadOriginDeath )
{

	Block b;
		IndexArray block_dims = {{2,2,2,2}};
		IndexArray local_origin = {{0,0,3,3}};
		IndexArray local_lattice_dims = {{4,4,4,4}};
		IndexArray local_latt_origin = {{0,0,0,0}};
	EXPECT_EXIT( b.create(local_lattice_dims,local_origin, block_dims,local_latt_origin) ,
			::testing::KilledBySignal(SIGABRT), "Assertion failed:*");


}

TEST(TestBlocking, TestBlockCreateBadOriginNegativeDeath )
{

	Block b;
		IndexArray block_dims = {{2,2,2,2}};
		IndexArray local_origin = {{0,0,-1,0}};
		IndexArray local_lattice_dims = {{4,4,4,4}};
		IndexArray local_latt_origin = {{0,0,0,0}};
		EXPECT_EXIT( b.create(local_lattice_dims,local_origin, block_dims,local_latt_origin) ,
				::testing::KilledBySignal(SIGABRT), "Assertion failed:*");

}

TEST(TestBlocking, TestCreateBlockList)
{
	using BlockList = std::vector<Block>;

	BlockList my_blocks;
	IndexArray local_lattice_dims = {{4,4,4,4}};
	IndexArray block_dims = {{2,2,2,2}};
	IndexArray blocked_lattice_dims;
	IndexArray blocked_lattice_origin;
	IndexArray node_orig=NodeInfo().NodeCoords();
	for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=local_lattice_dims[mu];

	// Create a blocklist.
	CreateBlockList(my_blocks,blocked_lattice_dims,blocked_lattice_origin,local_lattice_dims,block_dims,node_orig);
	unsigned int num_cb_blocks = my_blocks.size()/n_checkerboard;

	ASSERT_EQ( my_blocks.size(), 16);
	ASSERT_EQ( blocked_lattice_dims[0],2);
	ASSERT_EQ( blocked_lattice_dims[1],2);
	ASSERT_EQ( blocked_lattice_dims[2],2);
	ASSERT_EQ( blocked_lattice_dims[3],2);

	// Check that for all blocks the sites are the same as if I'd just created them

	// The blocks themselves are checkerboarded in order,

	for(unsigned int block_cb=0; block_cb < n_checkerboard; ++block_cb) {
		for( unsigned int block_cbsite=0; block_cbsite < num_cb_blocks; ++block_cbsite ) {

			// Get an index into the list of blocks
			unsigned int block_idx = block_cbsite + block_cb*num_cb_blocks;

			// Now we want to turn the block_cbsite, block_cb into a coordinate
			IndexArray block_coords;
			CBIndexToCoords(block_cbsite, block_cb, blocked_lattice_dims,node_orig,block_coords);

			// Convert coordinate to a block origin
			IndexArray block_origin(block_coords);
			for(int mu=0; mu < n_dim; ++mu)  block_origin[mu] *= block_dims[mu];

			// Manually make the block to compare
			Block compare; compare.create(local_lattice_dims, block_origin, block_dims, node_orig);

			// This is what CreateBlockList created
			Block& from_list = my_blocks[block_idx];

			// Now compare stuff
			ASSERT_TRUE( from_list.isCreated() );
			ASSERT_TRUE( compare.isCreated() );


			auto compare_sitelist = compare.getCBSiteList();
			auto from_list_sitelist = from_list.getCBSiteList();

#if 1
			std::cout << "Block: "<< block_idx << std::endl;
			std::cout << " \t sites from blocklist[block_idx] = { " <<std::endl;
			for(unsigned int i=0; i < from_list_sitelist.size(); ++i) {
				std::cout << "( " << from_list_sitelist[i].cb << " , " << from_list_sitelist[i].site << ") " ;
			}
			std::cout << "} " << std::endl;
			std::cout << " \t sites from compare sitelist     = { " <<std::endl;
			for(unsigned int i=0; i < from_list_sitelist.size(); ++i) {
				std::cout <<"( " << compare_sitelist[i].cb << " , " << from_list_sitelist[i].site << ") " ;
			}
			std::cout << "} " << std::endl;
#endif
			for(unsigned int i=0; i < from_list_sitelist.size(); ++i) {
				ASSERT_EQ( from_list_sitelist[i].cb, compare_sitelist[i].cb);
				ASSERT_EQ( from_list_sitelist[i].site, compare_sitelist[i].site );
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
	IndexArray blocked_lattice_origin;
	IndexArray node_orig=NodeInfo().NodeCoords();
	for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=local_lattice_dims[mu];

	// Create the blocked list
	// Notably the blocked list should be the same as the
	// unblocked lattice because the blocking is trivial

	CreateBlockList(my_blocks,blocked_lattice_dims,blocked_lattice_origin,
				local_lattice_dims,block_dims,node_orig);

	ASSERT_EQ( my_blocks.size(), 16);
	ASSERT_EQ( blocked_lattice_dims[0],2);
	ASSERT_EQ( blocked_lattice_dims[1],2);
	ASSERT_EQ( blocked_lattice_dims[2],2);
	ASSERT_EQ( blocked_lattice_dims[3],2);

	IndexType num_cbsites=8; // We know this to be true :)

	// loop through coarse cbs (which is the same as the fine cbs)
	for(IndexType cb =  0; cb < n_checkerboard; ++cb ) {

		// Go through the cbsites
		for(IndexType cbsite = 0; cbsite < num_cbsites; ++cbsite) {

			// get the 'coarse site'
			int block_idx = cbsite + cb*num_cbsites;

			// get the list of fine sites in the block. There should be only
			// one fine site in the block

			auto block_cbsitelist = my_blocks[block_idx].getCBSiteList();
			ASSERT_EQ( block_cbsitelist.size(), 1);

			ASSERT_EQ( cb, block_cbsitelist[0].cb);
			ASSERT_EQ( cbsite, block_cbsitelist[0].site);
		}
	}

}





int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

