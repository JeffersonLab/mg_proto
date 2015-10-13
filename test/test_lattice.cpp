#include "gtest/gtest.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/nodeinfo.h"
#include "lattice/indexers.h"
#include "utils/print_utils.h"
#include "lattice/coarsen.h"

#include <vector>

#include "test_env.h"
#include "mock_nodeinfo.h"

using namespace MGGeometry; 

TEST(TestLattice, TestLatticeInitialization)
{
  std::vector<unsigned int> latdims = {16,16,16,16};
  NodeInfo node;
  LatticeInfo lat(latdims, 4, 3,node);
}

TEST(TestLattice, TestLatticeInitializationConvenience)
{
	LatticeInfo  lattinfo({16,16,16,16}, 4, 3, NodeInfo());
}

TEST(TestLattice, TestLatticeCB)
{
  /* Even number of sites */
  std::vector<unsigned int> latdims = {16,16,16,16};
  NodeInfo node;
  LatticeInfo lat(latdims, 4, 3,node);

  // Even Number of Sites
  // Check that both CB-s have the same number of sites
  ASSERT_EQ( lat.GetNumCBSites(0),lat.GetNumCBSites(1));

  // Check that both CB's have half the number of sites
  ASSERT_EQ( lat.GetNumCBSites(0), lat.GetNumSites()/2 );
  ASSERT_EQ( lat.GetNumCBSites(1), lat.GetNumSites()/2 );

  // Check that the number of CB sites matches the
  // Size of the site table for that CB
  for(int cb=0; cb < 2;++cb) {
	  ASSERT_EQ( lat.GetNumCBSites(cb), lat.GetCBSiteTable(cb).size() );
  }
}

TEST(TestLattice, TestLatticeCooky)
{
  /* Even number of sites */
  std::vector<unsigned int> latdims = {3,1,1,1};
  NodeInfo node;
  LatticeInfo lat(latdims, 4, 3,node);
  ASSERT_EQ( lat.GetNumCBSites(0),(unsigned int)2);

  ASSERT_EQ( lat.GetNumCBSites(1), (unsigned int)1 );

  for(int cb=0; cb < 2;++cb) {
	  ASSERT_EQ( lat.GetNumCBSites(cb), lat.GetCBSiteTable(cb).size() );
  }
}

TEST(TestLattice, TestLatticeCookyCB)
{
  /* Even number of sites */
  std::vector<unsigned int> latdims = {3,1,1,1};
  std::vector<unsigned int> pe_dims = {4,4,4,4};
  std::vector<unsigned int> pe_coords = {1,0,0,0};

  MockNodeInfo node(pe_dims,pe_coords);
  LatticeInfo lat(latdims, 4, 3,node);
  ASSERT_EQ( lat.GetNumCBSites(0),(unsigned int)1);
  ASSERT_EQ( lat.GetNumCBSites(1), (unsigned int)2);

  for(int cb=0; cb < 2;++cb) {
	  ASSERT_EQ( lat.GetNumCBSites(cb), lat.GetCBSiteTable(cb).size() );
  }
}

TEST(TestLattice, TestSiteTablesSorted)
{
	std::vector<unsigned int> latdims = {8,8,8,8};
	NodeInfo node;
	LatticeInfo lat(latdims, 4,3, node);

	for(int cb=0; cb < 2; ++cb) {
		const std::vector<unsigned int>& site_table = lat.GetCBSiteTable(cb);
		unsigned int last = site_table[0]; // First element
		for( unsigned int i=1; i < site_table.size(); ++i) {
			// Next element has to be strictly greater than last
			// (strictly gt, since these are site indices, so equal would mean
			// repetition which would be an error )
			ASSERT_GT(last, site_table[i]);

			last = site_table[i];
		}
	}
}


TEST(TestLattice, TestSurfacesBasic)
{
	std::vector<unsigned int> latdims = {4,4,4,4};
	NodeInfo node;
	LatticeInfo lat(latdims, 4, 3, node);

	for(unsigned int mu=0; mu < n_dim; ++mu) {
		for(unsigned int fb = BACKWARD; fb <= FORWARD; ++fb) {

			unsigned int surface_sites = lat.GetNumCBSurfaceSites(mu, fb, 0)
					+ lat.GetNumCBSurfaceSites(mu,fb,1);
			ASSERT_EQ( surface_sites, static_cast<unsigned int>(4*4*4) );


		}
	}
}

TEST(TestLattice, TestSurfaceSiteTablesSorted)
{
	std::vector<unsigned int> latdims = {4,4,4,4};
	NodeInfo node;
	LatticeInfo lat(latdims, 4, 3, node);

	for(unsigned int mu=X_DIR; mu <= T_DIR; ++mu) {
		for(unsigned int fb = static_cast<unsigned int>(BACKWARD);
					fb < static_cast<unsigned int>(FORWARD); ++fb) {
			unsigned int fb_dir = static_cast<unsigned int>(fb);

			unsigned int surface_sites = lat.GetNumCBSurfaceSites(mu, fb_dir, 0)
					+ lat.GetNumCBSurfaceSites(mu,fb_dir,1);

			ASSERT_EQ( surface_sites, static_cast<unsigned int>(4*4*4) );


		}
	}

}
TEST(TestLattice, TestLatticeCookySurface)
{
  /* Even number of sites */
  std::vector<unsigned int> latdims = {3,1,1,1};
  NodeInfo node;
  LatticeInfo lat(latdims, 4, 3,node);

  ASSERT_EQ( lat.GetNumCBSurfaceSites(X_DIR,BACKWARD,0), static_cast<unsigned int>(1));
  ASSERT_EQ( lat.GetNumCBSurfaceSites(X_DIR,FORWARD,0), static_cast<unsigned int>(1));

  ASSERT_EQ( lat.GetNumCBSurfaceSites(X_DIR,BACKWARD,1), static_cast<unsigned int>(0));
  ASSERT_EQ( lat.GetNumCBSurfaceSites(X_DIR,FORWARD,1), static_cast<unsigned int>(0));

  for(unsigned int mu=Y_DIR; mu <= T_DIR; ++mu) {
	  ASSERT_EQ( lat.GetNumCBSurfaceSites(mu,BACKWARD,0), static_cast<unsigned int>(2));
	  ASSERT_EQ( lat.GetNumCBSurfaceSites(mu,FORWARD,0), static_cast<unsigned int>(2));
	  ASSERT_EQ( lat.GetNumCBSurfaceSites(mu,BACKWARD,1), static_cast<unsigned int>(1));
	  ASSERT_EQ( lat.GetNumCBSurfaceSites(mu,FORWARD,1), static_cast<unsigned int>(1));
  }

  /* Even number of sites */
   std::vector<unsigned int> pe_dims = {4,4,4,4};
   std::vector<unsigned int> pe_coords = {1,0,0,0};  /* Odd checkerboarded node */

   MockNodeInfo node_cb1(pe_dims,pe_coords);
   LatticeInfo lat_cb1(latdims, 4, 3,node_cb1);

   /* Everything opposite of before ie swap expectations for cb=0 and cb=1 */
   ASSERT_EQ( lat_cb1.GetNumCBSurfaceSites(X_DIR,BACKWARD,0), static_cast<unsigned int>(0));
   ASSERT_EQ( lat_cb1.GetNumCBSurfaceSites(X_DIR,FORWARD,0), static_cast<unsigned int>(0));

   ASSERT_EQ( lat_cb1.GetNumCBSurfaceSites(X_DIR,BACKWARD,1), static_cast<unsigned int>(1));
   ASSERT_EQ( lat_cb1.GetNumCBSurfaceSites(X_DIR,FORWARD,1), static_cast<unsigned int>(1));

   for(unsigned int mu=Y_DIR; mu <= T_DIR; ++mu) {
	   ASSERT_EQ( lat_cb1.GetNumCBSurfaceSites(mu,BACKWARD,0), static_cast<unsigned int>(1));
	   ASSERT_EQ( lat_cb1.GetNumCBSurfaceSites(mu,FORWARD,0), static_cast<unsigned int>(1));
	   ASSERT_EQ( lat_cb1.GetNumCBSurfaceSites(mu,BACKWARD,1), static_cast<unsigned int>(2));
	   ASSERT_EQ( lat_cb1.GetNumCBSurfaceSites(mu,FORWARD,1), static_cast<unsigned int>(2));
   }

}

TEST(TestLattice, TestLatticeCookySurface2)
{
  /* Even number of sites */
  std::vector<unsigned int> latdims = {1,1,1,1};
  NodeInfo node;
  LatticeInfo lat(latdims, 4, 3,node);

  for(auto mu=X_DIR; mu <= T_DIR; ++mu) {
	  ASSERT_EQ( lat.GetNumCBSurfaceSites(mu,BACKWARD,0), static_cast<unsigned int>(1));
	  ASSERT_EQ( lat.GetNumCBSurfaceSites(mu,FORWARD,0), static_cast<unsigned int>(1));

	  ASSERT_EQ( lat.GetNumCBSurfaceSites(mu,BACKWARD,1), static_cast<unsigned int>(0));
	  ASSERT_EQ( lat.GetNumCBSurfaceSites(mu,FORWARD,1), static_cast<unsigned int>(0));
  }

  /* Even number of sites */
   std::vector<unsigned int> pe_dims = {4,4,4,4};
   std::vector<unsigned int> pe_coords = {1,0,0,0};  /* Odd checkerboarded node */

   MockNodeInfo node_cb1(pe_dims,pe_coords);
   LatticeInfo lat_cb1(latdims, 4, 3,node_cb1);

   /* Everything opposite of before ie swap expectations for cb=0 and cb=1 */
   for(auto mu=X_DIR; mu <= T_DIR; ++mu) {
	   ASSERT_EQ( lat_cb1.GetNumCBSurfaceSites(mu,BACKWARD,0), static_cast<unsigned int>(0));
	   ASSERT_EQ( lat_cb1.GetNumCBSurfaceSites(mu,FORWARD,0), static_cast<unsigned int>(0));
	   ASSERT_EQ( lat_cb1.GetNumCBSurfaceSites(mu,BACKWARD,1), static_cast<unsigned int>(1));
	   ASSERT_EQ( lat_cb1.GetNumCBSurfaceSites(mu,FORWARD,1), static_cast<unsigned int>(1));
   }

}

TEST(TestLattice, TestLatticeCoarseningStandardAggregation)
{
 LatticeInfo fine_geom({16,16,16,16}, 4, 3, NodeInfo());
 StandardAggregation blocking({4,4,4,4});
 unsigned int num_vec = 24;

 LatticeInfo coarse_geom = CoarsenLattice(fine_geom,blocking,num_vec);
 for(unsigned int mu=0; mu < n_dim; ++mu)
 {
	 ASSERT_EQ((coarse_geom.GetLatticeDimensions())[mu], static_cast<unsigned int>(4));
 }
 ASSERT_EQ(coarse_geom.GetNumColors(), static_cast<unsigned int>(24));
 ASSERT_EQ(coarse_geom.GetNumSpins(), static_cast<unsigned int>(2));


}

TEST(TestLattice, TestLatticeCoarseningFullSpinAggregation)
{
 LatticeInfo fine_geom({16,16,16,16}, 4, 3, NodeInfo());
 FullSpinAggregation blocking({4,4,4,4});
 unsigned int num_vec = 24;

 LatticeInfo coarse_geom = CoarsenLattice(fine_geom,blocking,num_vec);
 for(unsigned int mu=0; mu < n_dim; ++mu)
 {
	 ASSERT_EQ((coarse_geom.GetLatticeDimensions())[mu], static_cast<unsigned int>(4));
 }
 ASSERT_EQ(coarse_geom.GetNumColors(), static_cast<unsigned int>(24));
 ASSERT_EQ(coarse_geom.GetNumSpins(), static_cast<unsigned int>(1));


}

/* I should test Lattice Info for deat
 * h when in an OMP
 * parallel region but death tests and threads dont work well.
 */


int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

