#include "gtest/gtest.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/nodeinfo.h"

#include "utils/print_utils.h"
#include "lattice/coarsen.h"
#include "lattice/geometry_utils.h"
#include <vector>
#include <random>
#include "test_env.h"
#include "mock_nodeinfo.h"

using namespace MGGeometry; 

TEST(TestLattice, TestLatticeInitialization)
{
  IndexArray latdims={{16,16,16,16}};
  NodeInfo node;
  LatticeInfo lat(latdims, 4, 3,node);
}

TEST(TestLattice, TestLatticeInitializationConvenience)
{
	LatticeInfo  lattinfo({{16,16,16,16}}, 4, 3, NodeInfo());
}

TEST(TestLattice, TestLatticeCB)
{
  /* Even number of sites */
  IndexArray latdims = {{16,16,16,16}};
  NodeInfo node;
  LatticeInfo lat(latdims, 4, 3,node);


  // Check that both CB's have half the number of sites
  ASSERT_EQ( lat.GetNumCBSites(), lat.GetNumSites()/2 );
  ASSERT_EQ( lat.GetNumCBSites(), lat.GetNumSites()/2 );

}



TEST(TestLattice, TestSurfacesBasic)
{
	IndexArray latdims = {{4,4,4,4}};
	NodeInfo node;
	LatticeInfo lat(latdims, 4, 3, node);

	for(IndexType mu=0; mu < n_dim; ++mu) {
		for(IndexType fb = BACKWARD; fb <= FORWARD; ++fb) {

			IndexType surface_sites = lat.GetNumCBSurfaceSites(mu)
					+ lat.GetNumCBSurfaceSites(mu);
			ASSERT_EQ( surface_sites, static_cast<IndexType>(4*4*4) );


		}
	}
}

TEST(TestLattice, TestSurfaceSiteTablesSorted)
{
	IndexArray latdims = {{4,4,4,4}};
	NodeInfo node;
	LatticeInfo lat(latdims, 4, 3, node);

	for(IndexType mu=X_DIR; mu <= T_DIR; ++mu) {

			IndexType surface_sites = 2*lat.GetNumCBSurfaceSites(mu);
			ASSERT_EQ( surface_sites, static_cast<IndexType>(4*4*4) );

	}
}


TEST(TestLattice, TestLatticeCoarseningStandardAggregation)
{
 LatticeInfo fine_geom({{16,16,16,16}}, 4, 3, NodeInfo());
 StandardAggregation blocking(fine_geom.GetLatticeDimensions(),{{4,4,4,4}});
 IndexType num_vec = 24;

 LatticeInfo coarse_geom = CoarsenLattice(fine_geom,blocking,num_vec);
 for(IndexType mu=0; mu < n_dim; ++mu)
 {
	 ASSERT_EQ((coarse_geom.GetLatticeDimensions())[mu], static_cast<IndexType>(4));
 }
 ASSERT_EQ(coarse_geom.GetNumColors(), static_cast<IndexType>(24));
 ASSERT_EQ(coarse_geom.GetNumSpins(), static_cast<IndexType>(2));


}



TEST(TestGeometryUtils, TestIndexToCoords)
{


	IndexArray dims1 = {{2,4,6,8}};
	IndexType index = 1;
	IndexArray coords;

	IndexToCoords(index, dims1, coords);

	IndexArray expected={{1,0,0,0}};

	ASSERT_EQ( coords,expected );
	index=2;
	expected={{0,1,0,0}};
	IndexToCoords(index,dims1,coords);
	ASSERT_EQ( coords, expected );

	index=8;
	expected={{0,0,1,0}};
	IndexToCoords(index,dims1,coords);
	ASSERT_EQ( coords, expected);

	index=48;
	expected={{0,0,0,1}};
	IndexToCoords(index,dims1, coords);
	ASSERT_EQ( coords, expected);

	expected={{1,2,3,4}};
	index=1+dims1[0]*(2+dims1[1]*(3 + dims1[2]*4));
	IndexToCoords(index,dims1,coords);
	ASSERT_EQ( coords, expected);
}

TEST(TestGeometryUtils, TestCoordsToIndex)
{
	IndexArray dims1 = {{2,4,6,8}};
	IndexArray coords ={{1,0,0,0}};

	IndexType index = CoordsToIndex(coords, dims1);
	IndexType expected=1;
	ASSERT_EQ(index,expected);

	coords={{0,1,0,0}};
	expected = 2;
	ASSERT_EQ(CoordsToIndex(coords,dims1), expected);

	coords={{0,0,1,0}};
	expected= 8;
	ASSERT_EQ(CoordsToIndex(coords,dims1), expected);

	coords={{0,0,0,1}};
	expected= 48;
	ASSERT_EQ(CoordsToIndex(coords,dims1), expected);

	coords={{1,3,5,2}};
	expected=1+dims1[0]*(3+dims1[1]*(5+dims1[2]*2));
	ASSERT_EQ(CoordsToIndex(coords,dims1), expected);
}

#if 0
TEST(TestGeometryUtils, TestCoordsToIndexIsInverseOfIndexToCoords)
{
	IndexArray  dims={{3,7,2,4}};
	IndexType max_index = (3*7*2*4)-1;
	IndexArray  coords;
	std::random_device rd;
	std::default_random_engine engine(rd());
	std::uniform_int_distribution<> dis(0,max_index);

	for(int i=0; i < 20; ++i) {
		IndexType index = dis(engine);
		IndexToCoords(index, dims, coords);
		ASSERT_EQ( index, CoordsToIndex(coords,dims));
	}


}
#endif
/* I should test Lattice Info for deat
 * h when in an OMP
 * parallel region but death tests and threads dont work well.
 */


int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

