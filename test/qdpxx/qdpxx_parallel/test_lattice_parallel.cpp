#include "gtest/gtest.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/nodeinfo.h"

#include "utils/print_utils.h"
#include "lattice/geometry_utils.h"
#include "../qdpxx_helpers.h"
#include <vector>
#include <random>
#include "../../test_env.h"


using namespace MG; 
using namespace MGTesting;

TEST(TestLatticeParallel, TestLatticeInitialization)
{

  IndexArray latdims={{4,4,4,4}};
  NodeInfo node;
  LatticeInfo lat(latdims, 4, 3, node);
  ASSERT_EQ( node.NumNodes(), 2);
  IndexArray gdims;
  lat.LocalDimsToGlobalDims(gdims,latdims);
  ASSERT_EQ( gdims[3],8);
  initQDPXXLattice(gdims);

}




int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}
