#include "gtest/gtest.h"
#include "lattice/lattice.h"
#include "lattice/nodeinfo.h"

#include <vector>
#include "test_env.h"
using namespace MGGeometry; 

TEST(TestLattice, TestLatticeInitialization)
{
  std::vector<int> latdims = {24,24,24,64};
  NodeInfo node;
  Lattice lat(latdims, 4, 3,node);
}


int main(int argc, char *argv[]) 
{
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::Environment* const chroma_env = ::testing::AddGlobalTestEnvironment(new TestEnv(&argc,&argv));
  return RUN_ALL_TESTS();
}

