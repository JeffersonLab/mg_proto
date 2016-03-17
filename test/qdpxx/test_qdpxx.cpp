#include "gtest/gtest.h"
#include "../test_env.h"
#include "../mock_nodeinfo.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "qdpxx_helpers.h"

using namespace MG;
using namespace MGTesting;
using namespace QDP;

TEST(TestLattice, TestLatticeInitialization)
{
	IndexArray latdims={{8,8,8,8}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

}


int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

