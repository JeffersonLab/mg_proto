#include "gtest/gtest.h"
#include "test_env.h"
#include <lattice/mg_level.h>

#include <vector>

using namespace MG;

TEST(MGSetup, CreateMGSetup)
{
	std::vector<MGLevel> mg_levels;
	SetupParams params = {
			3, // n_levels
			{24,32}, // n_vecs
			{16,16,16,16}, // local_lattice_size
			{
					{4,4,4,4},
					{2,2,2,2}
			} // block sizes
	};

	mgSetup(params, mg_levels );
}


int main(int argc, char *argv[])
{
	return MGTesting::TestMain(&argc, argv);
}
