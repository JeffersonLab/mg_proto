#include "gtest/gtest.h"
#include "lattice/constants.h"
#include "lattice/lattice_spinor.h"
#include "utils/print_utils.h"
#include <vector>

#include "test_env.h"


/* I should test Lattice Info for deat
 * h when in an OMP
 * parallel region but death tests and threads dont work well.
 */


int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

