#include "gtest/gtest.h"
#include "../test_env.h"
#include "../mock_nodeinfo.h"
#include "../qdpxx/qdpxx_utils.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/fine_qdpxx/dslashm_w.h"

#include "./kokkos_types.h"
#include "./kokkos_defaults.h"
#include "./kokkos_qdp_utils.h"
#include "./kokkos_spinproj.h"
#include "./kokkos_matvec.h"
#include "./kokkos_dslash.h"
#include "MG_config.h"
#include "kokkos_vnode.h"


using namespace MG;
using namespace MGTesting;
using namespace QDP;

int main(int argc, char *argv[]) 
{
	return ::MGTesting::TestMain(&argc, argv);
}

