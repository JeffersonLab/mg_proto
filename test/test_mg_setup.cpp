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
			}, // block sizes
			{150,150},        // Null solver max iter
			{5.0e-6,5.0e-6}   // Null solver rsd target
	};

	mgSetup(params, mg_levels );
}

TEST(MGSetup, CreateVCycle)
{
	std::vector<MGLevel> mg_levels;
	SetupParams params = {
			3, // n_levels
			{24,32}, // n_vecs
			{16,16,16,16}, // local_lattice_size
			{
					{4,4,4,4},   // Block Sizes on level 0
					{2,2,2,2}    // Block Sizes on level 1
			}, // block sizes
			{150,150},        // Null solver max iter
			{5.0e-6,5.0e-6}   // Null solver rsd target
	};

	// Set Up Null Vecs and Linear Operators
	mgSetup(params, mg_levels );

	// V Ctcle parametere
	std::vector<VCycleParams> v_params={

			// pre smooth       coarse solve                 post smooth
			// iter,omega,rsd  rsd,iters,n_krylov,omega    iters, omega,rsd
			{   4,1.0,0.2,         0.2,100,10,1.0,            4,1.0,0.2},
			{   4,1.0,0.2,         0.2,100,10,1.0,            4,1.0,0.2}
	};

	MasterLog(INFO, "Creating Preconditioner");
	VCyclePreconditioner V_top(0,v_params,mg_levels);
	MasterLog(INFO, "Done");

	MasterLog(INFO, "Apply Preconditioner");
	Spinor *r = allocateSpinor(*(mg_levels[0].info), 0);
	Spinor *delta_x = allocateSpinor(*(mg_levels[0].info), 0);

	SolverParams precond_par; precond_par.max_iter=1; precond_par.rsd_target=0.1;

	// Apply preconditioner
	V_top(*delta_x, *r, precond_par );

	freeSpinor(r);
	freeSpinor(delta_x);

}


int main(int argc, char *argv[])
{
	return MGTesting::TestMain(&argc, argv);
}
