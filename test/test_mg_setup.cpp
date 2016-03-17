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

	Gauge u;
		Clover clov;
		Clover invclov;

	mgSetup(params, mg_levels, &u, &clov, &invclov );
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


	Gauge u;
	Clover clov;
	Clover invclov;

	// Set Up Null Vecs and Linear Operators
	mgSetup(params, mg_levels, &u, &clov, &invclov );

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

TEST(MGSetup, MGPrecGCR)
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
	Gauge u;
	Clover clov;
	Clover invclov;

	// Set Up Null Vecs and Linear Operators
	mgSetup(params, mg_levels, &u, &clov, &invclov );

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

	Spinor *b = allocateSpinor(*(mg_levels[0].info), 0);
	Spinor *x = allocateSpinor(*(mg_levels[0].info), 0);

	SolverParams outer_gcr_params;
	outer_gcr_params.max_iter = 50;
	outer_gcr_params.rsd_target = 1.0e-7;
	outer_gcr_params.gmres_n_krylov = 10;
	outer_gcr_params.overrelax_omega = 1.0;

	Solver* OuterGCRSolver = createSolver(GCR, *(mg_levels[0].M), &V_top);

	// Solve system
	gaussian(*b);
	zero(*x);
	(*OuterGCRSolver)(*x,*b,outer_gcr_params);

	freeSpinor(x);
	freeSpinor(b);
	delete OuterGCRSolver;

}


int main(int argc, char *argv[])
{
	return MGTesting::TestMain(&argc, argv);
}
