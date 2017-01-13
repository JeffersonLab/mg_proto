#include "gtest/gtest.h"
#include "../test_env.h"
#include "../mock_nodeinfo.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_l1_blas.h"
#include "qdpxx_helpers.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"
// Block Stuff
#include "aggregate_block_qdpxx.h"

#include "reunit.h"
#include "wilson_clover_linear_operator.h"
#include "coarse_wilson_clover_linear_operator.h"
#include "invbicgstab.h"
#include "invfgmres.h"
#include "invfgmres_coarse.h"
#include "vcycle_qdpxx_coarse.h"

using namespace MG;
using namespace MGTesting;
using namespace QDP;


TEST(TestVCycle, TestVCycleApply)
{
	IndexArray latdims={{4,4,4,4}};         // Lattice
	IndexArray blockdims = {{2,2,2,2}};     // Blocking

	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	// Parameters
	float m_q = 0.1;
	float c_sw = 1.25;
	int t_bc=-1; // Antiperiodic t BCs

	// Setup QDP++ Lattice
	multi1d<LatticeColorMatrix> u(Nd);
	for(int mu=0; mu < Nd; ++mu) {
		gaussian(u[mu]);
		reunit(u[mu]);
	}

	// Create Fine linear operator
	QDPWilsonCloverLinearOperator M(m_q, c_sw, t_bc,u);



	// SETUP PHASE: Create a Nullspace using BiCGStab
	// Solve A x = 0 using random initial guesses
	//
	LinearSolverParamsBase params;
	params.MaxIter = 500;
	params.RsdTarget = 1.0e-5;
	params.VerboseP = true;
	BiCGStabSolver<LatticeFermion,multi1d<LatticeColorMatrix> >  BiCGStab(M, params);

	// Zero RHS
	LatticeFermion b=zero;

	// Generate the vectors
	QDPIO::cout << "Generating 6 Null Vectors" << std::endl;
	const int NumVecs=6;
	multi1d<LatticeFermion> vecs(NumVecs);
	for(int k=0; k < NumVecs; ++k) {
		gaussian(vecs[k]);
		LinearSolverResults res = BiCGStab(vecs[k],b, ABSOLUTE);
		QDPIO::cout << "BiCGStab Solver Took: " << res.n_count << " iterations"
				<< std::endl;
	}


	// Now Orthonormalize the nullspace vectors over the Blocks

	// Create List Of Blocks
	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	CreateBlockList(my_blocks,blocked_lattice_dims,latdims,blockdims);

	// Orthonormalize the vectors -- I heard once that for GS stability is improved
	// if you do it twice.
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);


	// Create the blocked Clover and Gauge Fields
	// This service needs the blocks, the vectors and is a convenience
	// Function of the M
	LatticeInfo info(blocked_lattice_dims, 2, NumVecs, NodeInfo());
	CoarseClover clov_coarse(info);
	CoarseGauge u_coarse(info);

	// Coarsen M to compute the coarsened Gauge and Clover fields
	M.generateCoarseClover(my_blocks,vecs, clov_coarse);
	M.generateCoarseGauge(my_blocks,vecs, u_coarse);

	// Create a coarse operator
	// FIXME: NB: M could have a method to create a coarsened operator.
	// However then it would have to allocate u_coarse and clov_coarse
	// and they would need to be held via some refcounted pointer...
	// Come back to that
	CoarseWilsonCloverLinearOperator M_coarse(&u_coarse, &clov_coarse, 1);


	// WE NOW HAVE: M_fine, M_coarse and the information to affect Intergrid
	// Transfers.

	// Set up the PreSmoother & Post Smootehr (on the fine level)
	MRSolverParams presmooth;
	presmooth.MaxIter=4;
	presmooth.RsdTarget = 0.1;
	presmooth.Omega = 1.1;
	presmooth.VerboseP = true;

	// MR Smoother holds only references
	MRSmoother<LatticeFermion, multi1d<LatticeColorMatrix>> the_smoother(M,presmooth);


	// Set up the CoarseSolver
	FGMRESParams coarse_solve_params;
	coarse_solve_params.MaxIter=200;
	coarse_solve_params.RsdTarget=0.1;
	coarse_solve_params.VerboseP = true;
	coarse_solve_params.NKrylov = 10;
	FGMRESSolverCoarse bottom_solver(M_coarse,coarse_solve_params);


	{
		// Create the 2 Level VCycle

		LinearSolverParamsBase vcycle_params;
		vcycle_params.MaxIter=1;                   // Single application
		vcycle_params.RsdTarget =0.1;              // The desired reduction in || r ||
		vcycle_params.VerboseP = true;			   // Verbosity

		// info my_blocks, and vecs can probably be collected in a 'Transfer' class
		VCycleQDPCoarse2 vcycle( info, my_blocks, vecs, M, the_smoother, the_smoother, bottom_solver, vcycle_params);


		// Now need to do the coarse test
		LatticeFermion psi_in, chi_out;
		gaussian(psi_in);
		chi_out = zero;
		QDPIO::cout << "psi_in has norm =" << sqrt(norm2(psi_in)) << std::endl;
		LinearSolverResults res = vcycle(chi_out,psi_in);
		ASSERT_EQ( res.n_count, 1);
		ASSERT_EQ( res.resid_type, RELATIVE );
		ASSERT_LT( res.resid, 3.8e-1 );
	}


}

TEST(TestVCycle, TestVCycleSolve)
{
	IndexArray latdims={{4,4,4,4}};
	IndexArray blockdims = {{2,2,2,2}};

	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;


	float m_q = 0.1;
	float c_sw = 1.25;

	int t_bc=-1; // Antiperiodic t BCs


	multi1d<LatticeColorMatrix> u(Nd);
	for(int mu=0; mu < Nd; ++mu) {
		gaussian(u[mu]);
		reunit(u[mu]);
	}

	// Create linear operator
	QDPWilsonCloverLinearOperator M(m_q, c_sw, t_bc,u);
	LinearSolverParamsBase params;
	params.MaxIter = 500;
	params.RsdTarget = 1.0e-5;
	params.VerboseP = true;
	BiCGStabSolver<LatticeFermion,multi1d<LatticeColorMatrix> >  BiCGStab(M, params);
	LatticeFermion b=zero;

	QDPIO::cout << "Generating 6 Null Vectors" << std::endl;
	const int NumVecs=6;
	multi1d<LatticeFermion> vecs(NumVecs);
	for(int k=0; k < NumVecs; ++k) {
		gaussian(vecs[k]);
		LinearSolverResults res = BiCGStab(vecs[k],b, ABSOLUTE);
		QDPIO::cout << "BiCGStab Solver Took: " << res.n_count << " iterations"
				<< std::endl;
	}



	// Someone once said doing this twice is good
	QDPIO::cout << "Orthonormalizing Nullvecs" << std::endl;

	// 1) Create the blocklist
	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	CreateBlockList(my_blocks,blocked_lattice_dims,latdims,blockdims);

	// Do the proper block orthogonalize
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);

	// Create the blocked Clover and Gauge Fields
	LatticeInfo info(blocked_lattice_dims, 2, NumVecs, NodeInfo());
	CoarseClover clov_coarse(info);
	CoarseGauge u_coarse(info);

	// Coarsen M to compute the coarsened Gauge and Clover fields
	M.generateCoarseClover(my_blocks,vecs, clov_coarse);
	M.generateCoarseGauge(my_blocks,vecs, u_coarse);

	// Create a coarse operator
	// FIXME: NB: M could have a method to create a coarsened operator.
	// However then it would have to allocate u_coarse and clov_coarse
	// and they would need to be held via some refcounted pointer...
	// Come back to that
	CoarseWilsonCloverLinearOperator M_coarse(&u_coarse, &clov_coarse, 1);

	MRSolverParams presmooth;
	presmooth.MaxIter=4;
	presmooth.RsdTarget = 0.1;
	presmooth.Omega = 1.1;
	presmooth.VerboseP = true;

	MRSmoother<LatticeFermion, multi1d<LatticeColorMatrix>> the_smoother(M,presmooth);

	FGMRESParams coarse_solve_params;
	coarse_solve_params.MaxIter=200;
	coarse_solve_params.RsdTarget=0.1;
	coarse_solve_params.VerboseP = true;
	coarse_solve_params.NKrylov = 10;
	FGMRESSolverCoarse bottom_solver(M_coarse,coarse_solve_params);

	{
		LinearSolverParamsBase vcycle_params;
		vcycle_params.MaxIter=500;
		vcycle_params.RsdTarget =0.00005;
		vcycle_params.VerboseP = true;

		VCycleQDPCoarse2 vcycle( info, my_blocks, vecs, M, the_smoother, the_smoother, bottom_solver, vcycle_params);


		// Now need to do the coarse test
		LatticeFermion psi_in, chi_out;
		gaussian(psi_in);
		chi_out = zero;
		QDPIO::cout << "psi_in has norm =" << sqrt(norm2(psi_in)) << std::endl;
		LinearSolverResults res = vcycle(chi_out,psi_in);

		LatticeFermion r=psi_in;
		LatticeFermion tmp;
		M(tmp,chi_out,LINOP_OP);
		r -= tmp;
		Double diff=sqrt(norm2(r));
		Double diff_rel = diff / sqrt(norm2(psi_in));
		QDPIO::cout << "|| b - A x || = " << diff << std::endl;
		QDPIO::cout << "|| b - A x || = " << diff_rel << std::endl;
		ASSERT_EQ( res.resid_type, RELATIVE);
		ASSERT_LT( res.resid, 5.0e-5);
		ASSERT_LT( toDouble(diff_rel), 5.0e-5);


	}


}

TEST(TestVCycle, TestVCyclePrec)
{
	IndexArray latdims={{4,4,4,4}};
	IndexArray blockdims = {{2,2,2,2}};

	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;


	float m_q = 0.1;
	float c_sw = 1.25;

	int t_bc=-1; // Antiperiodic t BCs


	multi1d<LatticeColorMatrix> u(Nd);
	for(int mu=0; mu < Nd; ++mu) {
		gaussian(u[mu]);
		reunit(u[mu]);
	}

	// Create linear operator
	QDPWilsonCloverLinearOperator M(m_q, c_sw, t_bc,u);
	LinearSolverParamsBase params;
	params.MaxIter = 500;
	params.RsdTarget = 1.0e-5;
	params.VerboseP = true;
	BiCGStabSolver<LatticeFermion,multi1d<LatticeColorMatrix> >  BiCGStab(M, params);
	LatticeFermion b=zero;

	QDPIO::cout << "Generating 6 Null Vectors" << std::endl;
	const int NumVecs=6;
	multi1d<LatticeFermion> vecs(NumVecs);
	for(int k=0; k < NumVecs; ++k) {
		gaussian(vecs[k]);
		LinearSolverResults res = BiCGStab(vecs[k],b, ABSOLUTE);
		QDPIO::cout << "BiCGStab Solver Took: " << res.n_count << " iterations"
				<< std::endl;
	}



	// Someone once said doing this twice is good
	QDPIO::cout << "Orthonormalizing Nullvecs" << std::endl;

	// 1) Create the blocklist
	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	CreateBlockList(my_blocks,blocked_lattice_dims,latdims,blockdims);

	// Do the proper block orthogonalize
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);

	// Create the blocked Clover and Gauge Fields
	QDPIO::cout << "NumVecs=" << NumVecs << std::endl;

	LatticeInfo info(blocked_lattice_dims, 2, NumVecs, NodeInfo());
	CoarseClover clov_coarse(info);
	CoarseGauge u_coarse(info);

	// Coarsen M to compute the coarsened Gauge and Clover fields
	M.generateCoarseClover(my_blocks,vecs, clov_coarse);
	M.generateCoarseGauge(my_blocks,vecs, u_coarse);

	// Create a coarse operator
	// FIXME: NB: M could have a method to create a coarsened operator.
	// However then it would have to allocate u_coarse and clov_coarse
	// and they would need to be held via some refcounted pointer...
	// Come back to that
	CoarseWilsonCloverLinearOperator M_coarse(&u_coarse, &clov_coarse, 1);

	MRSolverParams presmooth;
	presmooth.MaxIter=4;
	presmooth.RsdTarget = 0.1;
	presmooth.Omega = 1.1;
	presmooth.VerboseP = true;

	MRSmoother<LatticeFermion, multi1d<LatticeColorMatrix>> pre_smoother(M,presmooth);

	MRSolverParams postsmooth;
	postsmooth.MaxIter = 4;
	postsmooth.RsdTarget = 0.1;
	postsmooth.Omega = 1.1;
	postsmooth.VerboseP = true;

	MRSmoother<LatticeFermion, multi1d<LatticeColorMatrix>> post_smoother(M,postsmooth);

	FGMRESParams coarse_solve_params;
	coarse_solve_params.MaxIter=200;
	coarse_solve_params.RsdTarget=0.1;
	coarse_solve_params.VerboseP = false;
	coarse_solve_params.NKrylov = 10;
	FGMRESSolverCoarse bottom_solver(M_coarse,coarse_solve_params);

	LinearSolverParamsBase vcycle_params;
	vcycle_params.MaxIter=1;
	vcycle_params.RsdTarget =0.1;
	vcycle_params.VerboseP = true;

	VCycleQDPCoarse2 vcycle( info, my_blocks, vecs, M, pre_smoother, post_smoother, bottom_solver, vcycle_params);


	FGMRESParams fine_solve_params;
	fine_solve_params.MaxIter=200;
	fine_solve_params.RsdTarget=1.0e-5;
	fine_solve_params.VerboseP = true;
	fine_solve_params.NKrylov = 2;
	FGMRESSolver FGMRESOuter(M,fine_solve_params, &vcycle);

	// Now need to do the coarse test
	LatticeFermion psi_in, chi_out;
	gaussian(psi_in);
	chi_out = zero;
	QDPIO::cout << "psi_in has norm =" << sqrt(norm2(psi_in)) << std::endl;
	LinearSolverResults res=FGMRESOuter(chi_out, psi_in);
	QDPIO::cout << "Returned residue = || r || / || b ||="<< res.resid<< std::endl;

	LatticeFermion r=psi_in;
	LatticeFermion tmp;
	M(tmp,chi_out,LINOP_OP);
	r -= tmp;
	Double diff=sqrt(norm2(r));
	Double diff_rel = diff / sqrt(norm2(psi_in));
	QDPIO::cout << "|| b - A x || = " << diff << std::endl;
	QDPIO::cout << "|| b - A x || = " << diff_rel << std::endl;
	ASSERT_EQ( res.resid_type, RELATIVE);
	ASSERT_LT( res.resid, 1.0e-5);
	ASSERT_LT( toDouble(diff_rel), 1.0e-5);

}



int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

