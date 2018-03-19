#include <lattice/coarse/invbicgstab_coarse.h>
#include <lattice/coarse/invfgmres_coarse.h>
#include <lattice/coarse/invmr_coarse.h>
#include <lattice/fine_qdpxx/invbicgstab_qdpxx.h>
#include <lattice/fine_qdpxx/invfgmres_qdpxx.h>
#include "gtest/gtest.h"
#include "../test_env.h"
#include "../mock_nodeinfo.h"
#include "qdpxx_utils.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_l1_blas.h"
#include "lattice/coarse/block.h"
#include "lattice/fine_qdpxx/qdpxx_helpers.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"
// Block Stuff
#include "lattice/fine_qdpxx/aggregate_block_qdpxx.h"
#include "lattice/fine_qdpxx/wilson_clover_linear_operator.h"
#include "lattice/coarse/coarse_wilson_clover_linear_operator.h"
#include "lattice/fine_qdpxx/vcycle_qdpxx_coarse.h"
#include "lattice/coarse/vcycle_coarse.h"


#include <memory>

using namespace MG;
using namespace MGTesting;
using namespace QDP;


TEST(TestVCycle, TestVCycleApply)
{
	IndexArray latdims={{4,4,4,4}};         // Lattice
	IndexArray blockdims = {{2,2,2,2}};     // Blocking

	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	IndexArray node_orig=NodeInfo().NodeCoords();
		for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

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
	BiCGStabSolverQDPXX  BiCGStab(M, params);

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
	IndexArray blocked_lattice_orig;
	CreateBlockList(my_blocks,blocked_lattice_dims,blocked_lattice_orig,latdims,blockdims,node_orig);

	// Orthonormalize the vectors -- I heard once that for GS stability is improved
	// if you do it twice.
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);


	// Create the blocked Clover and Gauge Fields
	// This service needs the blocks, the vectors and is a convenience
	// Function of the M
	LatticeInfo info(blocked_lattice_orig,blocked_lattice_dims, 2, NumVecs, NodeInfo());

	std::shared_ptr<CoarseGauge> u_coarse=std::make_shared<CoarseGauge>(info);

	// Coarsen M to compute the coarsened Gauge and Clover fields

	M.generateCoarse(my_blocks,vecs, *u_coarse);

	// Create a coarse operator
	// FIXME: NB: M could have a method to create a coarsened operator.
	// However then it would have to allocate u_coarse and clov_coarse
	// and they would need to be held via some refcounted pointer...
	// Come back to that
	CoarseWilsonCloverLinearOperator M_coarse(u_coarse,  1);


	// WE NOW HAVE: M_fine, M_coarse and the information to affect Intergrid
	// Transfers.

	// Set up the PreSmoother & Post Smootehr (on the fine level)
	MRSolverParams presmooth;
	presmooth.MaxIter=4;
	presmooth.RsdTarget = 0.1;
	presmooth.Omega = 1.1;
	presmooth.VerboseP = true;

	// MR Smoother holds only references
	MRSmootherQDPXX the_smoother(M,presmooth);


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
	IndexArray node_orig=NodeInfo().NodeCoords();
		for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];


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
	BiCGStabSolverQDPXX   BiCGStab(M, params);
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
	IndexArray blocked_lattice_orig;
	CreateBlockList(my_blocks,blocked_lattice_dims,blocked_lattice_orig,latdims,blockdims,node_orig);

	// Do the proper block orthogonalize
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);

	// Create the blocked Clover and Gauge Fields
	LatticeInfo info(blocked_lattice_orig,blocked_lattice_dims, 2, NumVecs, NodeInfo());
	std::shared_ptr<CoarseGauge> u_coarse=std::make_shared<CoarseGauge>(info);


	M.generateCoarse(my_blocks,vecs, *u_coarse);

	// Create a coarse operator
	// FIXME: NB: M could have a method to create a coarsened operator.
	// However then it would have to allocate u_coarse and clov_coarse
	// and they would need to be held via some refcounted pointer...
	// Come back to that
	CoarseWilsonCloverLinearOperator M_coarse(u_coarse, 1);

	MRSolverParams presmooth;
	presmooth.MaxIter=4;
	presmooth.RsdTarget = 0.1;
	presmooth.Omega = 1.1;
	presmooth.VerboseP = true;

	MRSmootherQDPXX the_smoother(M,presmooth);

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

	IndexArray node_orig=NodeInfo().NodeCoords();
		for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

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
	BiCGStabSolverQDPXX BiCGStab(M, params);
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
	IndexArray blocked_lattice_orig;
	CreateBlockList(my_blocks,blocked_lattice_dims,blocked_lattice_orig,latdims,blockdims, node_orig);

	// Do the proper block orthogonalize
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);

	// Create the blocked Clover and Gauge Fields
	QDPIO::cout << "NumVecs=" << NumVecs << std::endl;

	LatticeInfo info(blocked_lattice_orig,blocked_lattice_dims, 2, NumVecs, NodeInfo());

	std::shared_ptr<CoarseGauge> u_coarse=std::make_shared<CoarseGauge>(info);

	// Coarsen M to compute the coarsened Gauge and Clover fields
	M.generateCoarse(my_blocks,vecs, *u_coarse);

	// Create a coarse operator
	// FIXME: NB: M could have a method to create a coarsened operator.
	// However then it would have to allocate u_coarse and clov_coarse
	// and they would need to be held via some refcounted pointer...
	// Come back to that
	CoarseWilsonCloverLinearOperator M_coarse(u_coarse, 1);

	MRSolverParams presmooth;
	presmooth.MaxIter=4;
	presmooth.RsdTarget = 0.1;
	presmooth.Omega = 1.1;
	presmooth.VerboseP = true;

	MRSmootherQDPXX pre_smoother(M,presmooth);

	MRSolverParams postsmooth;
	postsmooth.MaxIter = 4;
	postsmooth.RsdTarget = 0.1;
	postsmooth.Omega = 1.1;
	postsmooth.VerboseP = true;

	MRSmootherQDPXX post_smoother(M,postsmooth);

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
	FGMRESSolverQDPXX FGMRESOuter(M,fine_solve_params, &vcycle);

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

TEST(TestVCycle, TestVCyclePrec8888)
{
	IndexArray latdims={{8,8,8,8}};
	IndexArray blockdims = {{2,2,2,2}};

	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	IndexArray node_orig=NodeInfo().NodeCoords();
		for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

	float m_q = 0.1;
	float c_sw = 1.25;

	int t_bc=-1; // Antiperiodic t BCs


	multi1d<LatticeColorMatrix> u(Nd);
	for(int mu=0; mu < Nd; ++mu) {
	// gaussian(u[mu]);
	//	reunit(u[mu]);
		u[mu]=1.0;
	}

	// Create linear operator
	QDPWilsonCloverLinearOperator M(m_q, c_sw, t_bc,u);
	LinearSolverParamsBase params;
	params.MaxIter = 500;
	params.RsdTarget = 1.0e-5;
	params.VerboseP = true;
	BiCGStabSolverQDPXX BiCGStab(M, params);
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
	IndexArray blocked_lattice_orig;
	CreateBlockList(my_blocks,blocked_lattice_dims,blocked_lattice_orig,latdims,blockdims, node_orig);

	// Do the proper block orthogonalize
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);

	// Create the blocked Clover and Gauge Fields
	QDPIO::cout << "NumVecs=" << NumVecs << std::endl;

	LatticeInfo info(blocked_lattice_orig,blocked_lattice_dims, 2, NumVecs, NodeInfo());
	std::shared_ptr<CoarseGauge> u_coarse = std::make_shared<CoarseGauge>(info);

	// Coarsen M to compute the coarsened Gauge and Clover fields
	M.generateCoarse(my_blocks,vecs, *u_coarse);

	// Create a coarse operator
	// FIXME: NB: M could have a method to create a coarsened operator.
	// However then it would have to allocate u_coarse and clov_coarse
	// and they would need to be held via some refcounted pointer...
	// Come back to that
	CoarseWilsonCloverLinearOperator M_coarse(u_coarse, 1);

	MRSolverParams presmooth;
	presmooth.MaxIter=4;
	presmooth.RsdTarget = 0.1;
	presmooth.Omega = 1.1;
	presmooth.VerboseP = true;

	MRSmootherQDPXX pre_smoother(M,presmooth);

	MRSolverParams postsmooth;
	postsmooth.MaxIter = 4;
	postsmooth.RsdTarget = 0.1;
	postsmooth.Omega = 1.1;
	postsmooth.VerboseP = true;

	MRSmootherQDPXX post_smoother(M,postsmooth);

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
	fine_solve_params.NKrylov = 5;
	FGMRESSolverQDPXX FGMRESOuter(M,fine_solve_params, &vcycle);

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

TEST(TestVCycle, TestVCycle2Level)
{
	IndexArray latdims={{8,8,8,8}};
	IndexArray blockdims = {{2,2,2,2}};


	initQDPXXLattice(latdims);

	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	IndexArray node_orig=NodeInfo().NodeCoords();
		for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

	float m_q = 0.1;
	float c_sw = 1.25;

	int t_bc=-1; // Antiperiodic t BCs


	multi1d<LatticeColorMatrix> u(Nd);
	for(int mu=0; mu < Nd; ++mu) {
		u[mu]=1;
		//gaussian(u[mu]);
		//reunit(u[mu]);
	}

	// Create linear operator
	QDPWilsonCloverLinearOperator M(m_q, c_sw, t_bc,u);
	LinearSolverParamsBase params;
	params.MaxIter = 500;
	params.RsdTarget = 1.0e-5;
	params.VerboseP = true;
	BiCGStabSolverQDPXX BiCGStab(M, params);
	LatticeFermion b=zero;

	QDPIO::cout << "Generating 6 Null Vectors" << std::endl;
	const int NumVecs=8;
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
	IndexArray blocked_lattice_orig;
	CreateBlockList(my_blocks,blocked_lattice_dims,blocked_lattice_orig,latdims,blockdims, node_orig);

	// Do the proper block orthogonalize
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);

	QDPIO::cout << "Creating Level 1 LinearOperator" <<std::endl;

	LatticeInfo info(blocked_lattice_orig,blocked_lattice_dims, 2, NumVecs, NodeInfo());
	std::shared_ptr<CoarseGauge> u_coarse = std::make_shared<CoarseGauge>(info);
	M.generateCoarse(my_blocks,vecs, *u_coarse);
	CoarseWilsonCloverLinearOperator M_coarse(u_coarse, 1);
	BiCGStabSolverCoarse BiCGStabL1( M_coarse, params);  // Use this to solve for zero vecs

	QDPIO::cout << "Level 1: Linear Operator and BiCGStab Solver Created" << std::endl;

	QDPIO::cout << "Creating Level 2 Vectors" <<std::endl;

	//---------------------------------------------------
	// Now make a second level
	//---------------------------------------------------
	const int NumVecs2 =8;
	std::vector<std::shared_ptr<CoarseSpinor> >  vecs_l2(NumVecs2);
	CoarseSpinor zero_l2(info);
	ZeroVec(zero_l2);
	for(int k=0; k < NumVecs2; ++k) {
		vecs_l2[k] = std::make_shared<CoarseSpinor>(info);
		Gaussian( *(vecs_l2[k]) );
		LinearSolverResults res = BiCGStabL1(*(vecs_l2[k]), zero_l2,ABSOLUTE);
		QDPIO::cout << "BiCGStab Solver Took: " << res.n_count << " iterations"
						<< std::endl;
	}

	QDPIO::cout << "L2: Orthonormzlizing Nullvecs" << std::endl;
	std::vector<Block> my_blocks_l2;
	IndexArray blocked_lattice_dims2;
	IndexArray blocked_lattice_orig2;
	CreateBlockList(my_blocks_l2, blocked_lattice_dims2, blocked_lattice_orig2, blocked_lattice_dims, blockdims, node_orig);

	// Do the proper block orthogonalize
	orthonormalizeBlockAggregates(vecs_l2, my_blocks_l2);
	orthonormalizeBlockAggregates(vecs_l2, my_blocks_l2);

	QDPIO::cout << "Creating Level 2 LinearOperator" <<std::endl;

	LatticeInfo info2(blocked_lattice_orig2, blocked_lattice_dims2, 2, NumVecs2, NodeInfo());
	std::shared_ptr<CoarseGauge> u_coarse_coarse = std::make_shared<CoarseGauge>(info2);
	M_coarse.generateCoarse(my_blocks_l2,vecs_l2, *u_coarse_coarse);
	CoarseWilsonCloverLinearOperator M_coarse_coarse(u_coarse_coarse, 2);

	QDPIO::cout << "Done" << std::endl;

	QDPIO::cout << "Creating Level 2 Vcycle" << std::endl;

	QDPIO::cout << "  ... creating L1 PreSmoother" << std::endl;
	MRSolverParams presmooth_12_params;
	presmooth_12_params.MaxIter=4;
	presmooth_12_params.RsdTarget = 0.1;
	presmooth_12_params.Omega = 1.1;
	presmooth_12_params.VerboseP = true;
	MRSmootherCoarse pre_smoother_12(M_coarse, presmooth_12_params);

	QDPIO::cout << "  ... creating L2 (bottom) FGMRES Solver." << std::endl;
	FGMRESParams l2_solve_params;
	l2_solve_params.MaxIter=200;
	l2_solve_params.RsdTarget=0.1;
	l2_solve_params.VerboseP = false;
	l2_solve_params.NKrylov = 10;
	FGMRESSolverCoarse l2_solver(M_coarse_coarse,l2_solve_params,nullptr); // Bottom solver, no preconditioner

	QDPIO::cout << "  ... creating L1 PostSmoother" << std::endl;
	MRSolverParams postsmooth_12_params;
	postsmooth_12_params.MaxIter=4;
	postsmooth_12_params.RsdTarget = 0.1;
	postsmooth_12_params.Omega = 1.1;
	postsmooth_12_params.VerboseP = true;
	MRSmootherCoarse post_smoother_12(M_coarse, postsmooth_12_params);

	QDPIO::cout << " ... creating L1 -> L2 VCycle" << std::endl;
	LinearSolverParamsBase vcycle_12_params;
	vcycle_12_params.MaxIter=1;
	vcycle_12_params.RsdTarget =0.1;
	vcycle_12_params.VerboseP = true;


	VCycleCoarse vcycle12( info2, my_blocks_l2, vecs_l2, M_coarse, pre_smoother_12, post_smoother_12, l2_solver, vcycle_12_params);
	QDPIO::cout << " ... done" << std::endl << std::endl;

	QDPIO::cout << "  ... creating L0 PreSmoother" << std::endl;
	MRSolverParams presmooth_01_params;
	presmooth_01_params.MaxIter=4;
	presmooth_01_params.RsdTarget = 0.1;
	presmooth_01_params.Omega = 1.1;
	presmooth_01_params.VerboseP = true;
	MRSmootherQDPXX pre_smoother_01(M, presmooth_01_params);

	QDPIO::cout << "  ... creating L1 FGMRES Solver -- preconditioned with L1->L2 Vcycle" << std::endl;
	FGMRESParams l1_solve_params;
	l1_solve_params.MaxIter=200;
	l1_solve_params.RsdTarget=0.1;
	l1_solve_params.VerboseP = false;
	l1_solve_params.NKrylov = 10;
	FGMRESSolverCoarse l1_solver(M_coarse,l1_solve_params,&vcycle12); // Bottom solver, no preconditioner

	QDPIO::cout << "  ... creating L0 Post Smoother" << std::endl;
	MRSolverParams postsmooth_01_params;
	postsmooth_01_params.MaxIter=4;
	postsmooth_01_params.RsdTarget = 0.1;
	postsmooth_01_params.Omega = 1.1;
	postsmooth_01_params.VerboseP = true;
	MRSmootherQDPXX post_smoother_01(M, postsmooth_01_params);

	QDPIO::cout << " ... creating L0 -> L1 Vcycle Preconitioner " << std::endl;
	LinearSolverParamsBase vcycle_01_params;
	vcycle_01_params.MaxIter=1;
	vcycle_01_params.RsdTarget =0.1;
	vcycle_01_params.VerboseP = true;

	VCycleQDPCoarse2 vcycle_01( info, my_blocks, vecs, M, pre_smoother_01, post_smoother_01, l1_solver, vcycle_01_params);
	QDPIO::cout << " ... Done" << std::endl;


	QDPIO::cout << "Creating Outer Solver with L0->L1 VCycle Preconditioner" << std::endl;
	FGMRESParams fine_solve_params;
	fine_solve_params.MaxIter=200;
	fine_solve_params.RsdTarget=1.0e-5;
	fine_solve_params.VerboseP = true;
	fine_solve_params.NKrylov = 5;
	FGMRESSolverQDPXX FGMRESOuter(M,fine_solve_params, &vcycle_01);

	QDPIO::cout << "*** Recursive VCycle Structure + Solver Created" << std::endl;

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

