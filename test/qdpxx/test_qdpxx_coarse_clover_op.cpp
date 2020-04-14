#include <lattice/coarse/invbicgstab_coarse.h>
#include <lattice/coarse/invfgmres_coarse.h>
#include <lattice/coarse/invmr_coarse.h>
#include <lattice/fine_qdpxx/invbicgstab_qdpxx.h>
#include <lattice/fine_qdpxx/invmr_qdpxx.h>
#include "gtest/gtest.h"
#include "../test_env.h"
#include "../mock_nodeinfo.h"
#include "qdpxx_utils.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_l1_blas.h"
#include "lattice/coarse/block.h"
#include "lattice/coarse/coarse_wilson_clover_linear_operator.h"
#include "lattice/fine_qdpxx/qdpxx_helpers.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"
// Block Stuff
#include "lattice/fine_qdpxx/aggregate_block_qdpxx.h"
#include "lattice/fine_qdpxx/wilson_clover_linear_operator.h"


using namespace MG;
using namespace MGTesting;
using namespace QDP;

TEST(TestLattice, CoarseLinOpRandomNullVecs)
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

	// Make some Nullvecs
	// Random Basis vectors - for this test
	const int NumVecs=6;
	multi1d<LatticeFermion> vecs(NumVecs);
	for(int k=0; k < NumVecs; ++k) {
		gaussian(vecs[k]);
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
	LatticeInfo info(blocked_lattice_orig,blocked_lattice_dims, 2, 6, NodeInfo());
	std::shared_ptr<CoarseGauge> u_coarse = std::make_shared<CoarseGauge>(info);

	// Coarsen M to compute the coarsened Gauge and Clover fields
	M.generateCoarse(my_blocks,vecs, *u_coarse);


	// Create a coarse operator
	// FIXME: NB: M could have a method to create a coarsened operator.
	// However then it would have to allocate u_coarse and clov_coarse
	// and they would need to be held via some refcounted pointer...
	// Come back to that
	CoarseWilsonCloverLinearOperator M_coarse(u_coarse, 1);

	// Now need to do the coarse test
	LatticeFermion psi_in,tmp1,tmp2;
	gaussian(psi_in);

	CoarseSpinor psi_in_coarse(info);
	CoarseSpinor coarse_out(info);
	CoarseSpinor fake_coarse_out(info);

	for(IndexType OpType=LINOP_OP; OpType <= LINOP_DAGGER; ++OpType) {
		restrictSpinorQDPXXFineToCoarse(my_blocks, vecs, psi_in, psi_in_coarse);
		M_coarse(coarse_out,psi_in_coarse,OpType);

		// Prolongate to form coarse_d_psi = P psi_f
		prolongateSpinorCoarseToQDPXXFine(my_blocks,vecs, psi_in_coarse,tmp1);
		M(tmp2,tmp1,OpType);
		restrictSpinorQDPXXFineToCoarse(my_blocks, vecs, tmp2, fake_coarse_out);

		// Now I need to diff coarse_out and fake_coarse_out
		double diff_norm = XmyNorm2Vec(fake_coarse_out,coarse_out)[0];
		double coarse_out_norm = Norm2Vec(coarse_out)[0];
		double rel_diff = sqrt(diff_norm/coarse_out_norm);
		QDPIO::cout << "|| coarse_out - fake_coarse_out ||=" << sqrt(diff_norm) <<std::endl;
		QDPIO::cout << "|| coarse_out - fake_coarse_out ||/||coarse_out||=" <<rel_diff <<std::endl;
		ASSERT_LT( rel_diff, 5.0e-7);
	}


}


TEST(TestLattice, CoarseLinOpMRInv)
{
	IndexArray latdims={{4,4,4,4}};
	IndexArray blockdims = {{1,1,1,1}};

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

	const int NumVecs=6;
	multi1d<LatticeFermion> vecs(NumVecs);
	for(int k=0; k < NumVecs; ++k) {
		gaussian(vecs[k]);
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
	LatticeInfo info(blocked_lattice_orig,blocked_lattice_dims, 2, 6, NodeInfo());

	std::shared_ptr<CoarseGauge> u_coarse=std::make_shared<CoarseGauge>(info);

	M.generateCoarse(my_blocks,vecs, *u_coarse);

	// Create a coarse operator
	// FIXME: NB: M could have a method to create a coarsened operator.
	// However then it would have to allocate u_coarse and clov_coarse
	// and they would need to be held via some refcounted pointer...
	// Come back to that
	CoarseWilsonCloverLinearOperator M_coarse(u_coarse, 1);

	// Now need to do the coarse test
	LatticeFermion psi_in,tmp1,tmp2;
	gaussian(psi_in);
	QDPIO::cout << "psi_in has norm=" << sqrt(norm2(psi_in)) << std::endl;


	CoarseSpinor psi_in_coarse(info);

	// Do the solve
	// restrict psi_in to create the test vector
	restrictSpinorQDPXXFineToCoarse(my_blocks, vecs, psi_in, psi_in_coarse);



	CoarseSpinor coarse_solution(info);

	CoarseSpinor solution_check(info);

	MRSolverParams p;
	p.MaxIter=500;
	p.Omega=1.1;
	p.RsdTarget = 1.0e-5;
	p.VerboseP = true;

	// Create an FGMRES Solver for the coarse Op


	MRSolverCoarse CoarseMR( M_coarse, p );

	ZeroVec(coarse_solution);
	CoarseMR(coarse_solution, psi_in_coarse);

	M_coarse(solution_check, coarse_solution, LINOP_OP);

	double diff_norm = XmyNorm2Vec(solution_check,psi_in_coarse)[0];
		double psi_norm = Norm2Vec(psi_in_coarse)[0];
		double rel_diff = sqrt(diff_norm/psi_norm);
		QDPIO::cout << "|| b - Ax ||=" << sqrt(diff_norm) <<std::endl;
		QDPIO::cout << "|| b - Ax ||/|| b ||=" <<rel_diff <<std::endl;
		ASSERT_LT( rel_diff, 5.0e-5);

	LatticeFermion  solution_out;
	prolongateSpinorCoarseToQDPXXFine(my_blocks,vecs,coarse_solution,solution_out);

	LatticeFermion r;
	M(r,solution_out,LINOP_OP);
	r -= psi_in;
	Double norm_r = sqrt(norm2(r));
	Double rel_norm_r = norm_r / sqrt(norm2(psi_in));
	QDPIO::cout << " Fine Level: || M (Px) - b || = " << norm_r << std::endl;
	QDPIO::cout << " Fine Level: || M (Px) - b || = " << rel_norm_r << std::endl;
	ASSERT_LT( toDouble(rel_norm_r), 5.0e-5);

}

TEST(TestLattice, CoarseLinOpMRSmoother)
{
	IndexArray latdims={{4,4,4,4}};
	IndexArray blockdims = {{1,1,1,1}};

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

	const int NumVecs=6;
	multi1d<LatticeFermion> vecs(NumVecs);
	for(int k=0; k < NumVecs; ++k) {
		gaussian(vecs[k]);
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
	LatticeInfo info(blocked_lattice_orig,blocked_lattice_dims, 2, 6, NodeInfo());

	std::shared_ptr<CoarseGauge> u_coarse=std::make_shared<CoarseGauge>(info);

	M.generateCoarse(my_blocks,vecs, *u_coarse);

	// Create a coarse operator
	// FIXME: NB: M could have a method to create a coarsened operator.
	// However then it would have to allocate u_coarse and clov_coarse
	// and they would need to be held via some refcounted pointer...
	// Come back to that
	CoarseWilsonCloverLinearOperator M_coarse(u_coarse, 1);

	// Now need to do the coarse test
	LatticeFermion psi_in,smooth_out;
	gaussian(psi_in);
	QDPIO::cout << "psi_in has norm=" << sqrt(norm2(psi_in)) << std::endl;


	CoarseSpinor psi_in_coarse(info);
	CoarseSpinor smooth_out_coarse(info);
	restrictSpinorQDPXXFineToCoarse(my_blocks, vecs, psi_in, psi_in_coarse);


	MRSolverParams p;
	p.MaxIter=5;
	p.Omega=1.1;
	p.RsdTarget = 1.0e-5;
	p.VerboseP = true;

	// Create an FGMRES Solver for the coarse Op

	MRSmootherQDPXX FineMRSmoother( M, p);
	MRSmootherCoarse CoarseMRSmoother( M_coarse, p );

	smooth_out = zero; // Zero fine solution
	ZeroVec(smooth_out_coarse); // Zero Coarse solution

	FineMRSmoother(smooth_out,psi_in);
	CoarseMRSmoother(smooth_out_coarse,psi_in_coarse);
	LatticeFermion  P_smooth_out;
	prolongateSpinorCoarseToQDPXXFine(my_blocks,vecs,smooth_out_coarse,P_smooth_out);

	LatticeFermion diff = smooth_out - P_smooth_out;
	Double norm_diff = norm2(diff);
	Double rel_norm_diff = norm2(diff)/norm2(smooth_out);
	QDPIO::cout << "|| v_fine - P v_coarse ||=" << sqrt(norm_diff) <<std::endl;
	QDPIO::cout << "|| v_fine - P v_coarse ||/|| v_fine ||=" << sqrt(rel_norm_diff) <<std::endl;

}

TEST(TestLattice, CoarseLinOpBiCGStabInv)
{
	IndexArray latdims={{4,4,4,4}};
	IndexArray blockdims = {{1,1,1,1}};

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

	const int NumVecs=6;
	multi1d<LatticeFermion> vecs(NumVecs);
	for(int k=0; k < NumVecs; ++k) {
		gaussian(vecs[k]);
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
	LatticeInfo info(blocked_lattice_orig,blocked_lattice_dims, 2, 6, NodeInfo());

	std::shared_ptr<CoarseGauge> u_coarse=std::make_shared<CoarseGauge>(info);

	M.generateCoarse(my_blocks,vecs, *u_coarse);

	// Create a coarse operator
	// FIXME: NB: M could have a method to create a coarsened operator.
	// However then it would have to allocate u_coarse and clov_coarse
	// and they would need to be held via some refcounted pointer...
	// Come back to that
	CoarseWilsonCloverLinearOperator M_coarse(u_coarse, 1);

	// Now need to do the coarse test
	LatticeFermion psi_in,tmp1,tmp2;
	gaussian(psi_in);
	QDPIO::cout << "psi_in has norm=" << sqrt(norm2(psi_in)) << std::endl;


	CoarseSpinor psi_in_coarse(info);

	// Do the solve
	// restrict psi_in to create the test vector
	restrictSpinorQDPXXFineToCoarse(my_blocks, vecs, psi_in, psi_in_coarse);



	CoarseSpinor coarse_solution(info);

	CoarseSpinor solution_check(info);

	LinearSolverParamsBase p;
	p.MaxIter=500;
	p.RsdTarget = 1.0e-5;
	p.VerboseP = true;

	// Create an FGMRES Solver for the coarse Op


	BiCGStabSolverCoarse CoarseBiCG( M_coarse, p );

	ZeroVec(coarse_solution);
	CoarseBiCG(coarse_solution, psi_in_coarse);

	M_coarse(solution_check, coarse_solution, LINOP_OP);

	double diff_norm = XmyNorm2Vec(solution_check,psi_in_coarse)[0];
		double psi_norm = Norm2Vec(psi_in_coarse)[0];
		double rel_diff = sqrt(diff_norm/psi_norm);
		QDPIO::cout << "|| b - Ax ||=" << sqrt(diff_norm) <<std::endl;
		QDPIO::cout << "|| b - Ax ||/|| b ||=" <<rel_diff <<std::endl;
		ASSERT_LT( rel_diff, 5.0e-5);

	LatticeFermion  solution_out;
	prolongateSpinorCoarseToQDPXXFine(my_blocks,vecs,coarse_solution,solution_out);

	LatticeFermion r;
	M(r,solution_out,LINOP_OP);
	r -= psi_in;
	Double norm_r = sqrt(norm2(r));
	Double rel_norm_r = norm_r / sqrt(norm2(psi_in));
	QDPIO::cout << " Fine Level: || M (Px) - b || = " << norm_r << std::endl;
	QDPIO::cout << " Fine Level: || M (Px) - b || = " << rel_norm_r << std::endl;
	ASSERT_LT( toDouble(rel_norm_r), 5.0e-5);

}

TEST(TestLattice, CoarseLinOpFGMRESInvTrivial)
{
	IndexArray latdims={{4,4,4,4}};
	IndexArray blockdims = {{1,1,1,1}};

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

	const int NumVecs=6;
	multi1d<LatticeFermion> vecs(NumVecs);
	for(int k=0; k < NumVecs; ++k) {
		gaussian(vecs[k]);
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
	LatticeInfo info(blocked_lattice_orig,blocked_lattice_dims, 2, 6, NodeInfo());

	std::shared_ptr<CoarseGauge> u_coarse=std::make_shared<CoarseGauge>(info);

	M.generateCoarse(my_blocks,vecs, *u_coarse);

	// Create a coarse operator
	// FIXME: NB: M could have a method to create a coarsened operator.
	// However then it would have to allocate u_coarse and clov_coarse
	// and they would need to be held via some refcounted pointer...
	// Come back to that
	CoarseWilsonCloverLinearOperator M_coarse(u_coarse, 1);

	// Now need to do the coarse test
	LatticeFermion psi_in,tmp1,tmp2;
	gaussian(psi_in);
	QDPIO::cout << "psi_in has norm=" << sqrt(norm2(psi_in)) << std::endl;


	CoarseSpinor psi_in_coarse(info);

	// Do the solve
	// restrict psi_in to create the test vector
	restrictSpinorQDPXXFineToCoarse(my_blocks, vecs, psi_in, psi_in_coarse);

	{
		LatticeFermion out_M;
		M(out_M, psi_in, LINOP_OP);

		CoarseSpinor out_M_coarse(info);
		M_coarse(out_M_coarse,psi_in_coarse,LINOP_OP);
		LatticeFermion out_M_coarse_compare;
		prolongateSpinorCoarseToQDPXXFine(my_blocks, vecs, out_M_coarse,out_M_coarse_compare);
		Double diff_norm = norm2(out_M-out_M_coarse_compare);
		QDPIO::cout << "Diff = " << sqrt(diff_norm) << std::endl;
		QDPIO::cout << "RelDiff = "<< sqrt(diff_norm/norm2(out_M)) << std::endl;
	}

	CoarseSpinor coarse_solution(info);

	CoarseSpinor solution_check(info);

	FGMRESParams p;
	p.MaxIter=500;
	p.NKrylov = 10;
	p.RsdTarget = 1.0e-5;
	p.VerboseP = true;

	// Create an FGMRES Solver for the coarse Op


	FGMRESSolverCoarse CoarseFGMRES( M_coarse, p, nullptr);

	LatticeFermion tmp;
	CoarseSpinorToQDPSpinor( psi_in_coarse, tmp );
	ZeroVec(coarse_solution);
	CoarseFGMRES(coarse_solution, psi_in_coarse);

	M_coarse(solution_check, coarse_solution, LINOP_OP);

	double diff_norm = XmyNorm2Vec(solution_check,psi_in_coarse)[0];
		double psi_norm = Norm2Vec(psi_in_coarse)[0];
		double rel_diff = sqrt(diff_norm/psi_norm);
		QDPIO::cout << "|| b - Ax ||=" << sqrt(diff_norm) <<std::endl;
		QDPIO::cout << "|| b - Ax ||/|| b ||=" <<rel_diff <<std::endl;
		ASSERT_LT( rel_diff, 5.0e-5);

}


TEST(TestLattice, CoarseLinOpFGMRESInvBlocked)
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

	const int NumVecs=6;
	multi1d<LatticeFermion> vecs(NumVecs);
	for(int k=0; k < NumVecs; ++k) {
		gaussian(vecs[k]);
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
	LatticeInfo info(blocked_lattice_orig,blocked_lattice_dims, 2, 6, NodeInfo());

	std::shared_ptr<CoarseGauge> u_coarse= std::make_shared<CoarseGauge>(info);

	// Coarsen M to compute the coarsened Gauge and Clover fields

	M.generateCoarse(my_blocks,vecs, *u_coarse);

	// Create a coarse operator
	// FIXME: NB: M could have a method to create a coarsened operator.
	// However then it would have to allocate u_coarse and clov_coarse
	// and they would need to be held via some refcounted pointer...
	// Come back to that
	CoarseWilsonCloverLinearOperator M_coarse(u_coarse, 1);

	// Now need to do the coarse test
	LatticeFermion psi_in,tmp1,tmp2;
	gaussian(psi_in);
	QDPIO::cout << "psi_in has norm=" << sqrt(norm2(psi_in)) << std::endl;




	FGMRESParams p;
	p.MaxIter=500;
	p.NKrylov = 10;
	p.RsdTarget = 1.0e-5;
	p.VerboseP = true;

	// Create an FGMRES Solver for the coarse Op
	FGMRESSolverCoarse CoarseFGMRES( M_coarse, p, nullptr);

	// Create a source on the blocked lattice
	CoarseSpinor psi_in_coarse(info);
	restrictSpinorQDPXXFineToCoarse(my_blocks, vecs, psi_in, psi_in_coarse);


	CoarseSpinor coarse_solution(info);  // Solve into this
 	CoarseSpinor solution_check(info);   // multiply back into this.

	ZeroVec(coarse_solution);            // Initial guess

	// Solve
	CoarseFGMRES(coarse_solution, psi_in_coarse);

	// Multiply back to check the solution
	M_coarse(solution_check, coarse_solution, LINOP_OP);
	double diff_norm = XmyNorm2Vec(solution_check,psi_in_coarse)[0];
	double psi_norm = Norm2Vec(psi_in_coarse)[0];
	double rel_diff = sqrt(diff_norm/psi_norm);
	QDPIO::cout << "|| b - Ax ||=" << sqrt(diff_norm) <<std::endl;
	QDPIO::cout << "|| b - Ax ||/|| b ||=" <<rel_diff <<std::endl;
	ASSERT_LT( rel_diff, 5.0e-5);

}

int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

