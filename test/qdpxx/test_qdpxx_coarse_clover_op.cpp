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
#include "invfgmres_coarse.h"

using namespace MG;
using namespace MGTesting;
using namespace QDP;

TEST(TestLattice, CoarseLinOpRandomNullVecs)
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
	CreateBlockList(my_blocks,blocked_lattice_dims,latdims,blockdims);

	// Do the proper block orthogonalize
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);

	// Create the blocked Clover and Gauge Fields
	LatticeInfo info(blocked_lattice_dims, 2, 6, NodeInfo());
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
		double diff_norm = XmyNorm2Vec(fake_coarse_out,coarse_out);
		double coarse_out_norm = Norm2Vec(coarse_out);
		double rel_diff = sqrt(diff_norm/coarse_out_norm);
		QDPIO::cout << "|| coarse_out - fake_coarse_out ||=" << sqrt(diff_norm) <<std::endl;
		QDPIO::cout << "|| coarse_out - fake_coarse_out ||/||coarse_out||=" <<rel_diff <<std::endl;
		ASSERT_LT( rel_diff, 3.0e-7);
	}
}

TEST(TestLattice, CoarseLinOpFGMRESInvTrivial)
{
	IndexArray latdims={{4,4,4,4}};
	IndexArray blockdims = {{1,1,1,1}};

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
	CreateBlockList(my_blocks,blocked_lattice_dims,latdims,blockdims);

	// Do the proper block orthogonalize
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);

	// Create the blocked Clover and Gauge Fields
	LatticeInfo info(blocked_lattice_dims, 2, 6, NodeInfo());
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

	double diff_norm = XmyNorm2Vec(solution_check,psi_in_coarse);
		double psi_norm = Norm2Vec(psi_in_coarse);
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
	CreateBlockList(my_blocks,blocked_lattice_dims,latdims,blockdims);

	// Do the proper block orthogonalize
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);

	// Create the blocked Clover and Gauge Fields
	LatticeInfo info(blocked_lattice_dims, 2, 6, NodeInfo());
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
	double diff_norm = XmyNorm2Vec(solution_check,psi_in_coarse);
	double psi_norm = Norm2Vec(psi_in_coarse);
	double rel_diff = sqrt(diff_norm/psi_norm);
	QDPIO::cout << "|| b - Ax ||=" << sqrt(diff_norm) <<std::endl;
	QDPIO::cout << "|| b - Ax ||/|| b ||=" <<rel_diff <<std::endl;
	ASSERT_LT( rel_diff, 5.0e-5);

}

int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

