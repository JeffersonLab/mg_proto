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

using namespace MG;
using namespace MGTesting;
using namespace QDP;

TEST(TestLattice, CoarseLinOpRandomNullVecs)
{
	IndexArray latdims={{4,4,4,4}};
	IndexArray blockdims = {{2,2,2,2}};

	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	using FineSpinor = LatticeFermion;
	using FineGauge = multi1d<LatticeColorMatrix>;

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
		double diff_norm = xmyNorm2(fake_coarse_out,coarse_out);
		double coarse_out_norm = norm2(coarse_out);
		double rel_diff = sqrt(diff_norm/coarse_out_norm);
		QDPIO::cout << "|| coarse_out - fake_coarse_out ||=" << sqrt(diff_norm) <<std::endl;
		QDPIO::cout << "|| coarse_out - fake_coarse_out ||/||coarse_out||=" <<rel_diff <<std::endl;
		ASSERT_LT( rel_diff, 3.0e-7);
	}
}


int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

