#include "gtest/gtest.h"
#include "../../test_env.h"
#include "../qdpxx_utils.h"

#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/nodeinfo.h"

#include "utils/print_utils.h"
#include "lattice/geometry_utils.h"

#include "lattice/spinor_halo.h"
#include "lattice/coarse/coarse_op.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_l1_blas.h"

#include "lattice/fine_qdpxx/qdpxx_helpers.h"
#include "lattice/fine_qdpxx/dslashm_w.h"
#include "lattice/fine_qdpxx/transf.h"
#include "lattice/fine_qdpxx/aggregate_block_qdpxx.h"
#include "lattice/fine_qdpxx/aggregate_qdpxx.h"

#include <vector>
#include <random>
using namespace QDP;
using namespace MG; 
using namespace MGTesting;

TEST(TestParallelCoarseDslash, TestDiracOp)
{
	// Check the Halo is initialized properly in a coarse Dirac Op
	IndexArray latdims={{6,4,4,4}};
	NodeInfo node;
	LatticeInfo info(latdims,2,6,node);
	initQDPXXLattice(latdims);

	multi1d<LatticeColorMatrix> u(n_dim);
	for(int mu=0; mu < n_dim; ++mu) {
		gaussian(u[mu]);
		reunit(u[mu]);

	}



	multi1d<LatticeFermion> in_vecs(Nc*Ns/2);     // In terms of vectors
#if 0
	MasterLog(INFO,"Generating Eye\n");
	LatticePropagator eye=1;


	// Pack 'Eye' vectors into to the in_vecs;
	for(int spin=0; spin < Ns/2; ++spin) {
		for(int color =0; color < Nc; ++color) {
			LatticeFermion upper = zero;
			LatticeFermion lower = zero;

			PropToFerm(eye, lower, color, spin);
			PropToFerm(eye, upper, color, spin+Ns/2);
			in_vecs[color + Nc*spin] = upper + lower;
		}
	}
#else
		for(int j=0; j < 6; ++j) {
			gaussian(in_vecs[j]);
		}

#endif
	// Trivial blocking
	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	IndexArray blocked_lattice_orig;
	IndexArray blockdims={{1,1,1,1}};
	CreateBlockList(my_blocks,blocked_lattice_dims, blocked_lattice_orig, latdims,blockdims, info.GetLatticeOrigin());


	// Do the proper block orthogonalize
	orthonormalizeBlockAggregatesQDPXX(in_vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(in_vecs, my_blocks);

	// Next step should be to copy this into the fields needed for gauge and clover ops
	LatticeInfo blocked_info(blocked_lattice_orig,blocked_lattice_dims,2,6,node);
	CoarseGauge u_coarse(info);
	ZeroGauge(u_coarse);

	// Generate the triple products directly into the u_coarse
	for(int mu=0; mu < 8; ++mu) {
		QDPIO::cout << " Attempting Triple Product in direction: " << mu << std::endl;
		dslashTripleProductDirQDPXX(my_blocks, mu, u, in_vecs, u_coarse);
	}

	LatticeFermion psi, m_psi;
	gaussian(psi);



	for(int op=LINOP_OP; op <=LINOP_DAGGER; op++) {
		m_psi = zero;

		int isign = ( op == LINOP_OP ) ? 1 : -1;

		// Fine version:  m_psi_f =  D_f  psi_f
		// Apply Dslash to both CBs, isign=1
		// Result in m_psiu
		for(int cb=0; cb < 2; ++cb) {
			dslash(m_psi, u, psi, isign, cb);
		}

	// CoarsSpinors
	CoarseSpinor coarse_s_in(info);
	CoarseSpinor coarse_s_out(info);

	restrictSpinorQDPXXFineToCoarse(my_blocks,in_vecs, psi, coarse_s_in);


	// Create A coarse operator
	int n_smt = 1;
	CoarseDiracOp D_op_coarse(info, n_smt);

	// Apply Coarse Op Dslash in Threads
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		D_op_coarse(coarse_s_out, u_coarse, coarse_s_in, 0, op, tid);
		D_op_coarse(coarse_s_out, u_coarse, coarse_s_in, 1, op, tid);
	}

	// Export Coa            rse spinor to QDP++ spinors.
	LatticeFermion coarse_d_psi = zero;

	// Prolongate to form coarse_d_psi = P D_c R psi_f
	prolongateSpinorCoarseToQDPXXFine(my_blocks,in_vecs, coarse_s_out, coarse_d_psi);

	// Check   D_f psi_f = P D_c R psi_f
	LatticeFermion diff = m_psi - coarse_d_psi;

	QDPIO::cout << "OP=" << op << std::endl;
	QDPIO::cout << "Norm Diff[0] = " << sqrt(norm2(diff, rb[0])) << std::endl;
	QDPIO::cout << "Norm Diff[1] = " << sqrt(norm2(diff, rb[1])) << std::endl;
	QDPIO::cout << "Norm Diff = " << sqrt(norm2(diff)) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[0] = " << sqrt(norm2(diff, rb[0])/norm2(psi,rb[0])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[1] = " << sqrt(norm2(diff, rb[1])/norm2(psi,rb[1])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff = " << sqrt(norm2(diff)/norm2(psi)) << std::endl;


	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[0])/norm2(psi,rb[0])) ), 0, 1.e-6 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[1])/norm2(psi,rb[1])) ), 0, 1.e-6 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff)/norm2(psi)) ), 0, 1.e-6 );
	} // op

}

TEST(TestParallelCoarseDslash, TestDslashDir)
{
	// Check the Halo is initialized properly in a coarse Dirac Op
	IndexArray latdims={{6,4,4,4}};
	NodeInfo node;
	LatticeInfo info(latdims,2,6,node);
	initQDPXXLattice(latdims);

	multi1d<LatticeColorMatrix> u(n_dim);
	for(int mu=0; mu < n_dim; ++mu) {
		gaussian(u[mu]);
		reunit(u[mu]);

	}



	multi1d<LatticeFermion> in_vecs(Nc*Ns/2);     // In terms of vectors
#if 0
	MasterLog(INFO,"Generating Eye\n");
	LatticePropagator eye=1;


	// Pack 'Eye' vectors into to the in_vecs;
	for(int spin=0; spin < Ns/2; ++spin) {
		for(int color =0; color < Nc; ++color) {
			LatticeFermion upper = zero;
			LatticeFermion lower = zero;

			PropToFerm(eye, lower, color, spin);
			PropToFerm(eye, upper, color, spin+Ns/2);
			in_vecs[color + Nc*spin] = upper + lower;
		}
	}
#else
		for(int j=0; j < 6; ++j) {
			gaussian(in_vecs[j]);
		}

#endif
	// Trivial blocking
	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	IndexArray blocked_lattice_orig;
	IndexArray blockdims={{1,1,1,1}};
	CreateBlockList(my_blocks,blocked_lattice_dims, blocked_lattice_orig, latdims,blockdims, info.GetLatticeOrigin());


	// Do the proper block orthogonalize
	orthonormalizeBlockAggregatesQDPXX(in_vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(in_vecs, my_blocks);

	// Next step should be to copy this into the fields needed for gauge and clover ops
	LatticeInfo blocked_info(blocked_lattice_orig,blocked_lattice_dims,2,6,node);
	CoarseGauge u_coarse(blocked_info);
	ZeroGauge(u_coarse);

	// Generate the triple products directly into the u_coarse
	for(int mu=0; mu < 8; ++mu) {
		QDPIO::cout << " Attempting Triple Product in direction: " << mu << std::endl;
		dslashTripleProductDirQDPXX(my_blocks, mu, u, in_vecs, u_coarse);
	}

	LatticeFermion psi, m_psi;
	gaussian(psi);

	CoarseSpinor coarse_s_in(blocked_info);
	CoarseSpinor coarse_s_out(blocked_info);
	// Create A coarse operator
	int n_smt = 1;
	CoarseDiracOp D_op_coarse(info, n_smt);

	for(int mu=0; mu < 2*n_dim; ++mu) {
		m_psi = zero;
		ZeroVec(coarse_s_out);
		// Do dslash Dir in both checkerboards
		DslashDirQDPXX(m_psi,u,psi,mu);

		// Initialize coarse_s_in
		restrictSpinorQDPXXFineToCoarse(my_blocks,in_vecs, psi, coarse_s_in);

		// Apply Coarse Op Dslash in Threads
#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			D_op_coarse.DslashDir(coarse_s_out,u_coarse,coarse_s_in,0,mu,tid);
			D_op_coarse.DslashDir(coarse_s_out,u_coarse,coarse_s_in,1,mu,tid);
		}

		// Export Coa            rse spinor to QDP++ spinors.
		LatticeFermion coarse_d_psi = zero;

		// Prolongate to form coarse_d_psi = P D_c R psi_f
		prolongateSpinorCoarseToQDPXXFine(my_blocks,in_vecs, coarse_s_out, coarse_d_psi);

		// Check   D_f psi_f = P D_c R psi_f
		LatticeFermion diff = m_psi - coarse_d_psi;

		QDPIO::cout << "dir="<<mu << " Norm Diff[0] = " << sqrt(norm2(diff, rb[0])) << std::endl;
		QDPIO::cout << "dir="<<mu << " Norm Diff[1] = " << sqrt(norm2(diff, rb[1])) << std::endl;
		QDPIO::cout << "dir="<<mu << " Norm Diff = " << sqrt(norm2(diff)) << std::endl;
		QDPIO::cout << "dir="<<mu << " Rel. Norm Diff[0] = " << sqrt(norm2(diff, rb[0])/norm2(psi,rb[0])) << std::endl;
		QDPIO::cout << "dir="<<mu << " Rel. Norm Diff[1] = " << sqrt(norm2(diff, rb[1])/norm2(psi,rb[1])) << std::endl;
		QDPIO::cout << "dir="<<mu << " Rel. Norm Diff = " << sqrt(norm2(diff)/norm2(psi)) << std::endl;


		ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[0])/norm2(psi,rb[0])) ), 0, 1.e-6 );
		ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[1])/norm2(psi,rb[1])) ), 0, 1.e-6 );
		ASSERT_NEAR( toDouble( sqrt(norm2(diff)/norm2(psi)) ), 0, 1.e-6 );

	} //mu

}




int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

