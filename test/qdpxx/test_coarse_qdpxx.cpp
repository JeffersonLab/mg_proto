#include "gtest/gtest.h"
#include "../test_env.h"
#include "../mock_nodeinfo.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "qdpxx_helpers.h"
#include "reunit.h"
#include "transf.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_op.h"
#include "dslashm_w.h"

using namespace MG;
using namespace MGTesting;
using namespace QDP;

// Apply a single direction of Dslash
void DslashDir(LatticeFermion& out, const multi1d<LatticeColorMatrix>& u, const LatticeFermion& in, int dir)
{
	switch(dir) {
	case 0: // Dir 0, Forward
		out = spinReconstructDir0Minus(u[0] * shift(spinProjectDir0Minus(in), FORWARD, 0));
		break;
	case 1: // Dir 0, Backward
		out = spinReconstructDir0Plus(shift(adj(u[0]) * spinProjectDir0Plus(in), BACKWARD, 0));
		break;
	case 2: // Dir 1, Forward
		out = spinReconstructDir1Minus(u[1] * shift(spinProjectDir1Minus(in), FORWARD, 1));
		break;
	case 3: // Dir 1, Backward
		out = spinReconstructDir1Plus(shift(adj(u[1]) * spinProjectDir1Plus(in), BACKWARD, 1));
		break;
	case 4: // Dir 2, Forward
		out = spinReconstructDir2Minus(u[2] * shift(spinProjectDir2Minus(in), FORWARD, 2));
		break;
	case 5: // Dir 2, Backward
		out = spinReconstructDir2Plus(shift(adj(u[2]) * spinProjectDir2Plus(in), BACKWARD, 2));
		break;
	case 6: // Dir 3, Forward,
		out = spinReconstructDir3Minus(u[3] * shift(spinProjectDir3Minus(in), FORWARD, 3));
		break;
	case 7: // Dir 3, Backward
		out = spinReconstructDir3Plus(shift(adj(u[3]) * spinProjectDir3Plus(in), BACKWARD, 3));
		break;
	default:
		QDPIO::cerr<< "Unknown direction. You oughtnt call this" << std::endl;
		QDP_abort(1);
	}
}

//  prop_out(x) = prop_in^\dagger(x) ( 1 +/- gamma_mu ) U mu(x) prop_in(x + mu)
//
//  This routine is used to test Coarse Dslash
//  Initially, prop_in should be just I_{12x12}
//  However, any unitary matrix on the sites (orthonormal basis) would do
void dslashTripleProduct12x12SiteDir(int dir, const multi1d<LatticeColorMatrix>& u, const LatticePropagator& in_prop, LatticePropagator& out_prop)
{
	LatticeFermion in, out;
	LatticePropagator prop_tmp=zero;
	// Loop through spins and colors

	for(int spin=0; spin < 4; ++spin ) {
		for(int color=0; color < 3; ++color ) {

			// Extract component into 'in'
			PropToFerm(in_prop,in,color,spin);

			// Apply Dlsash in that Direction
			DslashDir(out, u, in, dir);

			// Place back into prop
			FermToProp(out, prop_tmp, color, spin);
		} // color
	} // spin

	// Technically I don't need this last part, since for this test
	// in_prop is the identity.
	//
	// However, if in_prop was some 12x12 unitary matrix (basis rotation)
	// This part would be necessary.
	// And actually that would constitute a useful test.
	out_prop=adj(in_prop)*prop_tmp;

}




TEST(TestInterface, TestQDPSpinorToCoarseSpinor)
{
	IndexArray latdims={{2,2,2,2}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	LatticeInfo info(latdims, 2, 6, NodeInfo());
	CoarseSpinor s_coarse(info);

	LatticeFermion in,out;
	gaussian(in);
	gaussian(out); // Make sure it is different from in

	QDPSpinorToCoarseSpinor(in,s_coarse);
	CoarseSpinorToQDPSpinor(s_coarse,out);

	LatticeFermion diff;
	diff = in -out;
	Double diff_norm = norm2(diff);
	Double rel_diff_norm = diff_norm/norm2(in);
	QDPIO::cout << "Diff Norm = " << sqrt(diff_norm) << std::endl;
	QDPIO::cout << "Relative Diff Norm = " << sqrt(rel_diff_norm) << std::endl;

}

TEST(TestInterface, TestQDPPropagatorToCoarsePropagator)
{
	IndexArray latdims={{2,2,2,2}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	LatticeInfo info(latdims, 2, 6, NodeInfo());
	CoarseGauge u_coarse(info);

	LatticePropagator in,out;

	for(int mu=0; mu < 2*n_dim; ++mu) {
		gaussian(in);
		gaussian(out); // Make sure it is different from in

		QDPPropToCoarseGaugeLink(in,u_coarse,mu);
		CoarseGaugeLinkToQDPProp(u_coarse,out,mu);

		LatticePropagator diff;
		diff = in -out;
		Double diff_norm = norm2(diff);
		Double rel_diff_norm = diff_norm/norm2(in);
		QDPIO::cout << "Dir: " << mu << " Diff Norm = " << sqrt(diff_norm) << std::endl;
		QDPIO::cout << "Dir: " << mu << " Relative Diff Norm = " << sqrt(rel_diff_norm) << std::endl;
	}
}



TEST(TestCoarseQDPXX, TestCoarseQDPXX)
{
	IndexArray latdims={{2,2,2,2}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	multi1d<LatticeColorMatrix> u(Nd);

	QDPIO::cout << "Generating Random Gauge with Gaussian Noise" << std::endl;
	for(int mu=0; mu < Nd; ++mu) {
//		u[mu] = 1;
		gaussian(u[mu]);
		reunit(u[mu]);
	}

	// Generate the 'vectors' of which there are to be 12. Funnily this fits nicely into a propagator
	// Later on would be better to have this be a general unitary matrix per site.
	QDPIO::cout << "Generating Eye" << std::endl;
	LatticePropagator eye=1;

	Double eye_norm = norm2(eye);
	Double eye_norm_per_site = eye_norm/Layout::vol();
	QDPIO::cout << "Eye Norm Per Site " << eye_norm_per_site << std::endl;


	multi1d<LatticePropagator> dslash_links(8);

	for(int mu=0; mu < 8; ++mu) {
		QDPIO::cout << "Attempting Triple Product in direction: " << mu << std::endl;
		dslashTripleProduct12x12SiteDir(mu, u, eye, dslash_links[mu]);
	}


	// Next step should be to copy this into the fields needed for gauge and clover ops
	LatticeInfo info(latdims, 2, 6, NodeInfo());
	CoarseGauge u_coarse(info);
	for(int mu=0; mu < 8; ++mu) {
		QDPPropToCoarseGaugeLink(dslash_links[mu],u_coarse, mu);
	}

	QDPIO::cout << "Coarse Gauge Field initialized " << std::endl;


	LatticeFermion psi, d_psi, m_psi;
	gaussian(psi);

	m_psi = zero;

	// Apply Dslash to both CBs, isign=1
	// Result in m_psiu
	dslash(m_psi, u, psi, 1, 0);
	dslash(m_psi, u, psi, 1, 1);

	// CoarsSpinors
	CoarseSpinor coarse_s_in(info);
	CoarseSpinor coarse_s_out(info);

	// Import psi
	QDPSpinorToCoarseSpinor(psi, coarse_s_in);


	// Create A coarse operator
	int n_smt = 1;
	CoarseDiracOp D_op_coarse(info, n_smt);

	// Apply Coarse Op Dslash in Threads
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		D_op_coarse.Dslash(coarse_s_out, u_coarse, coarse_s_in, 0, tid);
		D_op_coarse.Dslash(coarse_s_out, u_coarse, coarse_s_in, 1, tid);
	}

	// Export Coarse spinor to QDP++ spinors.
	LatticeFermion coarse_d_psi = zero;
	CoarseSpinorToQDPSpinor(coarse_s_out, coarse_d_psi);

	// Find the difference between regular dslash and 'coarse' dslash
	LatticeFermion diff = m_psi - coarse_d_psi;

	QDPIO::cout << "Norm Diff[0] = " << sqrt(norm2(diff, rb[0])) << std::endl;
	QDPIO::cout << "Norm Diff[1] = " << sqrt(norm2(diff, rb[1])) 	<< std::endl;
	QDPIO::cout << "Norm Diff = " << sqrt(norm2(diff)) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[0] = " << sqrt(norm2(diff, rb[0])/norm2(psi,rb[0])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[1] = " << sqrt(norm2(diff, rb[1])/norm2(psi,rb[1])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff = " << sqrt(norm2(diff)/norm2(psi)) << std::endl;

}


int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

