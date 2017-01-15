#include "gtest/gtest.h"
#include "aggregate_qdpxx.h"
#include "../test_env.h"
#include "../mock_nodeinfo.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "qdpxx_helpers.h"
#include "reunit.h"
#include "transf.h"
#include "clover_fermact_params_w.h"
#include "clover_term_qdp_w.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_op.h"
#include "dslashm_w.h"
#include <complex>

using namespace MG;
using namespace MGTesting;
using namespace QDP;



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
	ASSERT_NEAR( toDouble(sqrt(diff_norm)), 0, 1.0e-6 );

	QDPIO::cout << "Relative Diff Norm = " << sqrt(rel_diff_norm) << std::endl;
	ASSERT_NEAR( toDouble(sqrt(rel_diff_norm)), 0, 1.0e-7);
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
		ASSERT_NEAR( toDouble(sqrt(diff_norm)), 0, 1.0e-5 );

		QDPIO::cout << "Dir: " << mu << " Relative Diff Norm = " << sqrt(rel_diff_norm) << std::endl;
		ASSERT_NEAR( toDouble(sqrt(rel_diff_norm)), 0, 1.0e-6);
	}
}



TEST(TestCoarseQDPXX, TestCoarseQDPXXDslash)
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
		dslashTripleProduct12x12SiteDirQDPXX(mu, u, eye, dslash_links[mu]);
	}


	// Next step should be to copy this into the fields needed for gauge and clover ops
	LatticeInfo info(latdims, 2, 6, NodeInfo());
	CoarseGauge u_coarse(info);
	for(int mu=0; mu < 8; ++mu) {
		QDPPropToCoarseGaugeLink(dslash_links[mu],u_coarse, mu);
	}

	QDPIO::cout << "Coarse Gauge Field initialized " << std::endl;

	for(int op = LINOP_OP; op <= LINOP_DAGGER; ++op ) {

		int isign = ( op == LINOP_OP ? +1 : -1 );

	LatticeFermion psi, d_psi, m_psi;
	gaussian(psi);

	m_psi = zero;

	// Apply Dslash to both CBs, isign=1
	// Result in m_psiu
	dslash(m_psi, u, psi, isign, 0);
	dslash(m_psi, u, psi, isign, 1);

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
		D_op_coarse.Dslash(coarse_s_out, u_coarse, coarse_s_in, 0, op, tid);
		D_op_coarse.Dslash(coarse_s_out, u_coarse, coarse_s_in, 1, op, tid);
	}

	// Export Coarse spinor to QDP++ spinors.
	LatticeFermion coarse_d_psi = zero;
	CoarseSpinorToQDPSpinor(coarse_s_out, coarse_d_psi);

	// Find the difference between regular dslash and 'coarse' dslash
	LatticeFermion diff = m_psi - coarse_d_psi;

	QDPIO::cout << "OP=" << op << std::endl;
	QDPIO::cout << "Norm Diff[0] = " << sqrt(norm2(diff, rb[0])) << std::endl;
	QDPIO::cout << "Norm Diff[1] = " << sqrt(norm2(diff, rb[1])) 	<< std::endl;
	QDPIO::cout << "Norm Diff = " << sqrt(norm2(diff)) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[0] = " << sqrt(norm2(diff, rb[0])/norm2(psi,rb[0])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[1] = " << sqrt(norm2(diff, rb[1])/norm2(psi,rb[1])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff = " << sqrt(norm2(diff)/norm2(psi)) << std::endl;


	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[0])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[1])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff)) ) , 0, 1.5e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[0])/norm2(psi,rb[0])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[1])/norm2(psi,rb[1])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff)/norm2(psi)) ), 0, 1.e-5 );
	}//op

}

TEST(TestCoarseQDPXX, TestCoarseQDPXXDslash2)
{
	IndexArray latdims={{2,2,2,2}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	multi1d<LatticeColorMatrix> u(Nd);

	QDPIO::cout << "Generating Random Gauge with Gaussian Noise" << std::endl;
	for(int mu=0; mu < Nd; ++mu) {
		//u[mu] = 1;
		gaussian(u[mu]);
		reunit(u[mu]);
	}

	multi1d<LatticeFermion> in_vecs(Nc*Ns/2);     // In terms of vectors
	multi1d<LatticePropagator> dslash_links(8); // In terms of propagators

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

	QDPIO::cout << "Printing in-vecs: " << std::endl;
	for(int spin=0; spin < 4; ++spin ) {
		for(int color=0; color < 3; ++color ) {
			for(int j=0; j < Nc*Ns/2; ++j) {
				printf("( %1.2lf, %1.2lf ) ", in_vecs[j].elem(0).elem(spin).elem(color).real(),
									in_vecs[j].elem(0).elem(spin).elem(color).imag() );
			}
			printf("\n");
		}
	}

	// Generate the Triple product into dslash_links[mu]
	for(int mu=0; mu < 8; ++mu) {
			MasterLog(INFO,"Attempting Triple Product in direction: %d \n", mu);
			dslashTripleProduct12x12SiteDirQDPXX(mu, u, eye, dslash_links[mu]);
	}



	// Next step should be to copy this into the fields needed for gauge and clover ops
	LatticeInfo info(latdims, 2, 6, NodeInfo());
	CoarseGauge u_coarse(info);


	// Generate the triple products directly into the u_coarse
	for(int mu=0; mu < 8; ++mu) {
		QDPIO::cout << " Attempting Triple Product in direction: " << mu << std::endl;
		dslashTripleProductSiteDirQDPXX(mu, u, in_vecs, u_coarse);
	}

	for(int row=0; row < Ns*Nc; ++row) {
		int spin_row = row/Nc;
		int color_row = row % Nc;

		for(int column=0; column < Nc*Ns; ++column) {
			int spin_column = column / Nc;
			int color_column = column % Nc;

			printf("( %1.2lf, %1.2lf ) ", dslash_links[0].elem(0).elem(spin_row,spin_column).elem(color_row,color_column).real(),
					dslash_links[0].elem(0).elem(spin_row, spin_column).elem(color_row,color_column).imag() );

		}
		printf("\n");
	}
	printf("\n");

	float *coarse_link = u_coarse.GetSiteDirDataPtr(0,0,0);

	for(int row=0; row < Ns*Nc; ++row) {

		for(int column=0; column < Nc*Ns; ++column) {

			int coarse_link_index = n_complex*(column + Ns*Nc*row);
			printf(" ( %1.2lf, %1.2lf ) ", coarse_link[ RE+coarse_link_index], coarse_link[ IM + coarse_link_index]);
		}
		printf("\n");
	}


	MasterLog(INFO,"Coarse Gauge Field initialized\n");

	for(int op = LINOP_OP; op <= LINOP_DAGGER; ++op) {

	int isign = (op == LINOP_OP ) ? +1 : -1 ;

	LatticeFermion psi, d_psi, m_psi;
	gaussian(psi);

	m_psi = zero;

	// Apply Dslash to both CBs, isign=1
	// Result in m_psiu
	dslash(m_psi, u, psi, isign, 0);
	dslash(m_psi, u, psi, isign, 1);

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
		D_op_coarse.Dslash(coarse_s_out, u_coarse, coarse_s_in, 0, op, tid);
		D_op_coarse.Dslash(coarse_s_out, u_coarse, coarse_s_in, 1, op, tid);
	}

	// Export Coarse spinor to QDP++ spinors.
	LatticeFermion coarse_d_psi = zero;
	CoarseSpinorToQDPSpinor(coarse_s_out, coarse_d_psi);

	// Find the difference between regular dslash and 'coarse' dslash
	LatticeFermion diff = m_psi - coarse_d_psi;

	QDPIO::cout << "OP = " << op << std::endl;
	QDPIO::cout << "Norm Diff[0] = " << sqrt(norm2(diff, rb[0])) << std::endl;
	QDPIO::cout << "Norm Diff[1] = " << sqrt(norm2(diff, rb[1])) 	<< std::endl;
	QDPIO::cout << "Norm Diff = " << sqrt(norm2(diff)) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[0] = " << sqrt(norm2(diff, rb[0])/norm2(psi,rb[0])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[1] = " << sqrt(norm2(diff, rb[1])/norm2(psi,rb[1])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff = " << sqrt(norm2(diff)/norm2(psi)) << std::endl;

	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[0])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[1])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff)) ) , 0, 1.5e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[0])/norm2(psi,rb[0])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[1])/norm2(psi,rb[1])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff)/norm2(psi)) ), 0, 1.e-5 );
	}
}

TEST(TestCoarseQDPXX, TestCoarseQDPXXClov)
{
	IndexArray latdims={{2,2,2,2}};
	initQDPXXLattice(latdims);
	MasterLog(INFO,"QDP++ Testcase Initialized\n");

	multi1d<LatticeColorMatrix> u(Nd);

	MasterLog(INFO,"Generating Random Gauge with Gaussian Noise\n");
	for(int mu=0; mu < Nd; ++mu) {
//		u[mu] = 1;
		gaussian(u[mu]);
		reunit(u[mu]);
	}

	// Now need to make a clover op
	 CloverFermActParams clparam;
	 AnisoParam_t aniso;

	  // Aniso prarams
	 aniso.anisoP=true;
	 aniso.xi_0 = 1.5;
	 aniso.nu = 0.95;
	 aniso.t_dir = 3;

	  // Set up the Clover params
	 clparam.anisoParam = aniso;

	  // Some mass
	 clparam.Mass = Real(0.1);

	 // Some random clover coeffs
	  clparam.clovCoeffR=Real(1.35);
	  clparam.clovCoeffT=Real(0.8);
	  QDPCloverTerm clov_qdp;
	  clov_qdp.create(u,clparam);



	// Generate the 'vectors' of which there are to be 12. Funnily this fits nicely into a propagator
	// Later on would be better to have this be a general unitary matrix per site.
	MasterLog(INFO,"Generating Eye\n");
	LatticePropagator eye=1;

	Double eye_norm = norm2(eye);
	Double eye_norm_per_site = eye_norm/Layout::vol();
	QDPIO::cout << "Eye Norm Per Site " << eye_norm_per_site << std::endl;

	LatticePropagator tprod_result;

	clovTripleProduct12cx12SiteQDPXX(clov_qdp, eye, tprod_result);

	QDPIO::cout << "Checking Triple product result PropClover is still block diagonal" << std::endl;

	for(int spin_row=0; spin_row < 2; ++spin_row) {
		for(int spin_col=2; spin_col < 4; ++spin_col) {
			for(int col_row = 0; col_row < 3; ++col_row ) {
				for(int col_col = 0; col_col < 3; ++col_col ) {
					float re = tprod_result.elem(0).elem(spin_row,spin_col).elem(col_row,col_col).real();
					float im = tprod_result.elem(0).elem(spin_row,spin_col).elem(col_row,col_col).imag();

					ASSERT_FLOAT_EQ(re,0);
					ASSERT_FLOAT_EQ(im,0);

				}
			}
		}
	}

	for(int spin_row=2; spin_row < 4; ++spin_row) {
		for(int spin_col=0; spin_col < 2; ++spin_col) {
			for(int col_row = 0; col_row < 3; ++col_row ) {
				for(int col_col = 0; col_col < 3; ++col_col ) {
					float re = tprod_result.elem(0).elem(spin_row,spin_col).elem(col_row,col_col).real();
					float im = tprod_result.elem(0).elem(spin_row,spin_col).elem(col_row,col_col).imag();

					ASSERT_FLOAT_EQ(re,0);
					ASSERT_FLOAT_EQ(im,0);

				}
			}
		}
	}


	for(int op = LINOP_OP; op <= LINOP_DAGGER; ++op ) {

	int isign = ( op == LINOP_OP ? +1 : -1 ) ;

	LatticeFermion orig;
	gaussian(orig);
	LatticeFermion orig_res=zero;


	// orig_res = A orig
	clov_qdp.apply(orig_res, orig, isign, 0);
	clov_qdp.apply(orig_res, orig, isign, 1);


	LatticeFermion diff = zero;
	{
		QDPIO::cout << "Checking Triple product result PropClover can be multiplied with Fermion" << std::endl;

		LatticeFermion tprod_res_ferm = zero;

		// Just multiply by propgatator
		tprod_res_ferm = tprod_result*orig;

		diff = tprod_res_ferm - orig_res;
		QDPIO::cout << "Diff Norm = " << sqrt(norm2(diff)) << std::endl;
	}


	QDPIO::cout << "Importing Triple product result PropClover into CoarseClover " << std::endl;
	LatticeInfo info(latdims, 2, 6, NodeInfo());
	CoarseClover c_clov(info);
	QDPPropToCoarseClover(tprod_result, c_clov);

	CoarseSpinor s_in(info);
	QDPSpinorToCoarseSpinor(orig,s_in);

	CoarseSpinor s_out(info);

	int n_smt = 1;
	CoarseDiracOp D(info,n_smt);

#pragma omp parallel
	{
		int tid = omp_get_thread_num();

		D.CloverApply(s_out, c_clov, s_in,0,op, tid);
		D.CloverApply(s_out, c_clov, s_in,1,op, tid);


	}

	LatticeFermion coarse_res;
	CoarseSpinorToQDPSpinor(s_out,coarse_res);
	diff = orig_res - coarse_res;

	QDPIO::cout << "OP=" << op << std::endl;
	QDPIO::cout << "Norm Diff[0] = " << sqrt(norm2(diff, rb[0])) << std::endl;
	QDPIO::cout << "Norm Diff[1] = " << sqrt(norm2(diff, rb[1])) 	<< std::endl;
	QDPIO::cout << "Norm Diff = " << sqrt(norm2(diff)) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[0] = " << sqrt(norm2(diff, rb[0])/norm2(orig,rb[0])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[1] = " << sqrt(norm2(diff, rb[1])/norm2(orig,rb[1])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff = " << sqrt(norm2(diff)/norm2(orig)) << std::endl;


	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[0])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[1])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff)) ) , 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[0])/norm2(orig,rb[0])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[1])/norm2(orig,rb[1])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff)/norm2(orig)) ), 0, 1.e-5 );
	}  // op

}

TEST(TestCoarseQDPXX, TestCoarseQDPXXClov2)
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

	// Now need to make a clover op
	 CloverFermActParams clparam;
	 AnisoParam_t aniso;

	  // Aniso prarams
	 aniso.anisoP=true;
	 aniso.xi_0 = 1.5;
	 aniso.nu = 0.95;
	 aniso.t_dir = 3;

	  // Set up the Clover params
	 clparam.anisoParam = aniso;

	  // Some mass
	 clparam.Mass = Real(0.1);

	 // Some random clover coeffs
	  clparam.clovCoeffR=Real(1.35);
	  clparam.clovCoeffT=Real(0.8);
	  QDPCloverTerm clov_qdp;
	  clov_qdp.create(u,clparam);



	// Generate the 'vectors' of which there are to be 12. Funnily this fits nicely into a propagator
	// Later on would be better to have this be a general unitary matrix per site.
	QDPIO::cout << "Generating Eye" << std::endl;
	LatticePropagator eye=1;

	Double eye_norm = norm2(eye);
	Double eye_norm_per_site = eye_norm/Layout::vol();
	QDPIO::cout << "Eye Norm Per Site " << eye_norm_per_site << std::endl;

	// Pack 'Eye' vectors into to the in_vecs;
	// This is similar to the case when we will have noise filled vectors
	multi1d<LatticeFermion> in_vecs(Nc*Ns/2);

	for(int spin=0; spin < Ns/2; ++spin) {
		for(int color =0; color < Nc; ++color) {
			LatticeFermion upper = zero;
			LatticeFermion lower = zero;

			PropToFerm(eye, lower, color, spin);
			PropToFerm(eye, upper, color, spin+Ns/2);
			in_vecs[color + Nc*spin] = upper + lower;
		}
	}


	LatticePropagator tprod_result;
	clovTripleProduct12cx12SiteQDPXX(clov_qdp, eye, tprod_result);
#if 0

	QDPIO::cout << "Checking Triple product result PropClover is still block diagonal" << std::endl;

	for(int spin_row=0; spin_row < 2; ++spin_row) {
		for(int spin_col=2; spin_col < 4; ++spin_col) {
			for(int col_row = 0; col_row < 3; ++col_row ) {
				for(int col_col = 0; col_col < 3; ++col_col ) {
					float re = tprod_result.elem(0).elem(spin_row,spin_col).elem(col_row,col_col).real();
					float im = tprod_result.elem(0).elem(spin_row,spin_col).elem(col_row,col_col).imag();

					ASSERT_FLOAT_EQ(re,0);
					ASSERT_FLOAT_EQ(im,0);

				}
			}
		}
	}

	for(int spin_row=2; spin_row < 4; ++spin_row) {
		for(int spin_col=0; spin_col < 2; ++spin_col) {
			for(int col_row = 0; col_row < 3; ++col_row ) {
				for(int col_col = 0; col_col < 3; ++col_col ) {
					float re = tprod_result.elem(0).elem(spin_row,spin_col).elem(col_row,col_col).real();
					float im = tprod_result.elem(0).elem(spin_row,spin_col).elem(col_row,col_col).imag();

					ASSERT_FLOAT_EQ(re,0);
					ASSERT_FLOAT_EQ(im,0);

				}
			}
		}
	}

#endif

	QDPIO::cout << "Importing Triple product result PropClover into CoarseClover " << std::endl;


	// Now test the new packer.
	LatticeInfo info(latdims, 2, 6, NodeInfo());
	CoarseClover c_clov(info);
	clovTripleProductSiteQDPXX(clov_qdp,in_vecs, c_clov);


	for(int op = LINOP_OP; op <= LINOP_DAGGER; ++op) {

		int isign = ( op == LINOP_OP ) ? +1 : -1;

	// Now create a LatticeFermion and apply both the QDP++ and the Coarse Clover
	LatticeFermion orig;
	gaussian(orig);
	LatticeFermion orig_res=zero;


	// orig_res = A orig
	for(int cb=0; cb < 2; ++cb) {
		clov_qdp.apply(orig_res, orig, isign, cb);
	}

	// Convert original spinor to a coarse spinor
	CoarseSpinor s_in(info);
	QDPSpinorToCoarseSpinor(orig,s_in);
	CoarseSpinor s_out(info);

	int n_smt = 1;
	CoarseDiracOp D(info,n_smt);

#pragma omp parallel
	{
		int tid = omp_get_thread_num();

		D.CloverApply(s_out, c_clov, s_in,0,op,tid);
		D.CloverApply(s_out, c_clov, s_in,1,op,tid);
	}

	LatticeFermion coarse_res;
	CoarseSpinorToQDPSpinor(s_out,coarse_res);

	LatticeFermion diff = orig_res - coarse_res;

	QDPIO::cout << "OP = " << op << std::endl;
	QDPIO::cout << "Norm Diff[0] = " << sqrt(norm2(diff, rb[0])) << std::endl;
	QDPIO::cout << "Norm Diff[1] = " << sqrt(norm2(diff, rb[1])) 	<< std::endl;
	QDPIO::cout << "Norm Diff = " << sqrt(norm2(diff)) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[0] = " << sqrt(norm2(diff, rb[0])/norm2(orig,rb[0])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[1] = " << sqrt(norm2(diff, rb[1])/norm2(orig,rb[1])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff = " << sqrt(norm2(diff)/norm2(orig)) << std::endl;

	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[0])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[1])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff)) ) , 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[0])/norm2(orig,rb[0])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[1])/norm2(orig,rb[1])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff)/norm2(orig)) ), 0, 1.e-5 );
	} // op


}

TEST(TestCoarseQDPXX, TestCoarseOrthonormalize)
{
	IndexArray latdims={{2,2,2,2}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	multi1d<LatticeFermion> vecs(6);
	for(int k=0; k < 6; ++k) {
		gaussian(vecs[k]);
	}

	// Someone once said doing this twice is good
	orthonormalizeAggregatesQDPXX(vecs);
	orthonormalizeAggregatesQDPXX(vecs);

	for(int site=all.start(); site <= all.end(); ++site) {
		for(int aggr=0; aggr < 2; ++aggr ) {
			// Check normalization:
			for(int curr_vec = 0; curr_vec < 6; ++curr_vec) {

				//
				for(int test_vec = 0; test_vec < 6; ++test_vec ) {

					if( test_vec != curr_vec ) {
						//	MasterLog(DEBUG, "Checking inner product of pair (%d,%d), site=%d aggr=%d\n", curr_vec,test_vec, site,aggr);
						std::complex<double> iprod = innerProductAggrQDPXX(vecs[test_vec],vecs[curr_vec], site, aggr);
						ASSERT_NEAR( real(iprod), 0, 1.0e-15);
						ASSERT_NEAR( imag(iprod), 0, 1.0e-15);

					}
					else {

						std::complex<double> iprod = innerProductAggrQDPXX(vecs[test_vec],vecs[curr_vec], site, aggr);
						ASSERT_NEAR( real(iprod), 1, 1.0e-15);
						ASSERT_NEAR( imag(iprod), 0, 1.0e-15);

						// 	MasterLog(DEBUG, "Checking norm2 of vector %d site=%d aggr=%d\n", curr_vec, site,aggr);
						double norm = sqrt(norm2AggrQDPXX(vecs[curr_vec],site,aggr));
						ASSERT_NEAR(norm, 1, 1.0e-15);

					}
				}
			}
		}
	}

}

TEST(TestCoarseQDPXX, TestRestrictorIdentity1)
{
	IndexArray latdims={{2,2,2,2}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	multi1d<LatticeColorMatrix> u(Nd);

	QDPIO::cout << "Generating Random Gauge with Gaussian Noise" << std::endl;
	for(int mu=0; mu < Nd; ++mu) {
		//u[mu] = 1;
		gaussian(u[mu]);
		reunit(u[mu]);
	}

	multi1d<LatticeFermion> in_vecs(Nc*Ns/2);     // In terms of vectors

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

	LatticeInfo info(latdims, 2, 6, NodeInfo());
	CoarseSpinor coarse(info);

	LatticeFermion fine_in;
	LatticeFermion fine_out;

	gaussian(fine_in);

	// Restrict -- this should be just like packing
	restrictSpinorQDPXXFineToCoarse(in_vecs, fine_in, coarse);

	// Unpack --
	CoarseSpinorToQDPSpinor(coarse,fine_out);

	for(int site=all.start(); site <= all.end(); ++site ) {
		for(int spin=0; spin < Ns; spin++) {
			for(int color=0; color < Nc; color++) {
				ASSERT_FLOAT_EQ(  fine_out.elem(site).elem(spin).elem(color).real(),
							fine_in.elem(site).elem(spin).elem(color).real());
				ASSERT_FLOAT_EQ(  fine_out.elem(site).elem(spin).elem(color).imag(),
											fine_in.elem(site).elem(spin).elem(color).imag());

			}
		}
	}


}

TEST(TestCoarseQDPXX, TestRestrictorIdentity2)
{
	IndexArray latdims={{2,2,2,2}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	multi1d<LatticeColorMatrix> u(Nd);

	QDPIO::cout << "Generating Random Gauge with Gaussian Noise" << std::endl;
	for(int mu=0; mu < Nd; ++mu) {
		//u[mu] = 1;
		gaussian(u[mu]);
		reunit(u[mu]);
	}

	multi1d<LatticeFermion> in_vecs(Nc*Ns/2);     // In terms of vectors

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

	LatticeInfo info(latdims, 2, 6, NodeInfo());
	CoarseSpinor coarse(info);

	LatticeFermion fine_in;
	LatticeFermion fine_out;

	gaussian(fine_in);

	// Restrict -- this should be just like packing
	QDPSpinorToCoarseSpinor(fine_in,coarse);

	prolongateSpinorCoarseToQDPXXFine(in_vecs, coarse,fine_out);


	for(int site=all.start(); site <= all.end(); ++site ) {
		for(int spin=0; spin < Ns; spin++) {
			for(int color=0; color < Nc; color++) {
				ASSERT_FLOAT_EQ(  fine_out.elem(site).elem(spin).elem(color).real(),
							fine_in.elem(site).elem(spin).elem(color).real());
				ASSERT_FLOAT_EQ(  fine_out.elem(site).elem(spin).elem(color).imag(),
											fine_in.elem(site).elem(spin).elem(color).imag());

			}
		}
	}


}


TEST(TestCoarseQDPXX, TestRestrictorIdentity3)
{
	IndexArray latdims={{2,2,2,2}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	multi1d<LatticeColorMatrix> u(Nd);

	QDPIO::cout << "Generating Random Gauge with Gaussian Noise" << std::endl;
	for(int mu=0; mu < Nd; ++mu) {
		//u[mu] = 1;
		gaussian(u[mu]);
		reunit(u[mu]);
	}

	multi1d<LatticeFermion> in_vecs(Nc*Ns/2);     // In terms of vectors

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

	LatticeInfo info(latdims, 2, 6, NodeInfo());
	CoarseSpinor coarse(info);

	LatticeFermion fine_in;
	LatticeFermion fine_out;

	gaussian(fine_in);

	// Restrict -- this should be just like packing
	restrictSpinorQDPXXFineToCoarse(in_vecs,fine_in,coarse);
	prolongateSpinorCoarseToQDPXXFine(in_vecs, coarse,fine_out);


	for(int site=all.start(); site <= all.end(); ++site ) {
		for(int spin=0; spin < Ns; spin++) {
			for(int color=0; color < Nc; color++) {
				ASSERT_FLOAT_EQ(  fine_out.elem(site).elem(spin).elem(color).real(),
							fine_in.elem(site).elem(spin).elem(color).real());
				ASSERT_FLOAT_EQ(  fine_out.elem(site).elem(spin).elem(color).imag(),
											fine_in.elem(site).elem(spin).elem(color).imag());

			}
		}
	}


}

TEST(TestCoarseQDPXX, TestRestrictorIdentity4)
{
	IndexArray latdims={{2,2,2,2}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	multi1d<LatticeColorMatrix> u(Nd);

	QDPIO::cout << "Generating Random Gauge with Gaussian Noise" << std::endl;
	for(int mu=0; mu < Nd; ++mu) {
		//u[mu] = 1;
		gaussian(u[mu]);
		reunit(u[mu]);
	}

	multi1d<LatticeFermion> vecs(6);
	for(int k=0; k < 6; ++k) {
		gaussian(vecs[k]);
	}

	// Someone once said doing this twice is good
	orthonormalizeAggregatesQDPXX(vecs);
	orthonormalizeAggregatesQDPXX(vecs);

	LatticeInfo info(latdims, 2, 6, NodeInfo());
	CoarseSpinor coarse(info);

	LatticeFermion fine_in;
	LatticeFermion fine_out;

	gaussian(fine_in);

	// Restrict -- this should be just like packing
	restrictSpinorQDPXXFineToCoarse(vecs,fine_in,coarse);
	prolongateSpinorCoarseToQDPXXFine(vecs, coarse,fine_out);


	for(int site=all.start(); site <= all.end(); ++site ) {
		for(int spin=0; spin < Ns; spin++) {
			for(int color=0; color < Nc; color++) {
				ASSERT_NEAR(  fine_out.elem(site).elem(spin).elem(color).real(),
							fine_in.elem(site).elem(spin).elem(color).real(), 1.0e-6);
				ASSERT_NEAR(  fine_out.elem(site).elem(spin).elem(color).imag(),
											fine_in.elem(site).elem(spin).elem(color).imag(), 1.0e-6);

			}
		}
	}


}


TEST(TestCoarseQDPXX, TestCoarseQDPXXClov3)
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

	// Now need to make a clover op
	CloverFermActParams clparam;
	AnisoParam_t aniso;

	// Aniso prarams
	aniso.anisoP=true;
	aniso.xi_0 = 1.5;
	aniso.nu = 0.95;
	aniso.t_dir = 3;

	// Set up the Clover params
	clparam.anisoParam = aniso;

	// Some mass
	clparam.Mass = Real(0.1);

	// Some random clover coeffs
	clparam.clovCoeffR=Real(1.35);
	clparam.clovCoeffT=Real(0.8);
	QDPCloverTerm clov_qdp;
	clov_qdp.create(u,clparam);

	multi1d<LatticeFermion> vecs(6);
	for(int k=0; k < 6; ++k) {
		gaussian(vecs[k]);
	}

	// Someone once said doing this twice is good
	orthonormalizeAggregatesQDPXX(vecs);
	orthonormalizeAggregatesQDPXX(vecs);

	QDPIO::cout << "Coarsening Clover" << std::endl;

	LatticeInfo info(latdims, 2, 6, NodeInfo());
	CoarseClover c_clov(info);
	clovTripleProductSiteQDPXX(clov_qdp, vecs, c_clov);

	for(int op = LINOP_OP; op <= LINOP_DAGGER; ++op ) {

		int isign = ( op == LINOP_OP ) ? +1 : -1 ;

	// Now create a LatticeFermion and apply both the QDP++ and the Coarse Clover
	LatticeFermion orig;
	gaussian(orig);
	LatticeFermion orig_res=zero;

	// Apply QDP++ clover
	for(int cb=0; cb < 2; ++cb) {
		clov_qdp.apply(orig_res, orig, isign, cb);
	}

	// Convert original spinor to a coarse spinor
	CoarseSpinor s_in(info);

	// Restrict using orthonormal basis
	restrictSpinorQDPXXFineToCoarse(vecs, orig, s_in);

	// Output
	CoarseSpinor s_out(info);

	int n_smt = 1;
	CoarseDiracOp D(info,n_smt);

	// Apply Coarsened Clover
#pragma omp parallel
	{
		int tid = omp_get_thread_num();

		D.CloverApply(s_out, c_clov, s_in,0, op, tid);
		D.CloverApply(s_out, c_clov, s_in,1, op, tid);
	}

	LatticeFermion coarse_res;
	prolongateSpinorCoarseToQDPXXFine(vecs, s_out, coarse_res);


	LatticeFermion diff = orig_res - coarse_res;

	QDPIO::cout << "OP = " << op << std::endl;

	QDPIO::cout << "Norm Diff[0] = " << sqrt(norm2(diff, rb[0])) << std::endl;
	QDPIO::cout << "Norm Diff[1] = " << sqrt(norm2(diff, rb[1])) 	<< std::endl;
	QDPIO::cout << "Norm Diff = " << sqrt(norm2(diff)) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[0] = " << sqrt(norm2(diff, rb[0])/norm2(orig,rb[0])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[1] = " << sqrt(norm2(diff, rb[1])/norm2(orig,rb[1])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff = " << sqrt(norm2(diff)/norm2(orig)) << std::endl;

	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[0])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[1])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff)) ) , 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[0])/norm2(orig,rb[0])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[1])/norm2(orig,rb[1])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff)/norm2(orig)) ), 0, 1.e-5 );
	}


}


TEST(TestCoarseQDPXX, TestCoarseQDPXXDslash3)
{
	IndexArray latdims={{2,2,2,2}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	multi1d<LatticeColorMatrix> u(Nd);

	QDPIO::cout << "Generating Random Gauge with Gaussian Noise" << std::endl;
	for(int mu=0; mu < Nd; ++mu) {
		//u[mu] = 1;
		gaussian(u[mu]);
		reunit(u[mu]);
	}


	// Random Basis vectors
	multi1d<LatticeFermion> vecs(6);
	for(int k=0; k < 6; ++k) {
		gaussian(vecs[k]);
	}

	// Someone once said doing this twice is good
	orthonormalizeAggregatesQDPXX(vecs);
	orthonormalizeAggregatesQDPXX(vecs);


	// Next step should be to copy this into the fields needed for gauge and clover ops
	LatticeInfo info(latdims, 2, 6, NodeInfo());
	CoarseGauge u_coarse(info);

	// Generate the triple products directly into the u_coarse
	for(int mu=0; mu < 8; ++mu) {
		QDPIO::cout << " Attempting Triple Product in direction: " << mu << std::endl;
		dslashTripleProductSiteDirQDPXX(mu, u, vecs, u_coarse);
	}

	for(int op = LINOP_OP; op <= LINOP_DAGGER; ++op ) {
		int isign = ( op == LINOP_OP ) ? +1 : -1;


	LatticeFermion psi, d_psi, m_psi;

	gaussian(psi);

	m_psi = zero;


	// Fine version:  m_psi_f =  D_f  psi_f
	// Apply Dslash to both CBs, isign=1
	// Result in m_psiu
	for(int cb=0; cb < 2; ++cb) {
		dslash(m_psi, u, psi, isign, cb);
	}

	// CoarsSpinors
	CoarseSpinor coarse_s_in(info);
	CoarseSpinor coarse_s_out(info);

	restrictSpinorQDPXXFineToCoarse(vecs, psi, coarse_s_in);


	// Create A coarse operator
	int n_smt = 1;
	CoarseDiracOp D_op_coarse(info, n_smt);

	// Apply Coarse Op Dslash in Threads
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		D_op_coarse.Dslash(coarse_s_out, u_coarse, coarse_s_in, 0, op, tid);
		D_op_coarse.Dslash(coarse_s_out, u_coarse, coarse_s_in, 1, op, tid);
	}

	// Export Coa            rse spinor to QDP++ spinors.
	LatticeFermion coarse_d_psi = zero;

	// Prolongate to form coarse_d_psi = P D_c R psi_f
	prolongateSpinorCoarseToQDPXXFine(vecs, coarse_s_out, coarse_d_psi);

	// Check   D_f psi_f = P D_c R psi_f
	LatticeFermion diff = m_psi - coarse_d_psi;
	QDPIO::cout << "OP = " << op << std::endl;
	QDPIO::cout << "Norm Diff[0] = " << sqrt(norm2(diff, rb[0])) << std::endl;
	QDPIO::cout << "Norm Diff[1] = " << sqrt(norm2(diff, rb[1])) 	<< std::endl;
	QDPIO::cout << "Norm Diff = " << sqrt(norm2(diff)) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[0] = " << sqrt(norm2(diff, rb[0])/norm2(psi,rb[0])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[1] = " << sqrt(norm2(diff, rb[1])/norm2(psi,rb[1])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff = " << sqrt(norm2(diff)/norm2(psi)) << std::endl;

	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[0])) ), 0, 5.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[1])) ), 0, 5.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff)) ) , 0, 5.e-5);
	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[0])/norm2(psi,rb[0])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff, rb[1])/norm2(psi,rb[1])) ), 0, 1.e-5 );
	ASSERT_NEAR( toDouble( sqrt(norm2(diff)/norm2(psi)) ), 0, 1.e-5 );
	} // op
}




int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

