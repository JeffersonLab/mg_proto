#include "gtest/gtest.h"
#include "../test_env.h"

#include <lattice/fine_qdpxx/invfgmres_qdpxx.h>

#include "../mock_nodeinfo.h"
#include "qdpxx_utils.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/coarse/coarse_types.h"

#include "qdpxx_utils.h"
#include "lattice/fine_qdpxx/mg_params_qdpxx.h"
#include "lattice/fine_qdpxx/mg_level_qdpxx.h"
#include "lattice/fine_qdpxx/vcycle_recursive_qdpxx.h"



using namespace MGTesting;
using namespace MG;
using namespace QDP;

// Test fixture to save me writing code all the time.
class EOBitsTesting : public ::testing::Test {
protected:

	// Define this at End of File so we can cut straight to the tests
	void SetUp() override;

	// Some protected variables
	IndexArray latdims;
	IndexArray node_orig;
	const float m_q=0.01;
	const float c_sw = 1.25;
	const int t_bc = -1;
	QDP::multi1d<QDP::LatticeColorMatrix> u;
	std::shared_ptr<QDPWilsonCloverLinearOperator> M;
	SetupParams level_setup_params = {
			2,       // Number of levels
			{8},   // Null vecs on L0, L1
			{
					{2,2,2,2}
			},
			{500},          // Max Nullspace Iters
			{5e-6},        // Nullspace Target Resid
			{false}
	};

	// Setup will set this up
	MultigridLevels mg_levels;

	// A one liner to access the coarse links generated
	CoarseGauge& getCoarseLinks() {
		return *(mg_levels.coarse_levels[0].gauge);
	}

	// A one liner to get at the coarse info
	const LatticeInfo& getCoarseInfo() const {
		return (mg_levels.coarse_levels[0].gauge)->GetInfo();
	}

	// Utility function
	void applyGammaSite(float *out, const float *in, int N_color) const
	{

#pragma omp simd aligned(out,in:64)
		for(int i=0; i < n_complex*N_color; ++i) {
			out[i] = in[i];
		}

#pragma omp simd aligned(out,in:64)
		for(int i=n_complex*N_color; i < 2*n_complex*N_color; ++i ) {
			out[i] = -in[i];
		}
	}

	void applyGamma(CoarseSpinor& out, const CoarseSpinor& in, const CBSubset& subset) const
	{
		const LatticeInfo info=in.GetInfo();
		const int n_cbsites = info.GetNumCBSites();
		const int N_color = info.GetNumColors();
		AssertCompatible(out.GetInfo(),info);

#pragma omp parallel for
		for(int cb=subset.start; cb < subset.end; ++cb) {
			for(int cbsite =0; cbsite < n_cbsites; ++cbsite) {
				applyGammaSite( out.GetSiteDataPtr(cb,cbsite), in.GetSiteDataPtr(cb,cbsite), N_color);
			}
		}


	}
};




#if 0
TEST_F(EOBitsTesting, TestOffDiagG5Herm)
{
	// Get the coarse links
	CoarseGauge& coarse_links = getCoarseLinks();

	// Get the Lattice info for the coarse lattice
	const LatticeInfo& coarse_info = getCoarseInfo();

	CoarseDiracOp D(coarse_info);

	CoarseSpinor x(coarse_info);
	CoarseSpinor gc_tmp1(coarse_info);
	CoarseSpinor gc_tmp2(coarse_info);

	CoarseSpinor y_dag(coarse_info);
	CoarseSpinor y_dag2(coarse_info);

	// cb is the target cb
	for(int cb=0; cb < n_checkerboard; ++cb) {
		int source_cb = 1-cb;

		Gaussian(x,RB[source_cb]);          // set x on the source cb.
		Gaussian(y_dag,RB[cb]);        // y_dag is the result - xpay, so give it an initial value

		// Compute y_dag2 = Gamma y_dag
		applyGamma(gc_tmp2, y_dag, RB[cb]);



		Gaussian(gc_tmp1,SUBSET_ALL);    // This is a temporary so set it to junk
		float alpha = -2.3;

		// this does  y + alpha D^\dagger x
#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			D.M_offDiag_xpay(y_dag,alpha, coarse_links,x,cb,LINOP_DAGGER,tid);
		}


		// This does  y + alpha Gamma_c D  Gamma_c x
		//
		// Step 1:  tmp1 = Gamma_c x
		applyGamma(gc_tmp1, x, RB[source_cb]);


		// Step 2:  = Gamma_c y_dag2 + alpha D (Gamma_c x)
		//
#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			D.M_offDiag_xpay(gc_tmp2,alpha,coarse_links,gc_tmp1,cb,LINOP_OP,tid);
		}

		// Now apply y_dag2 = Gamma_c gc_tmp2 =>
		//  y_dag2 = Gamma_c( Gamma_c y_dag + alpha D Gamma_c x)
		//         = y_dag + alpha Gamma D Gamma

		applyGamma(y_dag2,gc_tmp2, RB[cb]);

		double norm_diff = XmyNorm2Vec(y_dag,y_dag2,RB[cb]);
		MasterLog(INFO, "cb=%d NormDiff=%16.8e", cb, norm_diff);
		ASSERT_LT( norm_diff, 5.0e-10);
	}
}



TEST_F(EOBitsTesting, TestG5HermDiagInv)
{
	// Get the coarse links
	CoarseGauge& coarse_links = getCoarseLinks();

	// Get the Lattice info for the coarse lattice
	const LatticeInfo& coarse_info = getCoarseInfo();

	CoarseDiracOp D(coarse_info);

	CoarseSpinor x(coarse_info);
	CoarseSpinor gc_tmp1(coarse_info);
	CoarseSpinor gc_tmp2(coarse_info);

	CoarseSpinor y_dag(coarse_info);
	CoarseSpinor y_dag2(coarse_info);


	for(int cb=0; cb < n_checkerboard; ++cb) {
		Gaussian(x, RB[cb]);
#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			D.M_diagInv(y_dag,coarse_links,x,cb,LINOP_DAGGER,tid);
		}

		applyGamma(gc_tmp1, x, RB[cb]);
#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			D.M_diagInv(gc_tmp2,coarse_links,gc_tmp1,cb,LINOP_OP,tid);
		}
		applyGamma(y_dag2,gc_tmp2, RB[cb]);

		double norm_diff = XmyNorm2Vec(y_dag,y_dag2,RB[cb]);
		MasterLog(INFO, "cb=%d NormDiff=%16.8e", cb, norm_diff);
		ASSERT_LT( norm_diff, 5.0e-10);
	}
}

TEST_F(EOBitsTesting, TestG5HermDiag)
{
	// Get the coarse links
	CoarseGauge& coarse_links = getCoarseLinks();

	// Get the Lattice info for the coarse lattice
	const LatticeInfo& coarse_info = getCoarseInfo();

	CoarseDiracOp D(coarse_info);

	CoarseSpinor x(coarse_info);
	CoarseSpinor gc_tmp1(coarse_info);
	CoarseSpinor gc_tmp2(coarse_info);

	CoarseSpinor y_dag(coarse_info);
	CoarseSpinor y_dag2(coarse_info);


	for(int cb=0; cb < n_checkerboard; ++cb) {
		Gaussian(x, RB[cb]);
#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			D.M_diag(y_dag,coarse_links,x,cb,LINOP_DAGGER,tid);
		}

		applyGamma(gc_tmp1, x, RB[cb]);
#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			D.M_diag(gc_tmp2,coarse_links,gc_tmp1,cb,LINOP_OP,tid);
		}
		applyGamma(y_dag2,gc_tmp2, RB[cb]);

		double norm_diff = XmyNorm2Vec(y_dag,y_dag2,RB[cb]);
		MasterLog(INFO, "cb=%d NormDiff=%16.8e", cb, norm_diff);
		ASSERT_LT( norm_diff, 5.0e-10);
	}
}


TEST_F(EOBitsTesting, TestCoarseInvDiag)
{
	// Get the coarse links
	CoarseGauge& coarse_links = getCoarseLinks();

	// Get the Lattice info for the coarse lattice
	const LatticeInfo& coarse_info = getCoarseInfo();



	// Test: apply A^{-1} A spinor
	// Use our own Mat Mult library.
	CoarseSpinor spinor(coarse_info);
	CoarseSpinor diag_spinor(coarse_info);
	CoarseSpinor spinor_inv(coarse_info);

	Gaussian(spinor);
	ZeroVec(diag_spinor);
	ZeroVec(spinor_inv);

	int num_colorspins = coarse_info.GetNumColorSpins();
	int num_coarse_cbsites = coarse_info.GetNumCBSites();
	for(int cb=0; cb < n_checkerboard; ++cb) {
		for(int cbsite=0; cbsite < num_coarse_cbsites; ++cbsite) {

			// Get the original spinor
			float *vsite_in = spinor.GetSiteDataPtr(cb,cbsite);

			// This wil be: U x spinor
			float *diag_out = diag_spinor.GetSiteDataPtr(cb,cbsite);

			// Get the U
			float *u_diag = coarse_links.GetSiteDirDataPtr(cb,cbsite,8);

			// Multiply: diag_out = U spinor
			CMatMultNaive(diag_out,u_diag,vsite_in, num_colorspins);

			// Get the inverted U
			float *u_diag_inv = coarse_links.GetSiteDirEODataPtr(cb,cbsite,8);

			// This will be U^{-1} U spinor
			float *s_inv = spinor_inv.GetSiteDataPtr(cb,cbsite);

			// Multiply
			CMatMultNaive(s_inv,u_diag_inv,diag_out,num_colorspins);
		}
	}

	double norm_diff = XmyNorm2Vec(spinor_inv,spinor);
	MasterLog(INFO, "NormDiff=%16.8e\n", norm_diff);

}

TEST_F(EOBitsTesting, TestCoarse_M_diag_inv_diag)
{
	// Get the coarse links
	CoarseGauge& coarse_links = getCoarseLinks();

	// Get the Lattice info for the coarse lattice
	const LatticeInfo& coarse_info = getCoarseInfo();

	CoarseDiracOp D(coarse_info);


	MasterLog(INFO, "Testing M_invDiag M_diag == 1");
	// Test M_invDiag ( M_diag ) == 1
	//
	{
		CoarseSpinor spinor(coarse_info);
		CoarseSpinor diag_spinor(coarse_info);
		CoarseSpinor spinor_inv(coarse_info);

		Gaussian(spinor);
		ZeroVec(diag_spinor);
		ZeroVec(spinor_inv);

		for(int cb =0; cb < 2; ++cb) {
#pragma omp parallel
			{
				int tid=omp_get_thread_num();
				D.M_diag(diag_spinor, coarse_links, spinor, cb, LINOP_OP, tid);
#pragma omp barrier
				D.M_diagInv(spinor_inv, coarse_links, diag_spinor, cb, LINOP_OP, tid );
			}

			double norm_diff = XmyNorm2Vec(spinor_inv,spinor,RB[cb]);
			MasterLog(INFO, "cb=%d NormDiff=%16.8e", cb, norm_diff);
			ASSERT_LT( norm_diff, 1.0e-10);
		}
	}
}

TEST_F(EOBitsTesting, M_diag_x_M_invOffDiag_eq_M_OffDiag)
{
	// Get the coarse links
	CoarseGauge& coarse_links = getCoarseLinks();

	// Get the Lattice info for the coarse lattice
	const LatticeInfo& coarse_info = getCoarseInfo();

	CoarseDiracOp D(coarse_info);

	// Test M_Diag ( M_invOffDiag ) == OffDiag
	//
	MasterLog(INFO, "Testing M_diag M_invOffDiag == 0 + M_offDiag_xpay");
	{
		CoarseSpinor x(coarse_info);
		Gaussian(x);

		CoarseSpinor invOffx(coarse_info);
		CoarseSpinor mInvOffx(coarse_info);
		CoarseSpinor offx(coarse_info);


		for(int cb =0; cb < 2; ++cb) {
			ZeroVec(offx, RB[cb]); // Zero on target CB
			ZeroVec(invOffx,RB[cb]);
			ZeroVec(mInvOffx,RB[cb]);
			ZeroVec(offx);

#pragma omp parallel
			{
				int tid=omp_get_thread_num();
				D.M_invOffDiag(invOffx, coarse_links, x, cb, LINOP_OP, tid);
#pragma omp barrier
				D.M_diag(mInvOffx, coarse_links, invOffx, cb, LINOP_OP, tid );
#pragma omp barrier
				D.M_offDiag_xpay(offx, 1.0, coarse_links, x, cb, LINOP_OP, tid);
			}




			double norm_diff = XmyNorm2Vec(mInvOffx,offx,RB[cb]);
			MasterLog(INFO, "cb=%d NormDiff=%16.8e", cb, norm_diff);
			ASSERT_LT(norm_diff, 1.0e-9);
		}
	}
}

TEST_F(EOBitsTesting, Test3)
{
	// Get stuff from fixture
	CoarseGauge& coarse_links = getCoarseLinks();
	const LatticeInfo& coarse_info = getCoarseInfo();
	CoarseDiracOp D(coarse_info);

	MasterLog(INFO, "Testing M_invDiag (0 + M_OffDiag_xpay) = M_invOffDiag");
	// Test M_invOffDiag == M_invDiag M_OffDiag
	{
		CoarseSpinor x(coarse_info);
		Gaussian(x);

		CoarseSpinor invOffx(coarse_info);

		CoarseSpinor offx(coarse_info);
		CoarseSpinor mInvOffx(coarse_info);

		for(int cb =0; cb < 2; ++cb) {
			ZeroVec(offx, RB[cb]); // Zero on target CB
			ZeroVec(invOffx,RB[cb]);
			ZeroVec(mInvOffx,RB[cb]);
			ZeroVec(offx);

#pragma omp parallel
			{
				int tid=omp_get_thread_num();
				D.M_invOffDiag(invOffx, coarse_links, x, cb, LINOP_OP, tid);

#pragma omp barrier
				D.M_offDiag_xpay(offx, 1.0, coarse_links, x, cb, LINOP_OP, tid);

#pragma omp barrier
				D.M_diagInv(mInvOffx, coarse_links, offx, cb, LINOP_OP, tid );
			}


			double norm_diff = XmyNorm2Vec(mInvOffx,invOffx,RB[cb]);
			MasterLog(INFO, "cb=%d NormDiff=%16.8e", cb, norm_diff);
			ASSERT_LT(norm_diff, 1.0e-9);
		}
	}
}

TEST_F(EOBitsTesting, Test4)
{
	// Get stuff from fixture
	CoarseGauge& coarse_links = getCoarseLinks();
	const LatticeInfo& coarse_info = getCoarseInfo();
	CoarseDiracOp D(coarse_info);
	MasterLog(INFO, "Testing R^{-1}R = 1");
	{
		CoarseSpinor x(coarse_info);
		CoarseSpinor Rx(coarse_info);
		CoarseSpinor RinvRx(coarse_info);

		Gaussian(x);
		D.R_matrix(Rx,coarse_links,x,LINOP_OP);
		D.R_inv_matrix(RinvRx,coarse_links, Rx, LINOP_OP);
		double norm_diff = XmyNorm2Vec(x,RinvRx);
		MasterLog(INFO, "NormDiff=%16.8e",norm_diff);
		ASSERT_LT(norm_diff, 1.0e-9);

	}
}

TEST_F(EOBitsTesting, Test5)
{
	// Get stuff from fixture
	CoarseGauge& coarse_links = getCoarseLinks();
	const LatticeInfo& coarse_info = getCoarseInfo();
	CoarseDiracOp D(coarse_info);

	MasterLog(INFO, "Testing RR^{-1} = 1");
	{
		CoarseSpinor x(coarse_info);
		CoarseSpinor Rinvx(coarse_info);
		CoarseSpinor RRinvx(coarse_info);

		Gaussian(x);
		D.R_inv_matrix(Rinvx,coarse_links, x, LINOP_OP);
		D.R_matrix(RRinvx,coarse_links,Rinvx,LINOP_OP);

		double norm_diff = XmyNorm2Vec(x,RRinvx);
		MasterLog(INFO, "NormDiff=%16.8e",norm_diff);
		ASSERT_LT(norm_diff, 1.0e-9);

	}
}

TEST_F(EOBitsTesting, Test6)
{
	// Get stuff from fixture
	CoarseGauge& coarse_links = getCoarseLinks();
	const LatticeInfo& coarse_info = getCoarseInfo();
	CoarseDiracOp D(coarse_info);

	MasterLog(INFO, "Testing L^{-1}L = 1");
	{
		CoarseSpinor x(coarse_info);
		CoarseSpinor Lx(coarse_info);
		CoarseSpinor LinvLx(coarse_info);

		Gaussian(x);
		D.L_matrix(Lx,coarse_links,x,LINOP_OP);
		D.L_inv_matrix(LinvLx,coarse_links, Lx, LINOP_OP);
		for(int cb = 0; cb < 2; ++cb) {
			double norm_diff = XmyNorm2Vec(x,LinvLx,RB[cb]);
			MasterLog(INFO, "cb=%d NormDiff=%16.8e",cb,norm_diff);
			ASSERT_LT(norm_diff, 1.0e-9);
		}

	}
}

TEST_F(EOBitsTesting, Test7)
{
	// Get stuff from fixture
	CoarseGauge& coarse_links = getCoarseLinks();
	const LatticeInfo& coarse_info = getCoarseInfo();
	CoarseDiracOp D(coarse_info);

	MasterLog(INFO, "Testing L^{-1}L = 1");
	{
		CoarseSpinor x(coarse_info);
		CoarseSpinor Lx(coarse_info);
		CoarseSpinor LinvLx(coarse_info);

		Gaussian(x);
		D.L_matrix(Lx,coarse_links,x,LINOP_OP);
		D.L_inv_matrix(LinvLx,coarse_links, Lx, LINOP_OP);
		for(int cb = 0; cb < 2; ++cb) {
			double norm_diff = XmyNorm2Vec(x,LinvLx,RB[cb]);
			MasterLog(INFO, "cb=%d NormDiff=%16.8e",cb,norm_diff);
			ASSERT_LT(norm_diff, 1.0e-9);
		}

	}
}
TEST_F(EOBitsTesting, Test8)
{
	// Get stuff from fixture
	CoarseGauge& coarse_links = getCoarseLinks();
	const LatticeInfo& coarse_info = getCoarseInfo();
	CoarseDiracOp D(coarse_info);

	MasterLog(INFO, "Testing LL^{-1} = 1");
	{
		CoarseSpinor x(coarse_info);
		CoarseSpinor Linvx(coarse_info);
		CoarseSpinor LLinvx(coarse_info);

		Gaussian(x);
		D.L_inv_matrix(Linvx,coarse_links,x,LINOP_OP);
		D.L_matrix(LLinvx,coarse_links, Linvx, LINOP_OP);
		for(int cb = 0; cb < 2; ++cb) {
			double norm_diff = XmyNorm2Vec(x,LLinvx,RB[cb]);
			MasterLog(INFO, "cb=%d NormDiff=%16.8e",cb,norm_diff);
			ASSERT_LT(norm_diff, 1.0e-9);
		}

	}
}

TEST_F(EOBitsTesting, Test9)
{
	// Get stuff from fixture
	CoarseGauge& coarse_links = getCoarseLinks();
	const LatticeInfo& coarse_info = getCoarseInfo();
	CoarseDiracOp D(coarse_info);

	MasterLog(INFO, "Testing L D R  = Schur");
	{
		CoarseSpinor x(coarse_info);
		CoarseSpinor Rx(coarse_info);
		CoarseSpinor DRx(coarse_info);
		CoarseSpinor LDRx(coarse_info);
		CoarseSpinor Full(coarse_info);

		ZeroVec(Rx);
		ZeroVec(DRx);
		ZeroVec(LDRx);
		ZeroVec(Full);
		Gaussian(x);

		D.R_matrix(Rx,coarse_links, x, LINOP_OP);
		D.Schur_matrix(DRx, coarse_links,Rx, LINOP_OP);
		D.L_matrix(LDRx, coarse_links, DRx, LINOP_OP);

#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			for(int cb=0; cb < 2;++cb) {
				D.unprecOp(Full,coarse_links,x,cb, LINOP_OP,tid);
			}
		}

		for(int cb = 0; cb < 2; ++cb) {
			double norm_diff = XmyNorm2Vec(Full, LDRx,RB[cb]);
			MasterLog(INFO, "cb=%d NormDiff=%16.8e",cb,norm_diff);
			ASSERT_LT(norm_diff, 1.0e-8);
		}

	}
}

#endif

int main(int argc, char *argv[]) 
{
	::testing::InitGoogleTest(&argc, argv);
	::testing::AddGlobalTestEnvironment(new MGTesting::TestEnv(&argc,&argv));
	return RUN_ALL_TESTS();

}


// This is our test fixture. No tear down or setup
// Constructor. Set Up Only Once rather than setup/teardown
void EOBitsTesting::SetUp()  {
		latdims={{8,8,8,8}};
		initQDPXXLattice(latdims);
		QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

		IndexArray node_orig=NodeInfo().NodeCoords();
		for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

		// Init Gauge field
		u.resize(n_dim);
		for(int mu=0; mu < n_dim; ++mu) {
			gaussian(u[mu]);
			reunit(u[mu]);
		}

		QDPIO::cout << "Creating M" << std::endl;
		M=std::make_shared<QDPWilsonCloverLinearOperator>(m_q, c_sw, t_bc,u);

		QDPIO::cout << "Calling Level setup" << std::endl;
		SetupMGLevels(level_setup_params, mg_levels, M);
		QDPIO::cout << "Done" << std::endl;

		CoarseGauge& coarse_links = getCoarseLinks();
		invertCloverDiag(coarse_links);
		multInvClovOffDiaLeft(coarse_links);

		MasterLog(INFO, "mg_levels has %d levels", mg_levels.n_levels);
		{
			const LatticeInfo& fine_info = *(mg_levels.fine_level.info);
			const LatticeInfo& M_info = (*mg_levels.fine_level.M).GetInfo();
			const IndexArray fine_v = fine_info.GetLatticeDimensions();
			const IndexArray M_v= M_info.GetLatticeDimensions();
			const int fine_nc = fine_info.GetNumColors();
			const int fine_ns= fine_info.GetNumSpins();
			const int fine_M_nc = M_info.GetNumColors();
			const int fine_M_ns = M_info.GetNumSpins();
			const int num_vecs = mg_levels.fine_level.null_vecs.size();

			MasterLog(INFO, "Level 0 has: Volume=(%d,%d,%d,%d) Ns=%d Nc=%d M->getInfo() has volume=(%d,%d,%d,%d) Nc=%d Ns=%d num_null_vecs=%d",
					fine_v[0],fine_v[1],fine_v[2],fine_v[3],fine_ns,fine_nc, M_v[0],M_v[1],M_v[2],M_v[3], fine_M_nc,fine_M_ns, num_vecs);
		}
		for(int level=0; level < mg_levels.n_levels-1;++level) {
			const LatticeInfo& fine_info = *(mg_levels.coarse_levels[level].info);
			const LatticeInfo& M_info = (*mg_levels.coarse_levels[level].M).GetInfo();
			const int num_vecs = mg_levels.coarse_levels[level].null_vecs.size();
			const IndexArray fine_v = fine_info.GetLatticeDimensions();
			const IndexArray M_v= M_info.GetLatticeDimensions();
			const int fine_nc = fine_info.GetNumColors();
			const int fine_ns= fine_info.GetNumSpins();
			const int fine_M_nc = M_info.GetNumColors();
			const int fine_M_ns = M_info.GetNumSpins();
			MasterLog(INFO, "Level %d has: Volume=(%d,%d,%d,%d) Ns=%d Nc=%d  M->getInfo() has volume=(%d,%d,%d,%d) Nc=%d Ns=%d num_null_vecs=%d",
					level+1,fine_v[0],fine_v[1],fine_v[2],fine_v[3],fine_ns,fine_nc, M_v[0],M_v[1],M_v[2],M_v[3], fine_M_nc,fine_M_ns, num_vecs);
		}

	}

