#include "gtest/gtest.h"
#include "../test_env.h"

#include <lattice/fine_qdpxx/invfgmres_qdpxx.h>

#include "lattice/nodeinfo.h"
#include "qdpxx_utils.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/coarse/coarse_types.h"

#include "qdpxx_utils.h"
#include "lattice/fine_qdpxx/mg_params_qdpxx.h"
#include "lattice/fine_qdpxx/mg_level_qdpxx.h"
#include "lattice/fine_qdpxx/vcycle_recursive_qdpxx.h"


#include "lattice/halo.h"
// Eigen Dense header
#include <Eigen/Dense>
using namespace Eigen;

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
				applyGammaSite( out.GetSiteDataPtr(0,cb,cbsite), in.GetSiteDataPtr(0,cb,cbsite), N_color);
			}
		}


	}
};


template<typename T>
struct MuLinkAccessor
{
	  inline
	  static
	  const float* get(const T& in, int cb, int cbsite, int dir, int fb);
};
template<typename T>
struct MuLinkADAccessor
{
	  inline
	  static
	  const float* get(const T& in, int cb, int cbsite, int dir, int fb);
};

template<>
inline
const float*
MuLinkAccessor<CoarseGauge>::get(const CoarseGauge& in, int cb, int cbsite, int dir, int fb)
{
	/* Unfortunately the link ordering is
	 * XPLUS,XMINUS, YPLUS, YMINUS, ZPLUS, ZMINUS
	 *
	 * but MG_FORWARD enum is 1 and MG_BACKWARD enum is 0, so
	 * hence back link (needed when 'fb' Is MG_FORWARD is 2*(XDIR) + 1 = 1 = XMINUS,
	 * and forward link (ndeeed when 'fb' is MG_WACKWARDS is 2*XDIR + 0 = 0 = XPLUS)
	 * it seems cack handed I know but it works.
	 */


    int mu;
    if ( dir == X_DIR && fb == MG_FORWARD) mu = 0;
    if ( dir == X_DIR && fb == MG_BACKWARD) mu = 1;

    if ( dir == Y_DIR && fb == MG_FORWARD) mu = 2;
    if ( dir == Y_DIR && fb == MG_BACKWARD) mu = 3;

    if ( dir == Z_DIR && fb == MG_FORWARD) mu = 4;
    if ( dir == Z_DIR && fb == MG_BACKWARD) mu = 5;

    if ( dir == T_DIR && fb == MG_FORWARD) mu = 6;
    if ( dir == T_DIR && fb == MG_BACKWARD) mu = 7;

    return in.GetSiteDirDataPtr(cb,cbsite,mu);
}

template<>
inline
const float*
MuLinkADAccessor<CoarseGauge>::get(const CoarseGauge& in, int cb, int cbsite, int dir, int fb)
{
	/* Unfortunately the link ordering is
	 * XPLUS,XMINUS, YPLUS, YMINUS, ZPLUS, ZMINUS
	 *
	 * but MG_FORWARD enum is 1 and MG_BACKWARD enum is 0, so
	 * hence back link (needed when 'fb' Is MG_FORWARD is 2*(XDIR) + 1 = 1 = XMINUS,
	 * and forward link (ndeeed when 'fb' is MG_WACKWARDS is 2*XDIR + 0 = 0 = XPLUS)
	 * it seems cack handed I know but it works.
	 */


    int mu;
    if ( dir == X_DIR && fb == MG_FORWARD) mu = 0;
    if ( dir == X_DIR && fb == MG_BACKWARD) mu = 1;

    if ( dir == Y_DIR && fb == MG_FORWARD) mu = 2;
    if ( dir == Y_DIR && fb == MG_BACKWARD) mu = 3;

    if ( dir == Z_DIR && fb == MG_FORWARD) mu = 4;
    if ( dir == Z_DIR && fb == MG_BACKWARD) mu = 5;

    if ( dir == T_DIR && fb == MG_FORWARD) mu = 6;
    if ( dir == T_DIR && fb == MG_BACKWARD) mu = 7;

    return in.GetSiteDirADDataPtr(cb,cbsite,mu);
}

template<typename T>
struct BackLinkAccessor
{
	  inline
	  static
	  const float* get(const T& in, int cb, int cbsite, int dir, int fb);
};

template<>
inline
const float*
BackLinkAccessor<CoarseGauge>::get(const CoarseGauge& in, int cb, int cbsite, int dir, int fb)
{
	  int mu;
	  if ( dir == X_DIR && fb == MG_FORWARD) mu = 1;
	  if ( dir == X_DIR && fb == MG_BACKWARD) mu = 0;

	  if ( dir == Y_DIR && fb == MG_FORWARD) mu = 3;
	  if ( dir == Y_DIR && fb == MG_BACKWARD) mu = 2;

	  if ( dir == Z_DIR && fb == MG_FORWARD) mu = 5;
	  if ( dir == Z_DIR && fb == MG_BACKWARD) mu = 4;

	  if ( dir == T_DIR && fb == MG_FORWARD) mu = 7;
	  if ( dir == T_DIR && fb == MG_BACKWARD) mu = 6;
	  return in.GetSiteDirDataPtr(cb,cbsite,mu);
}

template<typename T>
struct BackLinkADAccessor
{
	  inline
	  static
	  const float* get(const T& in, int cb, int cbsite, int dir, int fb);
};

template<>
inline
const float*
BackLinkADAccessor<CoarseGauge>::get(const CoarseGauge& in, int cb, int cbsite, int dir, int fb)
{
	// If asking for the worward direction, get the backward link.
	/* Unfortunately the link ordering is
		 * XPLUS,XMINUS, YPLUS, YMINUS, ZPLUS, ZMINUS
		 *
		 * but MG_FORWARD enum is 1 and MG_BACKWARD enum is 0, so
		 * hence back link (needed when 'fb' Is MG_FORWARD is 2*(XDIR) + 1 = 1 = XMINUS,
		 * and forward link (ndeeed when 'fb' is MG_WACKWARDS is 2*XDIR + 0 = 0 = XPLUS)
		 * it seems cack handed I know but it works.
		 */

	  int mu;
	  if ( dir == X_DIR && fb == MG_FORWARD) mu = 1;
	  if ( dir == X_DIR && fb == MG_BACKWARD) mu = 0;

	  if ( dir == Y_DIR && fb == MG_FORWARD) mu = 3;
	  if ( dir == Y_DIR && fb == MG_BACKWARD) mu = 2;

	  if ( dir == Z_DIR && fb == MG_FORWARD) mu = 5;
	  if ( dir == Z_DIR && fb == MG_BACKWARD) mu = 4;

	  if ( dir == T_DIR && fb == MG_FORWARD) mu = 7;
	  if ( dir == T_DIR && fb == MG_BACKWARD) mu = 6;

	  return in.GetSiteDirADDataPtr(cb,cbsite,mu);
}

// Eigen matrix wrappers for easy multiplication by Gamma_c (DiagonalMatrix
// obteined from EigenCDiag vector... usign the asDiagonal() method.

using EigenCMat = Matrix<std::complex<float>, Dynamic, Dynamic, ColMajor>;
using EigenCDiag = Matrix<std::complex<float>, Dynamic,1,ColMajor>;

TEST_F(EOBitsTesting, TestG5HermLinks)
{
	CoarseGauge& coarse_links = getCoarseLinks();
	const LatticeInfo& info = getCoarseInfo();

	/* Exchange Halos */
	HaloContainer<CoarseGauge> gauge_halo_cb0(info);
	HaloContainer<CoarseGauge> gauge_halo_cb1(info);
	HaloContainer<CoarseGauge>* halos[2] = { &gauge_halo_cb0, &gauge_halo_cb1 };

	// Communicate halos with BackLink accessors
	for(int target_cb=0; target_cb < 2; ++target_cb) {
		// Communicate MuLink, access with Back Link
		CommunicateHaloSync<CoarseGauge,MuLinkAccessor>(*(halos[target_cb]), coarse_links, target_cb);
	}

	// Make the Gamma_c matrix in Eigen
	const int num_colorspins = info.GetNumColorSpins();
	EigenCDiag g_c_vec(num_colorspins);
	for(int cspin=0; cspin < num_colorspins/2; cspin++) {
		g_c_vec(cspin)=std::complex<float>(1.0,0.0);
		g_c_vec(num_colorspins/2+cspin)=std::complex<float>(-1.0,0);
	}
	auto G_c = g_c_vec.asDiagonal();  // G_c is an 'Eigen' diagonal matrix for Gamma_c

	const int num_cbsites = info.GetNumCBSites();

	// For all directions
	MasterLog(INFO, "Check G_c D G_c = D^dagger");
	for(int mu_forw=0; mu_forw < 8; ++mu_forw) {
		MasterLog(INFO,"Doing mu=%d", mu_forw);

		// For all sites
		for(int cb=0; cb < 2; ++cb) {
			for(int cbsite=0; cbsite < num_cbsites; ++cbsite) {

				// Grab my link
				const float* my_link = coarse_links.GetSiteDirDataPtr(cb,cbsite,mu_forw);

				// Get the back link in mu direction.  Use 'GetNeighborDir' with BackLink accessor
				const float* back_link = GetNeighborDir<CoarseGauge,BackLinkAccessor>(*(halos[cb]),
						coarse_links, mu_forw, cb, cbsite);

				// Wrap links up as eigen matrix wrappers
				Eigen::Map< const EigenCMat > u_forw(reinterpret_cast<const std::complex<float>*>(my_link),
						num_colorspins,
						num_colorspins);
				Eigen::Map< const EigenCMat > u_back(reinterpret_cast<const std::complex<float>*>(back_link),
						num_colorspins,
						num_colorspins);

				// Generate G_c u_forw G_c, should equal u_back^\dagger
				EigenCMat gc_u_forw_gc = G_c * u_forw * G_c;

				// Subtract the u_back^\dagger
				gc_u_forw_gc -= u_back.adjoint();

				// Check result is zero
				for(int col=0; col < num_colorspins; ++col) {
					for(int row=0; row < num_colorspins; ++row) {

						ASSERT_LT( fabs((gc_u_forw_gc(col,row)).real()), 5.0e-7);
						ASSERT_LT( fabs((gc_u_forw_gc(col,row)).imag()), 5.0e-7);

					}
				}

			} // cbsites

		} // cb
	} //  dir

	// Now check link of clover term
	MasterLog(INFO, "Checking Clover Term");
	for(int cb=0; cb < 2; ++cb) {
		for(int cbsite=0; cbsite < num_cbsites; ++cbsite) {

			// Grab my link
			const float* my_link = coarse_links.GetSiteDiagDataPtr(cb,cbsite);


			// Wrap links up as eigen matrix wrappers
			Eigen::Map< const EigenCMat > u_clov(reinterpret_cast<const std::complex<float>*>(my_link),
					num_colorspins,
					num_colorspins);

			// Generate G_c u_forw G_c, should equal u_back^\dagger
			EigenCMat gc_u_clov_gc = G_c * u_clov * G_c;

			// Subtract the u_back^\dagger
			gc_u_clov_gc -= u_clov.adjoint();

			// Check result is zero
			for(int col=0; col < num_colorspins; ++col) {
				for(int row=0; row < num_colorspins; ++row) {

					ASSERT_LT( fabs((gc_u_clov_gc(col,row)).real()), 5.0e-7);
					ASSERT_LT( fabs((gc_u_clov_gc(col,row)).imag()), 5.0e-7);

				}
			}

		} // cbsites

	} // cb

	MasterLog(INFO, "Checking AD^dagger = G_c DA G_c");
	for(int target_cb=0; target_cb < 2; ++target_cb) {
		// Communicate with MuLink
		CommunicateHaloSync<CoarseGauge,MuLinkADAccessor>(*(halos[target_cb]), coarse_links, target_cb);
	}

	// For all directions
	for(int mu_forw=0; mu_forw < 8; ++mu_forw) {
		MasterLog(INFO,"Doing mu=%d", mu_forw);

		// For all sites
		for(int cb=0; cb < 2; ++cb) {
			for(int cbsite=0; cbsite < num_cbsites; ++cbsite) {

				// Grab my link
				const float* my_link = coarse_links.GetSiteDirDADataPtr(cb,cbsite,mu_forw);

				// Get the back link
				const float* back_link = GetNeighborDir<CoarseGauge,BackLinkADAccessor>(*(halos[cb]),
						coarse_links, mu_forw, cb, cbsite);

				// Wrap links up as eigen matrix wrappers
				Eigen::Map< const EigenCMat > u_forw(reinterpret_cast<const std::complex<float>*>(my_link),
						num_colorspins,
						num_colorspins);
				Eigen::Map< const EigenCMat > u_back(reinterpret_cast<const std::complex<float>*>(back_link),
						num_colorspins,
						num_colorspins);

				// Generate G_c u_forw G_c, should equal u_back^\dagger
				EigenCMat gc_u_forw_gc = G_c * u_forw * G_c;

				// Subtract the u_back^\dagger
				gc_u_forw_gc -= u_back.adjoint();

				// Check result is zero
				for(int col=0; col < num_colorspins; ++col) {
					for(int row=0; row < num_colorspins; ++row) {

						ASSERT_LT( fabs((gc_u_forw_gc(col,row)).real()), 5.0e-7);
						ASSERT_LT( fabs((gc_u_forw_gc(col,row)).imag()), 5.0e-7);

					}
				}

			} // cbsites

		} // cb
	} //  dir

	MasterLog(INFO, "Checking Inverse Clover Term");
	for(int cb=0; cb < 2; ++cb) {
		for(int cbsite=0; cbsite < num_cbsites; ++cbsite) {

			// Grab my link
			const float* my_link = coarse_links.GetSiteInvDiagDataPtr(cb,cbsite);


			// Wrap links up as eigen matrix wrappers
			Eigen::Map< const EigenCMat > u_clov(reinterpret_cast<const std::complex<float>*>(my_link),
					num_colorspins,
					num_colorspins);

			// Generate G_c u_forw G_c, should equal u_back^\dagger
			EigenCMat gc_u_clov_gc = G_c * u_clov * G_c;

			// Subtract the u_back^\dagger
			gc_u_clov_gc -= u_clov.adjoint();

			// Check result is zero
			for(int col=0; col < num_colorspins; ++col) {
				for(int row=0; row < num_colorspins; ++row) {

					ASSERT_LT( fabs((gc_u_clov_gc(col,row)).real()), 1.0e-7);
					ASSERT_LT( fabs((gc_u_clov_gc(col,row)).imag()), 1.0e-7);

				}
			}

		} // cbsites

	} // cb
}


// Test  (  y + alpha D x  ) is Gamma_5 Hermitian
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

		// Compute y_dag2 = Gamma y_dag = y_dag Gamma_c
		applyGamma(gc_tmp2, y_dag, RB[cb]);



		Gaussian(gc_tmp1,SUBSET_ALL);    // This is a temporary so set it to junk
		float alpha = -2.3;

		// this does  y_dag += alpha D^\dagger x
#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			D.M_D_xpay(y_dag,alpha, coarse_links,x,cb,LINOP_DAGGER,tid);
		}


		//
		//
		// Step 1:  tmp1 = Gamma_c x
		applyGamma(gc_tmp1, x, RB[source_cb]);


		// Step 2:  gc_tmp2 = gc_tmp2 + alpha D (Gamma_c x)
		//                  = y_dag Gamma_c + alpha D (Gamma_c x)
		//
#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			D.M_D_xpay(gc_tmp2,alpha,coarse_links, gc_tmp1,cb,LINOP_OP,tid);
		}

		// Now apply y_dag2 = Gamma_c gc_tmp2 =>
		//  y_dag2 = Gamma_c(  y_dag Gamma_c + alpha D Gamma_c x)
		//         = y_dag + alpha Gamma_c D Gamma_c
		//         = y_dag + alpha D^\dagger

		applyGamma(y_dag2,gc_tmp2, RB[cb]);

		double norm_diff = XmyNorm2Vec(y_dag,y_dag2,RB[cb])[0];
		MasterLog(INFO, "cb=%d NormDiff=%16.8e", cb, norm_diff);
		ASSERT_LT( norm_diff, 5.0e-9);
	}
}


// Test A^{-1} is Gamma_c Hermitian
TEST_F(EOBitsTesting, TestGcHermDiagInv)
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

		double norm_diff = XmyNorm2Vec(y_dag,y_dag2,RB[cb])[0];
		MasterLog(INFO, "cb=%d NormDiff=%16.8e", cb, norm_diff);
		ASSERT_LT( norm_diff, 5.0e-10);
	}
}

// Test A is Gamma_c hermitian
TEST_F(EOBitsTesting, TestGcHermDiag)
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

		double norm_diff = XmyNorm2Vec(y_dag,y_dag2,RB[cb])[0];
		MasterLog(INFO, "cb=%d NormDiff=%16.8e", cb, norm_diff);
		ASSERT_LT( norm_diff, 5.0e-10);
	}
}



// Test applying DA dagger is the same as applying Gamma_c AD Gamma_c
TEST_F(EOBitsTesting, TestADDagger)
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
		// Fill everything with junk unless it needs to be zero
		// so accidentally zeroed things won't interfere with the
		// tests.
		Gaussian(y_dag);
		Gaussian(gc_tmp2);
		Gaussian(gc_tmp1);
		Gaussian(x);

		CopyVec(y_dag2,y_dag);

#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			D.M_AD_xpayz(y_dag,1.0,coarse_links,y_dag,x,cb,LINOP_DAGGER,tid);
		}

		// Y dag should hold 0 +  (AD)^|dagger

		// Now in Y dag2 we want 0 + G_c DA G_c
		applyGamma(gc_tmp1, x, RB[1-cb]);
		applyGamma(gc_tmp2, y_dag2, RB[cb]);
#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			// y_dag' Gamma_c + DA Gamma_c x
			D.M_DA_xpayz(gc_tmp2,1.0,coarse_links,gc_tmp2,gc_tmp1,cb,LINOP_OP,tid);
		}
		// y_dag' + Gamma_c DA Gamma_c
		applyGamma(y_dag2,gc_tmp2, RB[cb]);

		double norm_diff = XmyNorm2Vec(y_dag,y_dag2,RB[cb])[0];
		MasterLog(INFO, "cb=%d NormDiff=%16.8e", cb, norm_diff);
		ASSERT_LT( norm_diff, 1.0e-9);


	}

	for(int cb=0; cb < n_checkerboard; ++cb) {
		// Fill everything with junk unless it needs to be zero
		// so accidentally zeroed things won't interfere with the
		// tests.
		Gaussian(y_dag);
		Gaussian(gc_tmp2);
		Gaussian(gc_tmp1);
		Gaussian(x);
		CopyVec(y_dag2,y_dag);

#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			D.M_DA_xpayz(y_dag,1.0,coarse_links,y_dag,x,cb,LINOP_DAGGER,tid);
		}

		// Y dag should hold 0 +  (AD)^|dagger

		// Now in Y dag2 we want 0 + G_c DA G_c
		applyGamma(gc_tmp1, x, RB[1-cb]);
		applyGamma(gc_tmp2, y_dag2, RB[cb]);

#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			D.M_AD_xpayz(gc_tmp2,1.0,coarse_links,gc_tmp2,gc_tmp1,cb,LINOP_OP,tid);
		}
		applyGamma(y_dag2,gc_tmp2, RB[cb]);

		double norm_diff = XmyNorm2Vec(y_dag,y_dag2,RB[cb])[0];
		MasterLog(INFO, "cb=%d NormDiff=%16.8e", cb, norm_diff);
		ASSERT_LT( norm_diff, 1.0e-9);
	}
}


// Applying   A^dagger ( D A^{-1} )^dagger = D^\dagger
TEST_F(EOBitsTesting, TestADagDAinvDaggerEqDag)
{
	CoarseGauge& coarse_links = getCoarseLinks();

	// Get the Lattice info for the coarse lattice
	const LatticeInfo& coarse_info = getCoarseInfo();

	CoarseDiracOp D(coarse_info);

	// A^dagger  ( D A^{-1} )^\dagger
	//
	{
		CoarseSpinor x(coarse_info);
		CoarseSpinor DA_invdag_x(coarse_info);
		CoarseSpinor Ddag_x(coarse_info);
		CoarseSpinor ADA_invdag_x(coarse_info);

		for(int cb=0; cb < 2; ++cb) {
			Gaussian(x,RB[1-cb]);
			ZeroVec(DA_invdag_x,RB[cb]); // In axpyz so zero
			ZeroVec(Ddag_x, RB[cb]); // In axpy so zero
#pragma omp parallel
			{
				int tid = omp_get_thread_num();
				// ( D A^{-1} )^\dagger = A^{-dagger} D_dagger
				D.M_DA_xpayz(DA_invdag_x,1.0, coarse_links, DA_invdag_x, x, cb, LINOP_DAGGER, tid);
#pragma omp barrier

				// A^\dagger ( A^{-dagger} D^dagger )  = D^\dagger
				D.M_diag(ADA_invdag_x,coarse_links,DA_invdag_x, cb, LINOP_DAGGER, tid);

#pragma omp barrier
				D.M_D_xpay(Ddag_x,1.0,coarse_links,x, cb,LINOP_DAGGER, tid);
			}
			double norm_diff = XmyNorm2Vec(Ddag_x, ADA_invdag_x ,RB[cb])[0];
			MasterLog(INFO, "cb=%d NormDiff=%16.8e",cb,norm_diff);
			ASSERT_LT(norm_diff, 1.0e-8);
		}
	}

	{
		CoarseSpinor x(coarse_info);
		CoarseSpinor Ax(coarse_info);
		CoarseSpinor AinvDdagAx(coarse_info);
		CoarseSpinor Ddag_x(coarse_info);

		for(int cb=0; cb < 2; ++cb) {
			Gaussian(x,RB[1-cb]);
			ZeroVec(AinvDdagAx,RB[cb]); // In axpyz so zero
			ZeroVec(Ddag_x, RB[cb]); // In axpy so zero
#pragma omp parallel
			{
				int tid = omp_get_thread_num();
				// A^\dagger x
				D.M_diag(Ax,coarse_links,x, 1-cb, LINOP_DAGGER, tid);
#pragma omp barrier

				// (A^{-1} D)^\dagger A =  D_dagger
				D.M_AD_xpayz(AinvDdagAx,1.0, coarse_links, AinvDdagAx, Ax, cb, LINOP_DAGGER, tid);
#pragma omp barrier

				// D_dag
				D.M_D_xpay(Ddag_x,1.0,coarse_links,x, cb,LINOP_DAGGER, tid);
			}
			double norm_diff = XmyNorm2Vec(Ddag_x, AinvDdagAx ,RB[cb])[0];
			MasterLog(INFO, "cb=%d NormDiff=%16.8e",cb,norm_diff);
			ASSERT_LT(norm_diff, 1.0e-8);
		}
	}
}

TEST_F(EOBitsTesting, TestSchur)
{
	// Get stuff from fixture
	CoarseGauge& coarse_links = getCoarseLinks();
	const LatticeInfo& coarse_info = getCoarseInfo();
	CoarseDiracOp D(coarse_info);

	MasterLog(INFO, "Testing Diag L D R  = Schur");
	{
		CoarseSpinor x(coarse_info);
		CoarseSpinor Rx(coarse_info);
		CoarseSpinor DRx(coarse_info);
		CoarseSpinor LDRx(coarse_info);
		CoarseSpinor ALDRx(coarse_info);
		CoarseSpinor Full(coarse_info);

		ZeroVec(Rx);
		ZeroVec(DRx);
		ZeroVec(LDRx);
		ZeroVec(ALDRx);
		ZeroVec(Full);
		Gaussian(x);

		D.R_matrix(Rx,coarse_links, x);
		CopyVec(DRx,Rx,RB[0]);
		D.EOPrecOp(DRx,coarse_links,Rx,1,LINOP_OP);
		D.L_matrix(LDRx, coarse_links, DRx);



#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			for(int cb=0; cb < 2; ++cb) {
				D.M_diag(ALDRx,coarse_links,LDRx,cb,LINOP_OP,tid);
			}
#pragma omp barrier

			for(int cb=0; cb < 2;++cb) {
				D.unprecOp(Full,coarse_links,x,cb, LINOP_OP,tid);
			}
		}

		for(int cb = 0; cb < 2; ++cb) {
			double norm_diff = XmyNorm2Vec(Full, ALDRx,RB[cb])[0];
			MasterLog(INFO, "cb=%d NormDiff=%16.8e",cb,norm_diff);
			ASSERT_LT(norm_diff, 2.0e-8);
		}

	}
}


// Tests that   A_oo ( 1 - A D A D )  is Gamma 5 hermitian.
// I.e.     [ A_op ( 1 - A D A D ) ]^dagger
//         = ( 1 - A D A D )^\dagger A^\dagger_op
//           = Gamma_c A ( 1 - A D A D ) Gamma_c
TEST_F(EOBitsTesting, Gamma5HermDiagEOPrecOp)
{
	CoarseGauge& coarse_links = getCoarseLinks();
	const LatticeInfo& info = getCoarseInfo();
	CoarseDiracOp D(info);

	CoarseSpinor  x(info);
	CoarseSpinor tmp1(info);
	CoarseSpinor SymmPrecOpDagAdagx(info);

	CoarseSpinor tmp2(info);
	CoarseSpinor GcASymmPrecOpGcx(info);

	Gaussian(x, SUBSET_ODD);
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		D.M_diag(tmp1,coarse_links, x, ODD, LINOP_DAGGER, tid);
	}

	D.EOPrecOp(SymmPrecOpDagAdagx, coarse_links, tmp1, ODD, LINOP_DAGGER );


	applyGamma(tmp1, x, SUBSET_ODD);
	D.EOPrecOp(tmp2, coarse_links, tmp1, ODD, LINOP_OP);
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		D.M_diag(tmp1,coarse_links, tmp2, ODD, LINOP_OP, tid);
	}
	applyGamma(GcASymmPrecOpGcx,tmp1, SUBSET_ODD);

	double norm_diff = XmyNorm2Vec(SymmPrecOpDagAdagx,GcASymmPrecOpGcx, SUBSET_ODD )[0];
	MasterLog(INFO, "NormDiff=%16.8e",norm_diff);
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
			float *vsite_in = spinor.GetSiteDataPtr(0,cb,cbsite);

			// This wil be: U x spinor
			float *diag_out = diag_spinor.GetSiteDataPtr(0,cb,cbsite);

			// Get the U
			float *u_diag = coarse_links.GetSiteDiagDataPtr(cb,cbsite);

			// Multiply: diag_out = U spinor
			CMatMultNaive(diag_out,u_diag,vsite_in, num_colorspins);

			// Get the inverted U
			float *u_diag_inv = coarse_links.GetSiteInvDiagDataPtr(cb,cbsite);

			// This will be U^{-1} U spinor
			float *s_inv = spinor_inv.GetSiteDataPtr(0,cb,cbsite);

			// Multiply
			CMatMultNaive(s_inv,u_diag_inv,diag_out,num_colorspins);
		}
	}

	double norm_diff = XmyNorm2Vec(spinor_inv,spinor)[0];
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

			double norm_diff = XmyNorm2Vec(spinor_inv,spinor,RB[cb])[0];
			MasterLog(INFO, "cb=%d NormDiff=%16.8e", cb, norm_diff);
			ASSERT_LT( norm_diff, 5.0e-10);
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
				D.M_AD(invOffx, coarse_links, x, cb, LINOP_OP, tid);
#pragma omp barrier
				D.M_diag(mInvOffx, coarse_links, invOffx, cb, LINOP_OP, tid );
#pragma omp barrier
				D.M_D_xpay(offx, 1.0, coarse_links, x, cb, LINOP_OP, tid);
			}




			double norm_diff = XmyNorm2Vec(mInvOffx,offx,RB[cb])[0];
			MasterLog(INFO, "cb=%d NormDiff=%16.8e", cb, norm_diff);
			ASSERT_LT(norm_diff, 1.0e-9);
		}
	}
}


// Test applying  A^{-1} . D = (A^{-1}D)
TEST_F(EOBitsTesting, TestADLinks )
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
				// invOffx = invOffx + A^{-1}D 	x = A^{-1}D x since invOffx is inited to zero
				D.M_AD_xpayz(invOffx, 1.0, coarse_links, invOffx, x, cb, LINOP_OP, tid);

#pragma omp barrier
				D.M_D_xpay(offx, 1.0, coarse_links, x, cb, LINOP_OP, tid);

#pragma omp barrier
				D.M_diagInv(mInvOffx, coarse_links, offx, cb, LINOP_OP, tid );
			}


			double norm_diff = XmyNorm2Vec(mInvOffx,invOffx,RB[cb])[0];
			MasterLog(INFO, "cb=%d NormDiff=%16.8e", cb, norm_diff);
			ASSERT_LT(norm_diff, 1.0e-9);
		}
	}
}

// Test applying DA in a single op is the same as applying D after applying A.
TEST_F(EOBitsTesting, TestDALinks )
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

		CoarseSpinor offInvx(coarse_info);

		CoarseSpinor offx(coarse_info);
		CoarseSpinor mInvx(coarse_info);

		for(int cb =0; cb < 2; ++cb) {
			ZeroVec(offx, RB[cb]); // Zero on target CB
			ZeroVec(offInvx,RB[cb]);
			ZeroVec(mInvx,RB[cb]);
			ZeroVec(offx);

#pragma omp parallel
			{
				int tid=omp_get_thread_num();
				// invOffx = invOffx + D A^{-1} x = D A^{-1} x since invOffx is inited to zero
				D.M_DA_xpayz(offInvx, 1.0, coarse_links, offInvx, x, cb, LINOP_OP, tid);

#pragma omp barrier
				D.M_diagInv(mInvx, coarse_links, x, 1-cb, LINOP_OP, tid );

#pragma omp barrier
				D.M_D_xpay(offx, 1.0, coarse_links, mInvx, cb, LINOP_OP, tid);


			}


			double norm_diff = XmyNorm2Vec(offInvx,offx,RB[cb])[0];
			MasterLog(INFO, "cb=%d NormDiff=%16.8e", cb, norm_diff);
			ASSERT_LT(norm_diff, 1.0e-9);
		}
	}
}



TEST_F(EOBitsTesting, TestRMat1)
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
		D.R_matrix(Rx,coarse_links,x);
		D.R_inv_matrix(RinvRx,coarse_links, Rx);
		double norm_diff = XmyNorm2Vec(x,RinvRx)[0];
		MasterLog(INFO, "NormDiff=%16.8e",norm_diff);
		ASSERT_LT(norm_diff, 1.0e-9);

	}
}

TEST_F(EOBitsTesting, TestRMat2)
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
		D.R_inv_matrix(Rinvx,coarse_links, x);
		D.R_matrix(RRinvx,coarse_links,Rinvx);

		double norm_diff = XmyNorm2Vec(x,RRinvx)[0];
		MasterLog(INFO, "NormDiff=%16.8e",norm_diff);
		ASSERT_LT(norm_diff, 1.0e-9);

	}
}

TEST_F(EOBitsTesting, TestLMat1)
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
		D.L_matrix(Lx,coarse_links,x);
		D.L_inv_matrix(LinvLx,coarse_links, Lx);
		for(int cb = 0; cb < 2; ++cb) {
			double norm_diff = XmyNorm2Vec(x,LinvLx,RB[cb])[0];
			MasterLog(INFO, "cb=%d NormDiff=%16.8e",cb,norm_diff);
			ASSERT_LT(norm_diff, 1.0e-9);
		}

	}
}


TEST_F(EOBitsTesting, TestLMat2)
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
		D.L_inv_matrix(Linvx,coarse_links,x);
		D.L_matrix(LLinvx,coarse_links, Linvx);
		for(int cb = 0; cb < 2; ++cb) {
			double norm_diff = XmyNorm2Vec(x,LLinvx,RB[cb])[0];
			MasterLog(INFO, "cb=%d NormDiff=%16.8e",cb,norm_diff);
			ASSERT_LT(norm_diff, 1.0e-9);
		}

	}
}



int main(int argc, char *argv[]) 
{
	::testing::InitGoogleTest(&argc, argv);
	::testing::AddGlobalTestEnvironment(new MGTesting::TestEnv(&argc,&argv));
	return RUN_ALL_TESTS();

}


// This is our test fixture. No tear down or setup
// Constructor. Set Up Only Once rather than setup/teardown
void EOBitsTesting::SetUp()  {
		latdims={{8,8,8,4}};
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

