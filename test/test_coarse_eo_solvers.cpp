/*
 * test_nodeinfo.cpp
 *
 *  Created on: Sep 25, 2015
 *      Author: bjoo
 */


#include "gtest/gtest.h"
#include "lattice/cmat_mult.h"
#include "utils/memory.h"
#include "utils/print_utils.h"
#include <random>
#include "MG_config.h"
#include "test_env.h"

#include <omp.h>
#include <cstdio>

#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_op.h"

using namespace MG;
using namespace MG;

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

TEST_F(EOBitsTesting,TestEOCoarseFGMRES)
{

}
int main(int argc, char *argv[])
{
	return MGTesting::TestMain(&argc, argv);
}


// This is our test fixture. No tear down or setup
// Constructor. Set Up Only Once rather than setup/teardown
void EOBitsTesting::SetUp()
{
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
	multInvClovOffDiagLeft(coarse_links);
	multInvClovOffDiagRight(coarse_links);

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

