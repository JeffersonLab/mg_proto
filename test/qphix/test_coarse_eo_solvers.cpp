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
#include "../test_env.h"

#include <omp.h>
#include <cstdio>

#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_op.h"
#include "lattice/coarse/coarse_l1_blas.h"
#include "lattice/coarse/coarse_eo_wilson_clover_linear_operator.h"
#include "lattice/qphix/qphix_eo_clover_linear_operator.h"
#include "lattice/qphix/mg_level_qphix.h"

#include "lattice/coarse/invmr_coarse.h"
#include "lattice/coarse/invbicgstab_coarse.h"
#include "lattice/coarse/invfgmres_coarse.h"
#include "utils/print_utils.h"

#include "../qdpxx/qdpxx_latticeinit.h"
#include "../qdpxx/qdpxx_utils.h"
using namespace MG;
using namespace MG;
using namespace MGTesting;

// Test fixture to save me writing code all the time.
class EOSolverTesting : public ::testing::Test {
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
	std::shared_ptr<QPhiXWilsonCloverEOLinearOperatorF> M;

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
	QPhiXMultigridLevelsEO mg_levels;

	// A one liner to access the coarse links generated
	std::shared_ptr<CoarseGauge> getCoarseLinks() {
		return (mg_levels.coarse_levels[0].gauge);
	}

	// A one liner to get at the coarse info
	const LatticeInfo& getCoarseInfo() const {
		return (mg_levels.coarse_levels[0].gauge)->GetInfo();
	}

};

TEST_F(EOSolverTesting,TestEOCoarseMRSmoother)
{
	const LatticeInfo& info=getCoarseInfo();
	std::shared_ptr<CoarseGauge> coarse_links = getCoarseLinks();

	//
	MRSolverParams p;
	p.MaxIter=5;
	p.Omega=1.1;
	p.RsdTarget = 1.0e-5;
	p.VerboseP = true;

	std::shared_ptr<CoarseEOWilsonCloverLinearOperator> M_coarse=std::make_shared<CoarseEOWilsonCloverLinearOperator>(coarse_links, 1);
	std::shared_ptr<const MRSmootherCoarse> smoother=std::make_shared<const MRSmootherCoarse>(*M_coarse, p);
	UnprecMRSmootherCoarseWrapper wrapped(smoother, M_coarse);

	CoarseSpinor source(info);
	CoarseSpinor result(info);

	Gaussian(source);
	ZeroVec(result);

	// DOING EO-smooth
	MasterLog(INFO, "Doing EO Smooth");
	wrapped(result, source);

	CoarseWilsonCloverLinearOperator M_unprec(coarse_links, 1);
	MRSmootherCoarse smoother_unprec(M_unprec, p);

	CoarseSpinor result_unprec(info);
	ZeroVec(result_unprec);
	MasterLog(INFO, "Doing Unprec Smooth");
	smoother_unprec(result_unprec, source);

	CoarseSpinor r(info);
	CopyVec(r,source);
	CoarseSpinor Ax(info);
	M_unprec(Ax,result,LINOP_OP);
	double norm_diff_prec = sqrt(XmyNorm2Vec(r,Ax));

	CopyVec(r,source);
	M_unprec(Ax,result_unprec,LINOP_OP);
	double norm_diff_unprec =  sqrt(XmyNorm2Vec(r, Ax));

	double norm_diff_source = sqrt(Norm2Vec(source));

	MasterLog(INFO, "Unprec Residuum after %d Unprec Smoother Reductions: || r || = %16.8e|| r || / || b ||=%16.8e",
			p.MaxIter, norm_diff_unprec, norm_diff_unprec/norm_diff_source);

	MasterLog(INFO, "Unprec Residuum after %d Prec Smoother Reductions: || r || = %16.8e || r || / || b ||=%16.8e",
			p.MaxIter, norm_diff_prec, norm_diff_prec/norm_diff_source);

	ASSERT_LT( norm_diff_prec, norm_diff_unprec);

}

TEST_F(EOSolverTesting,TestEOCoarseBiCGStab)
{
	const LatticeInfo& info=getCoarseInfo();
	std::shared_ptr<CoarseGauge> coarse_links = getCoarseLinks();

	//
	LinearSolverParamsBase p;
	p.MaxIter=500;
	p.RsdTarget = 1.0e-5;
	p.VerboseP = true;

	std::shared_ptr<CoarseEOWilsonCloverLinearOperator> M_coarse=std::make_shared<CoarseEOWilsonCloverLinearOperator>(coarse_links, 1);
	std::shared_ptr<const BiCGStabSolverCoarse> bicgstab=std::make_shared<const BiCGStabSolverCoarse>(*M_coarse, p);
	UnprecBiCGStabSolverCoarseWrapper wrapped(bicgstab, M_coarse);

	CoarseSpinor source(info);
	CoarseSpinor result(info);

	Gaussian(source);
	ZeroVec(result);

	// DOING EO-smooth
	MasterLog(INFO, "Doing EO Solver");
	wrapped(result, source);

	CoarseWilsonCloverLinearOperator M_unprec(coarse_links, 1);
	BiCGStabSolverCoarse bicgstab_unprec(M_unprec, p);

	CoarseSpinor result_unprec(info);
	ZeroVec(result_unprec);
	MasterLog(INFO, "Doing Unprec Smooth");
	bicgstab_unprec(result_unprec, source);

	CoarseSpinor r(info);
	CopyVec(r,source);
	CoarseSpinor Ax(info);
	M_unprec(Ax,result,LINOP_OP);
	double norm_diff_prec = sqrt(XmyNorm2Vec(r,Ax));

	CopyVec(r,source);
	M_unprec(Ax,result_unprec,LINOP_OP);
	double norm_diff_unprec =  sqrt(XmyNorm2Vec(r, Ax));

	double norm_diff_source = sqrt(Norm2Vec(source));

	MasterLog(INFO, "Unprec Residuum after Unprec Solve: || r || = %16.8e || r || / || b ||=%16.8e",
			norm_diff_unprec, norm_diff_unprec/norm_diff_source);

	MasterLog(INFO, "Unprec Residuum after Prec Solve: || r || = %16.8e || r || / || b ||=%16.8e",
			norm_diff_prec, norm_diff_prec/norm_diff_source);



}

TEST_F(EOSolverTesting,TestEOCoarseFGMRES)
{
	const LatticeInfo& info=getCoarseInfo();
	std::shared_ptr<CoarseGauge> coarse_links = getCoarseLinks();

	//
	FGMRESParams p;
	p.MaxIter=500;
	p.NKrylov = 10;
	p.RsdTarget = 1.0e-5;
	p.VerboseP = true;

	std::shared_ptr<CoarseEOWilsonCloverLinearOperator> M_coarse=std::make_shared<CoarseEOWilsonCloverLinearOperator>(coarse_links, 1);
	std::shared_ptr<const FGMRESSolverCoarse> fgmres=std::make_shared<const FGMRESSolverCoarse>(*M_coarse, p);
	UnprecFGMRESSolverCoarseWrapper wrapped(fgmres, M_coarse);

	CoarseSpinor source(info);
	CoarseSpinor result(info);

	Gaussian(source);
	ZeroVec(result);

	// DOING EO-smooth
	MasterLog(INFO, "Doing EO FGMRES Solver");
	wrapped(result, source);

	CoarseWilsonCloverLinearOperator M_unprec(coarse_links, 1);
	FGMRESSolverCoarse fgmres_unprec(M_unprec, p);

	CoarseSpinor result_unprec(info);
	ZeroVec(result_unprec);
	MasterLog(INFO, "Doing Unprec FGMRES Solver");
	fgmres_unprec(result_unprec, source);

	CoarseSpinor r(info);
	CopyVec(r,source);
	CoarseSpinor Ax(info);
	M_unprec(Ax,result,LINOP_OP);
	double norm_diff_prec = sqrt(XmyNorm2Vec(r,Ax));

	CopyVec(r,source);
	M_unprec(Ax,result_unprec,LINOP_OP);
	double norm_diff_unprec =  sqrt(XmyNorm2Vec(r, Ax));

	double norm_diff_source = sqrt(Norm2Vec(source));

	MasterLog(INFO, "Unprec Residuum after Unprec Solve: || r || = %16.8e || r || / || b ||=%16.8e",
			norm_diff_unprec, norm_diff_unprec/norm_diff_source);

	MasterLog(INFO, "Unprec Residuum after Prec Solve: || r || = %16.8e || r || / || b ||=%16.8e",
			norm_diff_prec, norm_diff_prec/norm_diff_source);



}

int main(int argc, char *argv[])
{
	return MGTesting::TestMain(&argc, argv);
}


// This is our test fixture. No tear down or setup
// Constructor. Set Up Only Once rather than setup/teardown
void EOSolverTesting::SetUp()
{
	latdims={{8,8,8,8}};
	initQDPXXLattice(latdims);

	LatticeInfo info(latdims);
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

	M=std::make_shared<QPhiXWilsonCloverEOLinearOperatorF>(info,m_q, c_sw, t_bc,u);

	QDPIO::cout << "Calling Level setup" << std::endl;
	SetupQPhiXMGLevels(level_setup_params,
	 			  	  mg_levels,
					  M);

	QDPIO::cout << "Done" << std::endl;




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

