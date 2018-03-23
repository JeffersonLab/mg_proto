/*
 * recursive_vcycle_profile.cpp
 *
 *  Created on: Nov 8, 2017
 *      Author: bjoo
 */
#include "gtest/gtest.h"
#include "../test_env.h"
#include "../mock_nodeinfo.h"
#include "../qdpxx/qdpxx_utils.h"

#include <lattice/qphix/invfgmres_qphix.h>

#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/coarse/coarse_types.h"


#include "lattice/fine_qdpxx/mg_params_qdpxx.h"
#include "lattice/qphix/mg_level_qphix.h"
#include "lattice/qphix/vcycle_recursive_qphix.h"

// f#include <ittnotify.h>

using namespace MG;
using namespace MGTesting;
using namespace QDP;

TEST(QPhiXTestRecursiveVCycle, TestLevelSetup2Level)
{
	IndexArray latdims={{16,16,16,16}};



	initQDPXXLattice(latdims);

	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	IndexArray node_orig=NodeInfo().NodeCoords();
	for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

	float m_q = 0.01;
	float c_sw = 1.25;

	int t_bc=-1; // Antiperiodic t BCs


	multi1d<LatticeColorMatrix> u(Nd);
	for(int mu=0; mu < Nd; ++mu) {
		//u[mu]=1;
		gaussian(u[mu]);
		reunit(u[mu]);
	}
	// Move to QPhiX space:
	LatticeInfo fine_info(node_orig,latdims,4,3,NodeInfo());

	MasterLog(INFO,"Creating M");

	// Create float prec linop for preconditioner
	std::shared_ptr<QPhiXWilsonCloverLinearOperatorF> M_f=
			std::make_shared<QPhiXWilsonCloverLinearOperatorF>(fine_info, m_q, c_sw, t_bc,u);

	// Create full prec linop for solver
	QPhiXWilsonCloverLinearOperator M(fine_info,m_q, c_sw, t_bc,u);

	SetupParams level_setup_params = {
			3,       // Number of levels
			{24,24},   // Null vecs on L0, L1
			{
					{4,4,4,4},  // Block Size from L0->L1
					{2,2,2,2}   // Block Size from L1->L2
			},
			{500,500},          // Max Nullspace Iters
			{5e-6,5e-6},        // Nullspace Target Resid
			{false,false}
	};


	QPhiXMultigridLevels mg_levels;
	MasterLog(INFO,"Calling Level setup");
	SetupQPhiXMGLevels(level_setup_params, mg_levels, M_f);
}

#if 0
TEST(QPhiXTestRecursiveVCycle, TestLevelSetup2Level)
{
	IndexArray latdims={{16,16,16,16}};



	initQDPXXLattice(latdims);

	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	IndexArray node_orig=NodeInfo().NodeCoords();
	for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

	float m_q = 0.01;
	float c_sw = 1.25;

	int t_bc=-1; // Antiperiodic t BCs


	multi1d<LatticeColorMatrix> u(Nd);
	for(int mu=0; mu < Nd; ++mu) {
		//u[mu]=1;
		gaussian(u[mu]);
		reunit(u[mu]);
	}
	// Move to QPhiX space:
	LatticeInfo fine_info(node_orig,latdims,4,3,NodeInfo());

	MasterLog(INFO,"Creating M");

	// Create float prec linop for preconditioner
	std::shared_ptr<QPhiXWilsonCloverLinearOperatorF> M_f=
			std::make_shared<QPhiXWilsonCloverLinearOperatorF>(fine_info, m_q, c_sw, t_bc,u);

	// Create full prec linop for solver
	QPhiXWilsonCloverLinearOperator M(fine_info,m_q, c_sw, t_bc,u);

	SetupParams level_setup_params = {
			3,       // Number of levels
			{24,24},   // Null vecs on L0, L1
			{
					{4,4,4,4},  // Block Size from L0->L1
					{2,2,2,2}   // Block Size from L1->L2
			},
			{500,500},          // Max Nullspace Iters
			{5e-6,5e-6},        // Nullspace Target Resid
			{false,false}
	};


	QPhiXMultigridLevels mg_levels;
	MasterLog(INFO,"Calling Level setup");
	SetupQPhiXMGLevels(level_setup_params, mg_levels, M_f);


	MasterLog(INFO, "mg_levels has %d levels", mg_levels.n_levels);

	{
		const LatticeInfo& fine_info = (*mg_levels.fine_level.info);
		const LatticeInfo& M_info = (*mg_levels.fine_level.M).GetInfo();
		const IndexArray fine_va = fine_info.GetLatticeDimensions();
		const IndexArray M_v= M_info.GetLatticeDimensions();
		const int fine_nc = fine_info.GetNumColors();
		const int fine_ns= fine_info.GetNumSpins();
		const int fine_M_nc = M_info.GetNumColors();
		const int fine_M_ns = M_info.GetNumSpins();
		const int num_vecs = mg_levels.fine_level.null_vecs.size();

		MasterLog(INFO, "Level 0 has: Volume=(%d,%d,%d,%d) Ns=%d Nc=%d "
				"M->getInfo() has volume=(%d,%d,%d,%d) Nc=%d Ns=%d num_null_vecs=%d",
				fine_v[0],fine_v[1],fine_v[2],fine_v[3],fine_ns,fine_nc,
				M_v[0],M_v[1],M_v[2],M_v[3], fine_M_nc,fine_M_ns, num_vecs);
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
		MasterLog(INFO, "Level %d has: Volume=(%d,%d,%d,%d) Ns=%d Nc=%d "
				"M->getInfo() has volume=(%d,%d,%d,%d) Nc=%d Ns=%d num_null_vecs=%d",
				level+1,fine_v[0],fine_v[1],fine_v[2],fine_v[3],fine_ns,fine_nc,
				M_v[0],M_v[1],M_v[2],M_v[3], fine_M_nc,fine_M_ns, num_vecs);
	}
	// V Cycle parametere
	std::vector<VCycleParams> v_params(2);

	for(int level=0; level < mg_levels.n_levels-1; level++) {
		MasterLog(INFO,"Level = %d",level);

		v_params[level].pre_smoother_params.MaxIter=4;
		v_params[level].pre_smoother_params.RsdTarget = 0.1;
		v_params[level].pre_smoother_params.VerboseP = false;
		v_params[level].pre_smoother_params.Omega = 1.1;

		v_params[level].post_smoother_params.MaxIter=3;
		v_params[level].post_smoother_params.RsdTarget = 0.1;
		v_params[level].post_smoother_params.VerboseP = false;
		v_params[level].post_smoother_params.Omega = 1.1;

		v_params[level].bottom_solver_params.MaxIter=25;
		v_params[level].bottom_solver_params.NKrylov = 6;
		v_params[level].bottom_solver_params.RsdTarget= 0.1;
		v_params[level].bottom_solver_params.VerboseP = false;

		v_params[level].cycle_params.MaxIter=1;
		v_params[level].cycle_params.RsdTarget=0.1;
		v_params[level].cycle_params.VerboseP = false;
	}


	VCycleRecursiveQPhiX v_cycle(v_params,mg_levels);


	FGMRESParams fine_solve_params;
	fine_solve_params.MaxIter=200;
	fine_solve_params.RsdTarget=1.0e-13;
	fine_solve_params.VerboseP = true;
	fine_solve_params.NKrylov = 5;


	FGMRESSolverQPhiX FGMRESOuter(M,fine_solve_params, &v_cycle);

	MasterLog(INFO, "*** Recursive VCycle Structure + Solver Created");

	QPhiXSpinor psi_in(fine_info);
	QPhiXSpinor chi_out(fine_info);
	Gaussian(psi_in);
	ZeroVec(chi_out);
	double psi_norm = sqrt(Norm2Vec(psi_in));
	MasterLog(INFO, "psi_in has norm = %16.8e",psi_norm);

	QDP::StopWatch swatch;
	swatch.reset();
	// __itt_resume();
	swatch.start();
	LinearSolverResults res=FGMRESOuter(chi_out, psi_in);
	swatch.stop();
	// __itt_pause();

	MasterLog(INFO, "Solve time is %16.8e sec.", swatch.getTimeInSeconds());

	// Compute true residuum
	QPhiXSpinor Ax(fine_info);
	M(Ax,chi_out,LINOP_OP);
	double diff = sqrt(XmyNorm2Vec(psi_in,Ax));
	double diff_rel = diff/psi_norm;
	MasterLog(INFO,"|| b - A x || = %16.8e", diff);
	MasterLog(INFO,"|| b - A x ||/ || b || = %16.8e",diff_rel);

	ASSERT_EQ( res.resid_type, RELATIVE);
	ASSERT_LT( res.resid, 1.0e-13);
	ASSERT_LT( toDouble(diff_rel), 1.0e-13);

}
#endif

int main(int argc, char *argv[])
{
	return MGTesting::TestMain(&argc, argv);
}




