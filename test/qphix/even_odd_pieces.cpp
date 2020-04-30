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
#include "utils/timer.h"

// f#include <ittnotify.h>

using namespace MG;
using namespace MGTesting;
using namespace QDP;
using namespace MG::aux;

TEST(QPhiXTestRecursiveVCycle, TestLevelSetup2Level)
{
	//IndexArray latdims={{32,32,32,32}};
	// IndexArray latdims={{8,8,8,8}};



	// initQDPXXLattice(latdims);

	// QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	// IndexArray node_orig=NodeInfo().NodeCoords();
	// for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

	// float m_q = 0.01;
	// float c_sw = 1.25;

	// int t_bc=-1; // Antiperiodic t BCs


	// multi1d<LatticeColorMatrix> u(Nd);
	// for(int mu=0; mu < Nd; ++mu) {
	// 	//u[mu]=1;
	// 	gaussian(u[mu]);
	// 	reunit(u[mu]);
	// }
	// // Move to QPhiX space:
	// LatticeInfo fine_info(node_orig,latdims,4,3,NodeInfo());

	// MasterLog(INFO,"Creating M");

	// // Create float prec linop for preconditioner
	// std::shared_ptr<QPhiXWilsonCloverLinearOperatorF> M_f=
	// 		std::make_shared<QPhiXWilsonCloverLinearOperatorF>(fine_info, m_q, c_sw, t_bc,u);

	// // Create full prec linop for solver
	// QPhiXWilsonCloverLinearOperator M(fine_info,m_q, c_sw, t_bc,u);

	// SetupParams level_setup_params = {
	// 		3,       // Number of levels
	// 		{24,32},   // Null vecs on L0, L1
	// 		{
	// 				{2,2,2,2},  // Block Size from L0->L1
	// 				{2,2,2,2}   // Block Size from L1->L2
	// 		},
	// 		{500,500},          // Max Nullspace Iters
	// 		{5e-6,5e-6},        // Nullspace Target Resid
	// 		{false,false}
	// };


	// QPhiXMultigridLevels mg_levels;
	// MasterLog(INFO,"Calling Level setup");
	// SetupQPhiXMGLevels(level_setup_params, mg_levels, M_f);
}

#if 1
TEST(QPhiXTestRecursiveVCycle, TestVCycle2Level)
{
	//IndexArray latdims={{32,32,32,32}};
	IndexArray latdims={{8,8,16,16}};



	initQDPXXLattice(latdims);

	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	IndexArray node_orig=NodeInfo().NodeCoords();
	for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

	float m_q = 0.01;
	float c_sw = 1.25;

	int t_bc=-1; // Antiperiodic t BCs
	double tol=5e-7;

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
			2,       // Number of levels
			{24},   // Null vecs on L0, L1
			{
					{2,2,2,2},  // Block Size from L0->L1
			},
			{500},          // Max Nullspace Iters
			{5e-6},        // Nullspace Target Resid
			{false,false}
	};


	QPhiXMultigridLevels mg_levels;
	MasterLog(INFO,"Calling Level setup");
	SetupQPhiXMGLevels(level_setup_params, mg_levels, M_f);


	MasterLog(INFO, "mg_levels has %d levels", mg_levels.n_levels);

	{
		const LatticeInfo& fine_info = (*mg_levels.fine_level.info);
		const LatticeInfo& M_info = (*mg_levels.fine_level.M).GetInfo();
		const IndexArray fine_v = fine_info.GetLatticeDimensions();
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
	std::vector<VCycleParams> v_params(1);

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
	fine_solve_params.RsdTarget=tol;
	fine_solve_params.VerboseP = true;
	fine_solve_params.NKrylov = 5;


	FGMRESSolverQPhiX FGMRESOuter(M,fine_solve_params, &v_cycle);

	MasterLog(INFO, "*** Recursive VCycle Structure + Solver Created");

	std::vector<int> ncols = {1, 4, 16, 64};
	for (int ncoli = 0; ncoli < ncols.size(); ncoli++) {
		int ncol = ncols[ncoli];
		MasterLog(INFO, "== Cols %d ==", ncol);

		QPhiXSpinor psi_in(fine_info, ncol);
		QPhiXSpinor chi_out(fine_info, ncol);
		Gaussian(psi_in);
		ZeroVec(chi_out);
		std::vector<double> psi_norm = sqrt(Norm2Vec(psi_in));
		std::vector<double> inv_psi_norm(ncol);
		for (int col=0; col<ncol; ++col) inv_psi_norm[col] = 1/psi_norm[col];
		AxVec(inv_psi_norm, psi_in);

		QDP::StopWatch swatch;
		swatch.reset();
		// __itt_resume();
		swatch.start();
		std::vector<LinearSolverResults> res=FGMRESOuter(chi_out, psi_in);
		swatch.stop();
		// __itt_pause();

		MasterLog(INFO, "Solve time is %16.8e sec.", swatch.getTimeInSeconds());

		// Compute true residuum
		QPhiXSpinor Ax(fine_info, ncol);
		M(Ax,chi_out,LINOP_OP);
		std::vector<double> diff = sqrt(XmyNorm2Vec(psi_in,Ax));
		for (int col=0; col < ncol; ++col){
			double diff_rel = diff[col]/psi_norm[col];
			// MasterLog(INFO,"|| b - A x || = %16.8e", diff[col]);
			// MasterLog(INFO,"|| b - A x ||/ || b || = %16.8e",diff_rel);

			ASSERT_EQ( res[col].resid_type, RELATIVE);
			ASSERT_LT( res[col].resid, tol);
			ASSERT_LT( toDouble(diff_rel), tol);
		}
	}
    Timer::TimerAPI::reportAllTimer();
}
#endif

int main(int argc, char *argv[])
{
	return MGTesting::TestMain(&argc, argv);
}




