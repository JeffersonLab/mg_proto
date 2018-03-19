#include <lattice/fine_qdpxx/invfgmres_qdpxx.h>
#include "gtest/gtest.h"
#include "../test_env.h"
#include "../mock_nodeinfo.h"
#include "qdpxx_utils.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/coarse/coarse_types.h"

#include "qdpxx_utils.h"
#include "lattice/fine_qdpxx/mg_params_qdpxx.h"
#include "lattice/fine_qdpxx/mg_level_qdpxx.h"
#include "lattice/fine_qdpxx/vcycle_recursive_qdpxx.h"
using namespace MG;
using namespace MGTesting;
using namespace QDP;




TEST(TestRecursiveVCycle, TestLevelSetup2Level)
{
	IndexArray latdims={{8,8,8,8}};


	initQDPXXLattice(latdims);

	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	IndexArray node_orig=NodeInfo().NodeCoords();
		for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

	float m_q = 0.01;
	float c_sw = 1.25;

	int t_bc=-1; // Antiperiodic t BCs


	multi1d<LatticeColorMatrix> u(Nd);
	for(int mu=0; mu < Nd; ++mu) {
		u[mu]=1;
		//gaussian(u[mu]);
		//reunit(u[mu]);
	}

	QDPIO::cout << "Creatig M" << std::endl;
	// Create linear operator
	std::shared_ptr<QDPWilsonCloverLinearOperator> M=
			std::make_shared<QDPWilsonCloverLinearOperator>(m_q, c_sw, t_bc,u);


	SetupParams level_setup_params = {
		3,       // Number of levels
		{8,24},   // Null vecs on L0, L1
		{
				{2,2,2,2},  // Block Size from L0->L1
				{2,2,2,2}   // Block Size from L1->L2
		},
		{500,500},          // Max Nullspace Iters
		{5e-6,5e-6},        // Nullspace Target Resid
		{true,false}
	};


	MultigridLevels mg_levels;
	QDPIO::cout << "Calling Level setup" << std::endl;
	SetupMGLevels(level_setup_params, mg_levels, M);
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

	MasterLog(INFO, "Level 0 has: Volume=(%d,%d,%d,%d) Ns=%d Nc=%d M->getInfo() has volume=(%d,%d,%d,%d) Nc=%dNs=%d num_null_vecs=%d",
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
		MasterLog(INFO, "Level %d has: Volume=(%d,%d,%d,%d) Ns=%d Nc=%d  M->getInfo() has volume=(%d,%d,%d,%d) Nc=%dNs=%d num_null_vecs=%d",
				level+1,fine_v[0],fine_v[1],fine_v[2],fine_v[3],fine_ns,fine_nc, M_v[0],M_v[1],M_v[2],M_v[3], fine_M_nc,fine_M_ns, num_vecs);
	}
	// V Ctcle parametere
	std::vector<VCycleParams> v_params(2);

	for(int level=0; level < mg_levels.n_levels-1; level++) {
		QDPIO::cout << "Level =" << level << std::endl;
		v_params[level].pre_smoother_params.MaxIter=4;
		v_params[level].pre_smoother_params.RsdTarget = 0.1;
		v_params[level].pre_smoother_params.VerboseP = false;
		v_params[level].pre_smoother_params.Omega = 1.0;

		v_params[level].post_smoother_params.MaxIter=3;
		v_params[level].post_smoother_params.RsdTarget = 0.1;
		v_params[level].post_smoother_params.VerboseP = false;
		v_params[level].post_smoother_params.Omega = 1.0;

		v_params[level].bottom_solver_params.MaxIter=25;
		v_params[level].bottom_solver_params.NKrylov = 6;
		v_params[level].bottom_solver_params.RsdTarget= 0.1;
		v_params[level].bottom_solver_params.VerboseP = false;

		v_params[level].cycle_params.MaxIter=1;
		v_params[level].cycle_params.RsdTarget=0.1;
		v_params[level].cycle_params.VerboseP = false;
	}


	VCycleRecursiveQDPXX v_cycle(v_params,mg_levels);

#if 1
	FGMRESParams fine_solve_params;
	fine_solve_params.MaxIter=200;
	fine_solve_params.RsdTarget=1.0e-5;
	fine_solve_params.VerboseP = true;
	fine_solve_params.NKrylov = 5;
	FGMRESSolverQDPXX FGMRESOuter(*M,fine_solve_params, &v_cycle);

	QDPIO::cout << "*** Recursive VCycle Structure + Solver Created" << std::endl;

	// Now need to do the coarse test
	LatticeFermion psi_in, chi_out;
	gaussian(psi_in);
	chi_out = zero;
	QDPIO::cout << "psi_in has norm =" << sqrt(norm2(psi_in)) << std::endl;
	LinearSolverResults res=FGMRESOuter(chi_out, psi_in);
	QDPIO::cout << "Returned residue = || r || / || b ||="<< res.resid<< std::endl;

	LatticeFermion r=psi_in;
	LatticeFermion tmp;
	(*M)(tmp,chi_out,LINOP_OP);
	r -= tmp;
	Double diff=sqrt(norm2(r));
	Double diff_rel = diff / sqrt(norm2(psi_in));
	QDPIO::cout << "|| b - A x || = " << diff << std::endl;
	QDPIO::cout << "|| b - A x || = " << diff_rel << std::endl;
	ASSERT_EQ( res.resid_type, RELATIVE);
	ASSERT_LT( res.resid, 1.0e-5);
	ASSERT_LT( toDouble(diff_rel), 1.0e-5);

#endif
}

int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

