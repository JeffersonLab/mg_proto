#include <gtest/gtest.h>
#include "../test_env.h"
#include "../mock_nodeinfo.h"
#include "../qdpxx/qdpxx_latticeinit.h"
#include "../qdpxx/qdpxx_utils.h"

#include <lattice/coarse/invbicgstab_coarse.h>
#include <lattice/coarse/invfgmres_coarse.h>
#include <lattice/coarse/invmr_coarse.h>
#include <lattice/qphix/invbicgstab_qphix.h>
#include <lattice/qphix/invfgmres_qphix.h>



#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_l1_blas.h"
#include "lattice/coarse/block.h"
#include "lattice/fine_qdpxx/qdpxx_helpers.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"
#include "lattice/unprec_solver_wrappers.h"
// Block Stuff
#include "lattice/fine_qdpxx/aggregate_block_qdpxx.h"
#include "lattice/fine_qdpxx/wilson_clover_linear_operator.h"
#include "lattice/coarse/coarse_eo_wilson_clover_linear_operator.h"
#include "lattice/fine_qdpxx/vcycle_qdpxx_coarse.h"
#include "lattice/coarse/vcycle_coarse.h"
#include "lattice/qphix/vcycle_recursive_qphix.h"


// New QPhiX includes
#include "lattice/qphix/qphix_types.h"
#include "lattice/qphix/qphix_qdp_utils.h"
#include "lattice/qphix/qphix_clover_linear_operator.h"
#include "lattice/qphix/qphix_eo_clover_linear_operator.h"
#include "lattice/qphix/mg_level_qphix.h"
#include "lattice/qphix/invmr_qphix.h"
#include "lattice/qphix/vcycle_qphix_coarse.h"

#include <memory>

using namespace MG;
using namespace MGTesting;
using namespace QDP;

// Test fixture to save me writing code all the time.
class VCycleEOTesting : public ::testing::Test {
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
	std::shared_ptr<QPhiXWilsonCloverEOLinearOperator> M_fine_prec;
	std::shared_ptr<QPhiXWilsonCloverEOLinearOperatorF> M_fine;
	std::shared_ptr<QPhiXWilsonCloverLinearOperatorF> M_fine_unprec;
	std::shared_ptr<QPhiXWilsonCloverLinearOperator> M_fine_unprec_full;

	SetupParams level_setup_params = {
			3,       // Number of levels
			{8,8},   // Null vecs on L0, L1
			{
					{2,2,2,2},
					{2,2,2,2}
			},
			{500,500},          // Max Nullspace Iters
			{5e-6,5e-6},        // Nullspace Target Resid
			{false,false}
	};

	// Setup will set this up
	QPhiXMultigridLevelsEO mg_levels;
	QPhiXMultigridLevels mg_levels_unprec;

	// A one liner to access the coarse links generated
	std::shared_ptr<CoarseGauge> getCoarseLinks(int level) {
		return (mg_levels.coarse_levels[level].gauge);
	}

	// A one liner to get at the coarse info
	const LatticeInfo& getCoarseInfo(int level) const {
		return (mg_levels.coarse_levels[level].gauge)->GetInfo();
	}

	const LatticeInfo& getFineInfo(void) const {
			return *(mg_levels.fine_level.info);
		}
	const std::shared_ptr<const CoarseEOWilsonCloverLinearOperator> getCoarseLinOp(int level) const
	{
		return mg_levels.coarse_levels[level].M;
	}

	const std::shared_ptr<const CoarseWilsonCloverLinearOperator> getCoarseUnprecLinOp(int level) const
	{
		return mg_levels_unprec.coarse_levels[level].M;
	}
};



TEST_F(VCycleEOTesting, TestVCycleApply)
{
	const LatticeInfo& fine_info = getCoarseInfo(0);
	const LatticeInfo& coarse_info = getCoarseInfo(1);

	MRSolverParams presmooth;
	presmooth.MaxIter=4;
	presmooth.RsdTarget = 0.1;
	presmooth.Omega = 1.1;
	presmooth.VerboseP = true;

	auto L0LinOp = getCoarseLinOp(0);
	auto L0UnprecOp = getCoarseUnprecLinOp(0);

	// MR Smoother holds only references
	UnprecMRSmootherCoarseWrapper the_smoother( std::make_shared<MRSmootherCoarse>(*L0LinOp,presmooth),L0LinOp);
	MRSmootherCoarse the_unprec_smoother(*L0UnprecOp,presmooth);

	// Set up the CoarseSolver
	FGMRESParams coarse_solve_params;
	coarse_solve_params.MaxIter=200;
	coarse_solve_params.RsdTarget=0.1;
	coarse_solve_params.VerboseP = true;
	coarse_solve_params.NKrylov = 10;
	auto L1LinOp = getCoarseLinOp(1);
	auto L1UnprecOp = getCoarseUnprecLinOp(1);

	UnprecFGMRESSolverCoarseWrapper bottom_solver( std::make_shared<FGMRESSolverCoarse>(*(L1LinOp),coarse_solve_params),L1LinOp);
	FGMRESSolverCoarse bottom_solver_unprec(*L1UnprecOp,coarse_solve_params);
	{
		// Create the 2 Level VCycle

		LinearSolverParamsBase vcycle_params;
		vcycle_params.MaxIter=1;                   // Single application
		vcycle_params.RsdTarget =0.1;              // The desired reduction in || r ||
		vcycle_params.VerboseP = true;			   // Verbosity

		// info my_blocks, and vecs can probably be collected in a 'Transfer' class
		VCycleCoarseEO vcycle( *(mg_levels.coarse_levels[1].info),
				mg_levels.coarse_levels[0].blocklist,
				mg_levels.coarse_levels[0].null_vecs,
				*(L0LinOp),
				the_smoother,
				the_smoother,
				bottom_solver,
				vcycle_params);

		VCycleCoarse vcycle_unprec(*(mg_levels.coarse_levels[1].info),
				mg_levels.coarse_levels[0].blocklist,
				mg_levels.coarse_levels[0].null_vecs,
				*(L0UnprecOp),
				the_unprec_smoother,
				the_unprec_smoother,
				bottom_solver_unprec,
				vcycle_params);

		// Now need to do the coarse test
		CoarseSpinor psi_in(fine_info);
		CoarseSpinor chi_out(fine_info);
		ZeroVec(psi_in,SUBSET_EVEN);
		Gaussian(psi_in, SUBSET_ODD);
		ZeroVec(chi_out);

		double psi_norm = sqrt(Norm2Vec(psi_in));
		MasterLog(INFO, "psi_in has norm = %16.8e",psi_norm);

		LinearSolverResults res = vcycle(chi_out,psi_in);
		ASSERT_EQ( res.n_count, 1);
		ASSERT_EQ( res.resid_type, RELATIVE );
		ASSERT_LT( res.resid, 3.8e-1 );


		CoarseSpinor chi_out_unprec(fine_info);
		ZeroVec(chi_out_unprec);
		psi_norm = sqrt(Norm2Vec(psi_in));
		MasterLog(INFO, "psi_in has norm = %16.8e",psi_norm);

		res = vcycle_unprec(chi_out_unprec,psi_in);
		ASSERT_EQ( res.n_count, 1);
		ASSERT_EQ( res.resid_type, RELATIVE );
		ASSERT_LT( res.resid, 3.8e-1 );

		// Compute true residuum
		CoarseSpinor Ax(fine_info);
		(*L0LinOp)(Ax,chi_out,LINOP_OP);
		double normdiff = sqrt(XmyNorm2Vec(Ax,psi_in, L0LinOp->GetSubset()));
		MasterLog(INFO, "Actual Relative Residuum = %16.8e", normdiff/psi_norm);
		ASSERT_LT( (normdiff/psi_norm), 3.8e-1);

		(*L0UnprecOp)(Ax,chi_out_unprec,LINOP_OP);
		normdiff = sqrt(XmyNorm2Vec(Ax,psi_in));
		MasterLog(INFO, "Unprec Actual Relative Residuum = %16.8e", normdiff/psi_norm);
		ASSERT_LT( (normdiff/psi_norm), 3.8e-1);

	}

}


TEST_F(VCycleEOTesting, TestLevelSetup2Level)
{

   IndexArray latdims={{16,16,16,16}};


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


	VCycleRecursiveQPhiXEO v_cycle(v_params,mg_levels);
	VCycleRecursiveQPhiX v_cycle_unprec(v_params,mg_levels_unprec);

	FGMRESParams fine_solve_params;
	fine_solve_params.MaxIter=200;
	fine_solve_params.RsdTarget=1.0e-13;
	fine_solve_params.VerboseP = true;
	fine_solve_params.NKrylov = 5;

	UnprecFGMRESSolverQPhiXWrapper fgmres_wrapper(std::make_shared<const FGMRESSolverQPhiX>(*M_fine_prec,fine_solve_params, &v_cycle),
			M_fine_prec);

	FGMRESSolverQPhiX fgmres_unprec(*M_fine_unprec_full,fine_solve_params, &v_cycle_unprec);


	MasterLog(INFO, "*** Recursive VCycle Structure + Solver Created");
	const LatticeInfo& fine_info = getFineInfo();

	QPhiXSpinor psi_in(fine_info);
	QPhiXSpinor chi_out(fine_info);
	QPhiXSpinor chi_out_unprec(fine_info);
	Gaussian(psi_in);
	ZeroVec(chi_out);
	ZeroVec(chi_out_unprec);

	double psi_norm = sqrt(Norm2Vec(psi_in));
	MasterLog(INFO, "psi_in has norm = %16.8e",psi_norm);

	MasterLog(INFO, "About to do preconditioned solve");
	double stime_prec = omp_get_wtime();
	LinearSolverResults res=fgmres_wrapper(chi_out, psi_in);
	double etime_prec = omp_get_wtime();

	MasterLog(INFO, "About to do unpreconditioned solve");
	double stime_unprec = omp_get_wtime();
	LinearSolverResults res2=fgmres_unprec(chi_out_unprec, psi_in);
	double etime_unprec = omp_get_wtime();

	// Compute true residuum

	QPhiXSpinor Ax(fine_info);
	{
		(*M_fine_unprec_full)(Ax,chi_out_unprec,LINOP_OP);
		double diff = sqrt(XmyNorm2Vec(Ax,psi_in));
		double diff_rel = diff/psi_norm;
		MasterLog(INFO,"Unprec Solution: || b - A x || = %16.8e", diff);
		MasterLog(INFO,"Unprec Solution: || b - A x ||/ || b || = %16.8e",diff_rel);
		ASSERT_EQ( res.resid_type, RELATIVE);
		ASSERT_LT( res.resid, 1.0e-13);
		ASSERT_LT( toDouble(diff_rel), 1.0e-13);
	}
	{
		(*M_fine_unprec_full)(Ax,chi_out,LINOP_OP);
		double diff = sqrt(XmyNorm2Vec(Ax,psi_in));
		double diff_rel = diff/psi_norm;
		MasterLog(INFO,"Prec Solution: || b - A x || = %16.8e", diff);
		MasterLog(INFO,"Prec Solution: || b - A x ||/ || b || = %16.8e",diff_rel);

		ASSERT_EQ( res.resid_type, RELATIVE);
		ASSERT_LT( res.resid, 1.0e-13);
		ASSERT_LT( toDouble(diff_rel), 1.0e-13);
	}

	MasterLog(INFO, "Unprec Solve Took: %16.8e sec", etime_unprec-stime_unprec);
	MasterLog(INFO, "Prec Solve Took: %16.8e sec", etime_prec-stime_prec);

}

#if 0
TEST(TestQPhiXVCycle, TestVCycleSolve)
{
#if 0
	IndexArray latdims={{8,8,8,8}};
	IndexArray blockdims = {{2,2,2,2}};

	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	IndexArray node_orig=NodeInfo().NodeCoords();
	for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];


	float m_q = 0.1;
	float c_sw = 1.25;

	int t_bc=-1; // Antiperiodic t BCs


	multi1d<LatticeColorMatrix> u(Nd);
	for(int mu=0; mu < Nd; ++mu) {
		gaussian(u[mu]);
		reunit(u[mu]);
	}

  // Move to QPhiX space:
  LatticeInfo fine_info(node_orig,latdims,4,3,NodeInfo());


  // Create Fine linear operator -- need this for checking
  QDPWilsonCloverLinearOperator M_qdp(m_q, c_sw, t_bc,u);

  // Create QPhiX Fine Linear Operators
  QPhiXWilsonCloverLinearOperator M(fine_info, m_q, c_sw,t_bc, u);

  std::shared_ptr<QPhiXWilsonCloverLinearOperatorF> M_f =
     std::make_shared<QPhiXWilsonCloverLinearOperatorF>(fine_info, m_q, c_sw,t_bc, u);

  SetupParams level_setup_params = {
      2,       // Number of levels
      {8},   // Null vecs on L0, L1
      {
          {2,2,2,2},  // Block Size from L0->L1
      },
      {500},          // Max Nullspace Iters
      {5e-6},        // Nullspace Target Resid
      {false}
  };

  QPhiXMultigridLevels mg_levels;
  SetupQPhiXMGLevels(level_setup_params, mg_levels, M_f);

  // WE NOW HAVE: M_fine, M_coarse and the information to affect Intergrid
  // Transfers.

  // Set up the PreSmoother & Post Smootehr (on the fine level)
  MRSolverParams presmooth;
  presmooth.MaxIter=2;
  presmooth.RsdTarget = 0.1;
  presmooth.Omega = 1.1;
  presmooth.VerboseP = true;

  // MR Smoother holds only references
  MRSmootherQPhiXF the_smoother(*(mg_levels.fine_level.M),presmooth);


  // Set up the CoarseSolver
  FGMRESParams coarse_solve_params;
  coarse_solve_params.MaxIter=200;
  coarse_solve_params.RsdTarget=0.1;
  coarse_solve_params.VerboseP = true;
  coarse_solve_params.NKrylov = 10;
  FGMRESSolverCoarse bottom_solver(*(mg_levels.coarse_levels[0].M),coarse_solve_params);



	{
		LinearSolverParamsBase vcycle_params;
		vcycle_params.MaxIter=500;
		vcycle_params.RsdTarget = 1.0e-7;
		vcycle_params.VerboseP = true;

		// info my_blocks, and vecs can probably be collected in a 'Transfer' class
		VCycleQPhiXCoarse2 vcycle( *(mg_levels.fine_level.info),
		    *(mg_levels.coarse_levels[0].info),
		      mg_levels.fine_level.blocklist,
		      mg_levels.fine_level.null_vecs,
		      *(mg_levels.fine_level.M),
		      the_smoother,
		        the_smoother,
		        bottom_solver,
		        vcycle_params);

		// Now need to do the coarse test
		QPhiXSpinor psi_in(fine_info);
		QPhiXSpinor chi_out(fine_info);
		Gaussian(psi_in);
		ZeroVec(chi_out);
		double psi_norm = sqrt(Norm2Vec(psi_in));

		MasterLog(INFO,"psi_in has norm = %16.8e",psi_norm);
		LinearSolverResults res = vcycle(chi_out,psi_in);

    // Compute true residuum
    QPhiXSpinor Ax(fine_info);
    M(Ax,chi_out,LINOP_OP);
    double diff = sqrt(XmyNorm2Vec(psi_in,Ax));
    double diff_rel = diff/psi_norm;
  	MasterLog(INFO,"|| b - A x || = %16.8e", diff);
		MasterLog(INFO,"|| b - A x ||/ || b || = %16.8e",diff_rel);
		ASSERT_EQ( res.resid_type, RELATIVE);
		ASSERT_LT( res.resid, 5.0e-7);
		ASSERT_LT( diff_rel, 5.0e-7);


	}

#endif
}


TEST(TestQPhiXVCycle, TestVCyclePrec)
{
#if 0
	IndexArray latdims={{8,8,8,8}};
	IndexArray blockdims = {{2,2,2,2}};

	initQDPXXLattice(latdims);
	IndexArray node_orig=NodeInfo().NodeCoords();
		for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

	float m_q = 0.1;
	float c_sw = 1.25;

	int t_bc=-1; // Antiperiodic t BCs


	multi1d<LatticeColorMatrix> u(Nd);
	for(int mu=0; mu < Nd; ++mu) {
	  gaussian(u[mu]);
	  reunit(u[mu]);
		u[mu]=1.0;
	}

	 // Move to QPhiX space:
	  LatticeInfo fine_info(node_orig,latdims,4,3,NodeInfo());

	  // Create QPhiX Fine Linear Operators
	  QPhiXWilsonCloverLinearOperator M(fine_info, m_q, c_sw,t_bc, u);

	  std::shared_ptr<QPhiXWilsonCloverLinearOperatorF> M_f =
	     std::make_shared<QPhiXWilsonCloverLinearOperatorF>(fine_info, m_q, c_sw,t_bc, u);

	  SetupParams level_setup_params = {
	      2,       // Number of levels
	      {8},   // Null vecs on L0, L1
	      {
	          {2,2,2,2},  // Block Size from L0->L1
	      },
	      {500},          // Max Nullspace Iters
	      {5e-6},        // Nullspace Target Resid
	      {false}
	  };

	  QPhiXMultigridLevels mg_levels;
	  SetupQPhiXMGLevels(level_setup_params, mg_levels, M_f);

	MRSolverParams presmooth;
	presmooth.MaxIter=4;
	presmooth.RsdTarget = 0.1;
	presmooth.Omega = 1.1;
	presmooth.VerboseP = true;

	MRSmootherQPhiXF pre_smoother(*M_f,presmooth);

	MRSolverParams postsmooth;
	postsmooth.MaxIter = 4;
	postsmooth.RsdTarget = 0.1;
	postsmooth.Omega = 1.1;
	postsmooth.VerboseP = true;

	MRSmootherQPhiXF post_smoother(*M_f,postsmooth);

	FGMRESParams coarse_solve_params;
	coarse_solve_params.MaxIter=200;
	coarse_solve_params.RsdTarget=0.1;
	coarse_solve_params.VerboseP = false;
	coarse_solve_params.NKrylov = 10;
	FGMRESSolverCoarse bottom_solver(*(mg_levels.coarse_levels[0].M),coarse_solve_params);

	LinearSolverParamsBase vcycle_params;
	vcycle_params.MaxIter=2;
	vcycle_params.RsdTarget =0.1;
	vcycle_params.VerboseP = true;

  // info my_blocks, and vecs can probably be collected in a 'Transfer' class
    VCycleQPhiXCoarse2 vcycle( *(mg_levels.fine_level.info),
        *(mg_levels.coarse_levels[0].info),
          mg_levels.fine_level.blocklist,
          mg_levels.fine_level.null_vecs,
          *(mg_levels.fine_level.M),
          pre_smoother,
          post_smoother,
          bottom_solver,
          vcycle_params);


	FGMRESParams fine_solve_params;
	fine_solve_params.MaxIter=10000;
	fine_solve_params.RsdTarget=1.0e-13;
	fine_solve_params.VerboseP = true;
	fine_solve_params.NKrylov = 4;
	FGMRESSolverQPhiX FGMRESOuter(M,fine_solve_params, &vcycle);

	// Now need to do the coarse test
  // Now need to do the coarse test
  QPhiXSpinor psi_in(fine_info);
  QPhiXSpinor chi_out(fine_info);
  Gaussian(psi_in);
  ZeroVec(chi_out);
  double psi_norm = sqrt(Norm2Vec(psi_in));
  MasterLog(INFO, "psi_in has norm = %16.8e",psi_norm);

  LinearSolverResults res=FGMRESOuter(chi_out, psi_in);

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
#endif
}


TEST(TestQPhiXVCycle, TestVCyclePrecEOPrec)
{
#if 0
	IndexArray latdims={{8,8,8,8}};
	IndexArray blockdims = {{2,2,2,2}};

	initQDPXXLattice(latdims);
	IndexArray node_orig=NodeInfo().NodeCoords();
		for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

	float m_q = 0.1;
	float c_sw = 1.25;

	int t_bc=-1; // Antiperiodic t BCs


	multi1d<LatticeColorMatrix> u(Nd);
	for(int mu=0; mu < Nd; ++mu) {
	  gaussian(u[mu]);
	  reunit(u[mu]);
		u[mu]=1.0;
	}

	 // Move to QPhiX space:
	  LatticeInfo fine_info(node_orig,latdims,4,3,NodeInfo());

	  // Create QPhiX Fine Linear Operators
	  std::shared_ptr<const QPhiXWilsonCloverEOLinearOperator> M =
			  std::make_shared<const QPhiXWilsonCloverEOLinearOperator>(fine_info, m_q, c_sw,t_bc, u);

	  std::shared_ptr<QPhiXWilsonCloverEOLinearOperatorF> M_f =
	     std::make_shared<QPhiXWilsonCloverEOLinearOperatorF>(fine_info, m_q, c_sw,t_bc, u);

	  SetupParams level_setup_params = {
	      2,       // Number of levels
	      {8},   // Null vecs on L0, L1
	      {
	          {2,2,2,2},  // Block Size from L0->L1
	      },
	      {500},          // Max Nullspace Iters
	      {5e-6},        // Nullspace Target Resid
	      {false}
	  };

	  QPhiXMultigridLevelsEO mg_levels;
	  SetupQPhiXMGLevels(level_setup_params, mg_levels, M_f);

	MRSolverParams presmooth;
	presmooth.MaxIter=4;
	presmooth.RsdTarget = 0.1;
	presmooth.Omega = 1.1;
	presmooth.VerboseP = true;

	MRSmootherQPhiXF pre_smoother(*M_f,presmooth);

	MRSolverParams postsmooth;
	postsmooth.MaxIter = 4;
	postsmooth.RsdTarget = 0.1;
	postsmooth.Omega = 1.1;
	postsmooth.VerboseP = true;

	MRSmootherQPhiXF post_smoother(*M_f,postsmooth);

	FGMRESParams coarse_solve_params;
	coarse_solve_params.MaxIter=200;
	coarse_solve_params.RsdTarget=0.1;
	coarse_solve_params.VerboseP = false;
	coarse_solve_params.NKrylov = 10;
	FGMRESSolverCoarse bottom_solver(*(mg_levels.coarse_levels[0].M),coarse_solve_params);

	LinearSolverParamsBase vcycle_params;
	vcycle_params.MaxIter=2;
	vcycle_params.RsdTarget =0.1;
	vcycle_params.VerboseP = true;

  // info my_blocks, and vecs can probably be collected in a 'Transfer' class
    VCycleQPhiXCoarseEO2 vcycle( *(mg_levels.fine_level.info),
        *(mg_levels.coarse_levels[0].info),
          mg_levels.fine_level.blocklist,
          mg_levels.fine_level.null_vecs,
          *(mg_levels.fine_level.M),
          pre_smoother,
          post_smoother,
          bottom_solver,
          vcycle_params);


	FGMRESParams fine_solve_params;
	fine_solve_params.MaxIter=10000;
	fine_solve_params.RsdTarget=1.0e-13;
	fine_solve_params.VerboseP = true;
	fine_solve_params.NKrylov = 4;

	// Create even odd preconditioned FGMRES
	std::shared_ptr<const FGMRESSolverQPhiX> FGMRES=std::make_shared<const FGMRESSolverQPhiX>(*M, fine_solve_params,&vcycle);
	//std::shared_ptr<const FGMRESSolverQPhiX> FGMRES=std::make_shared<const FGMRESSolverQPhiX>(*M, fine_solve_params,nullptr);

	// Wrap in source prep and solution recreation
	UnprecFGMRESSolverQPhiXWrapper FGMRESWrapper(FGMRES, M);

	// Now need to do the coarse test
  // Now need to do the coarse test
  QPhiXSpinor psi_in(fine_info);
  QPhiXSpinor chi_out(fine_info);
  Gaussian(psi_in);
  ZeroVec(chi_out);
  double psi_norm = sqrt(Norm2Vec(psi_in));
  MasterLog(INFO, "psi_in has norm = %16.8e",psi_norm);

  LinearSolverResults res=FGMRESWrapper(chi_out, psi_in);

  // Compute true residuum
  QPhiXSpinor Ax(fine_info);
  (*M).unprecOp(Ax,chi_out,LINOP_OP);
  double diff = sqrt(XmyNorm2Vec(psi_in,Ax));
  double diff_rel = diff/psi_norm;
  MasterLog(INFO,"|| b - A x || = %16.8e", diff);
  MasterLog(INFO,"|| b - A x ||/ || b || = %16.8e",diff_rel);

	ASSERT_EQ( res.resid_type, RELATIVE);
	ASSERT_LT( res.resid, 1.0e-13);
	ASSERT_LT( toDouble(diff_rel), 1.0e-13);
#endif
}


TEST(TestQPhiXVCycle, TestVCycle2Level)
{
#if 0
	IndexArray latdims={{8,8,8,8}};
	IndexArray blockdims = {{2,2,2,2}};


	initQDPXXLattice(latdims);

	IndexArray node_orig=NodeInfo().NodeCoords();
		for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

	float m_q = 0.1;
	float c_sw = 1.25;

	int t_bc=-1; // Antiperiodic t BCs


	multi1d<LatticeColorMatrix> u(Nd);
	for(int mu=0; mu < Nd; ++mu) {

		gaussian(u[mu]);
		reunit(u[mu]);
	}

  // Move to QPhiX space:
   LatticeInfo fine_info(node_orig,latdims,4,3,NodeInfo());

   // Create QPhiX Fine Linear Operators
   QPhiXWilsonCloverLinearOperator M(fine_info, m_q, c_sw,t_bc, u);

   std::shared_ptr<QPhiXWilsonCloverLinearOperatorF> M_f =
      std::make_shared<QPhiXWilsonCloverLinearOperatorF>(fine_info, m_q, c_sw,t_bc, u);

   SetupParams level_setup_params = {
       3,       // Number of levels
       {8,8},   // Null vecs on L0, L1
       {
           {2,2,2,2},
           {2,2,2,2}// Block Size from L0->L1
       },
       {500,500},          // Max Nullspace Iters
       {5e-6,5e-6},        // Nullspace Target Resid
       {false,false}
   };

   QPhiXMultigridLevels mg_levels;
   SetupQPhiXMGLevels(level_setup_params, mg_levels, M_f);


	MasterLog(INFO, "Creating Level 2 Vcycle");

	MasterLog(INFO, " ...creating L1 PreSmoother");
	MRSolverParams presmooth_12_params;
	presmooth_12_params.MaxIter=4;
	presmooth_12_params.RsdTarget = 0.1;
	presmooth_12_params.Omega = 1.1;
	presmooth_12_params.VerboseP = true;
	MRSmootherCoarse pre_smoother_12(*(mg_levels.coarse_levels[0].M), presmooth_12_params);

	MasterLog(INFO," ...creating Bottom (L2) Solver");
	FGMRESParams l2_solve_params;
	l2_solve_params.MaxIter=200;
	l2_solve_params.RsdTarget=0.1;
	l2_solve_params.VerboseP = false;
	l2_solve_params.NKrylov = 10;
	FGMRESSolverCoarse l2_solver(*(mg_levels.coarse_levels[1].M),
	    l2_solve_params,nullptr); // Bottom solver, no preconditioner

	MasterLog(INFO, " ...creating L1 PostSmoother");
	MRSolverParams postsmooth_12_params;
	postsmooth_12_params.MaxIter=4;
	postsmooth_12_params.RsdTarget = 0.1;
	postsmooth_12_params.Omega = 1.1;
	postsmooth_12_params.VerboseP = true;
	MRSmootherCoarse post_smoother_12(*(mg_levels.coarse_levels[0].M), postsmooth_12_params);

	MasterLog(INFO, " ... creating L1 -> L2 VCycle");
	LinearSolverParamsBase vcycle_12_params;
	vcycle_12_params.MaxIter=1;
	vcycle_12_params.RsdTarget =0.1;
	vcycle_12_params.VerboseP = true;


	VCycleCoarse vcycle12( *(mg_levels.coarse_levels[1].info),
	      mg_levels.coarse_levels[0].blocklist,
	      mg_levels.coarse_levels[0].null_vecs,
	      *(mg_levels.coarse_levels[0].M),
	      pre_smoother_12,
	      post_smoother_12,
	      l2_solver,
	      vcycle_12_params);


	MasterLog(INFO, "  ... creating L0 PreSmoother");
	MRSolverParams presmooth_01_params;
	presmooth_01_params.MaxIter=4;
	presmooth_01_params.RsdTarget = 0.1;
	presmooth_01_params.Omega = 1.1;
	presmooth_01_params.VerboseP = true;
	MRSmootherQPhiXF pre_smoother_01(*(mg_levels.fine_level.M),
	    presmooth_01_params);

	MasterLog(INFO,"  ... creating L1 FGMRES Solver -- preconditioned with L1->L2 Vcycle");
	FGMRESParams l1_solve_params;
	l1_solve_params.MaxIter=200;
	l1_solve_params.RsdTarget=0.1;
	l1_solve_params.VerboseP = false;
	l1_solve_params.NKrylov = 10;
	FGMRESSolverCoarse l1_solver(*(mg_levels.coarse_levels[0].M),
	    l1_solve_params,&vcycle12); // Bottom solver, no preconditioner

	MasterLog(INFO,"  ... creating L0 Post Smoother");
	MRSolverParams postsmooth_01_params;
	postsmooth_01_params.MaxIter=4;
	postsmooth_01_params.RsdTarget = 0.1;
	postsmooth_01_params.Omega = 1.1;
	postsmooth_01_params.VerboseP = true;
	MRSmootherQPhiXF post_smoother_01(*(mg_levels.fine_level.M), postsmooth_01_params);

	MasterLog(INFO," ... creating L0 -> L1 Vcycle Preconitioner ");
	LinearSolverParamsBase vcycle_01_params;
	vcycle_01_params.MaxIter=1;
	vcycle_01_params.RsdTarget =0.1;
	vcycle_01_params.VerboseP = true;

	VCycleQPhiXCoarse2 vcycle_01( *(mg_levels.fine_level.info),
	    *(mg_levels.coarse_levels[0].info),
	    mg_levels.fine_level.blocklist,
	    mg_levels.fine_level.null_vecs,
	    *(mg_levels.fine_level.M),
	    pre_smoother_01,
	    post_smoother_01,
	    l1_solver,
	    vcycle_01_params);




	MasterLog(INFO,"Creating Outer Solver with L0->L1 VCycle Preconditioner");
	FGMRESParams fine_solve_params;
	fine_solve_params.MaxIter=200;
	fine_solve_params.RsdTarget=1.0e-13;
	fine_solve_params.VerboseP = true;
	fine_solve_params.NKrylov = 5;
	FGMRESSolverQPhiX FGMRESOuter(M,fine_solve_params, &vcycle_01);

	MasterLog(INFO,"*** Recursive VCycle Structure + Solver Created");

	QPhiXSpinor psi_in(fine_info);
	QPhiXSpinor chi_out(fine_info);
	Gaussian(psi_in);
	ZeroVec(chi_out);
	double psi_norm = sqrt(Norm2Vec(psi_in));
	MasterLog(INFO, "psi_in has norm = %16.8e",psi_norm);

	LinearSolverResults res=FGMRESOuter(chi_out, psi_in);

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
#endif
}

#endif
int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

void VCycleEOTesting::SetUp()
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

	M_fine=std::make_shared<QPhiXWilsonCloverEOLinearOperatorF>(info,m_q, c_sw, t_bc,u);
	M_fine_prec=std::make_shared<QPhiXWilsonCloverEOLinearOperator>(info,m_q, c_sw, t_bc,u);

	QDPIO::cout << "Calling EO Level setup" << std::endl;
	SetupQPhiXMGLevels(level_setup_params,
	 			  	  mg_levels,
					  M_fine);

	M_fine_unprec=std::make_shared<QPhiXWilsonCloverLinearOperatorF>(info,m_q, c_sw, t_bc,u);
	M_fine_unprec_full=std::make_shared<QPhiXWilsonCloverLinearOperator>(info,m_q, c_sw, t_bc,u);
	QDPIO::cout << "Calling EO Level setup" << std::endl;
	SetupQPhiXMGLevels(level_setup_params,
			mg_levels_unprec,
			M_fine_unprec);

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


