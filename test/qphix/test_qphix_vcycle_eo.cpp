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
#include "lattice/qphix/qphix_blas_wrappers.h"

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
	const float m_q=0.00;
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

	const std::shared_ptr<const QPhiXWilsonCloverEOLinearOperatorF> getFineLinOp() const
	{
		return M_fine;
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


TEST_F(VCycleEOTesting, TestQPhiXVCycleEO3)
{
	const LatticeInfo& fine_info = *(mg_levels.fine_level.info);
	const LatticeInfo& coarse_info = *(mg_levels.coarse_levels[0].info);

	MRSolverParams presmooth;
	presmooth.MaxIter=1;
	presmooth.RsdTarget = 0.1;
	presmooth.Omega = 1.1;
	presmooth.VerboseP = false;

	QPhiXWilsonCloverEOLinearOperatorF& FineLinOp = const_cast<QPhiXWilsonCloverEOLinearOperatorF&>( *M_fine);
	// MR Smoother holds only references
	QDPIO::cout << " *********** Creating Wrapped Smoother ********" << std::endl;
	MRSmootherQPhiXF the_wrapped_smoother( FineLinOp, presmooth);

	QDPIO::cout << " *********** Creating Non-Wrapped Wrapped Smoother ********" << std::endl;
	MRSmootherQPhiXEOF the_eo_smoother(FineLinOp,presmooth);

	// Set up the CoarseSolver
	FGMRESParams coarse_solve_params;
	coarse_solve_params.MaxIter=200;
	coarse_solve_params.RsdTarget=0.1;
	coarse_solve_params.VerboseP = false;
	coarse_solve_params.NKrylov = 10;
	auto L0LinOp = mg_levels.coarse_levels[0].M;

	// Bottom solver on level 1 is a wrapper.

	QDPIO::cout << " *********** Creating Non-Wrapped Bottom Solver ********" << std::endl;
	std::shared_ptr<const FGMRESSolverCoarse> eoprec_bottom_solver = std::make_shared<FGMRESSolverCoarse>(*(L0LinOp),coarse_solve_params);

	QDPIO::cout << " *********** Creating Wrapped Bottom Solver ********" << std::endl;
	UnprecFGMRESSolverCoarseWrapper wrapped_bottom_solver( eoprec_bottom_solver, L0LinOp);

	{
		// Create the 2 Level VCycle

		LinearSolverParamsBase vcycle_params;
		vcycle_params.MaxIter=10;                   // Single application
		vcycle_params.RsdTarget =1.0e-2;              // The desired reduction in || r ||
		vcycle_params.VerboseP = false;			   // Verbosity

		// info my_blocks, and vecs can probably be collected in a 'Transfer' class
		QDPIO::cout << "******* CREATING VCYCLE EO2 with wrapped bottom solver" << std::endl;

		VCycleQPhiXCoarseEO2 vcycle_eo2(fine_info,coarse_info,
				mg_levels.fine_level.blocklist,
				mg_levels.fine_level.null_vecs,
				FineLinOp,
				the_wrapped_smoother,
				the_wrapped_smoother,
				wrapped_bottom_solver,
				vcycle_params);

		VCycleQPhiXCoarseEO3 vcycle_eo3(fine_info,coarse_info,
				mg_levels.fine_level.blocklist,
				mg_levels.fine_level.null_vecs,
				FineLinOp,
				the_eo_smoother,
				the_eo_smoother,
				*eoprec_bottom_solver,
				vcycle_params);

		QPhiXSpinor psi_in(fine_info);
		QPhiXSpinor chi_out(fine_info);
		ZeroVec(psi_in,SUBSET_EVEN);
		Gaussian(psi_in, SUBSET_ODD);
		ZeroVec(chi_out);

		double psi_norm = sqrt(Norm2Vec(psi_in,SUBSET_ODD));
		MasterLog(INFO, "psi_in has norm = %16.8e",psi_norm);

		QDPIO::cout << "********* Applying EO2 Vcycle " << std::endl;
		double oldtime = -omp_get_wtime();
		LinearSolverResults res = vcycle_eo2(chi_out,psi_in);
		oldtime += omp_get_wtime();
		QDPIO::cout << res.n_count << " iterations" << std::endl;


		QPhiXSpinor chi_out_eo3(fine_info);
		ZeroVec(chi_out_eo3);
		MasterLog(INFO, "psi_in has norm = %16.8e",psi_norm);
		QDPIO::cout << "********* Applying EO3 Vcycle " << std::endl;

		double newtime= -omp_get_wtime();
		res = vcycle_eo3(chi_out_eo3,psi_in);
		newtime += omp_get_wtime();
		QDPIO::cout << res.n_count << " iterations" << std::endl;

		QPhiXSpinor Ax(fine_info);
		(*M_fine_prec)(Ax,chi_out,LINOP_OP);
		double normdiff = sqrt(XmyNorm2Vec(Ax,psi_in, L0LinOp->GetSubset()));
		MasterLog(INFO, "Actual Relative Residuum After Previous QPhiX EO2 Vcycle = %16.8e", normdiff/psi_norm);
		//ASSERT_LT( (normdiff/psi_norm), 3.8e-1);
		MasterLog(INFO, "Old VCycle Took %16.8e sec", oldtime);


		(*M_fine_prec)(Ax,chi_out_eo3,LINOP_OP);
		normdiff = sqrt(XmyNorm2Vec(Ax,psi_in, L0LinOp->GetSubset()));
		MasterLog(INFO, "Actual Relative Residuum of New QPhiX EO3 Vcycle = %16.8e", normdiff/psi_norm);
		//ASSERT_LT( (normdiff/psi_norm), 3.8e-1);
		MasterLog(INFO, "New VCycle Took %16.8e sec", newtime);

	}

}


TEST_F(VCycleEOTesting, TestVCycleApplyEO2)
{
	const LatticeInfo& fine_info = getCoarseInfo(0);
	const LatticeInfo& coarse_info = getCoarseInfo(1);

	MRSolverParams presmooth;
	presmooth.MaxIter=1;
	presmooth.RsdTarget = 0.1;
	presmooth.Omega = 1.1;
	presmooth.VerboseP = false;

	auto L0LinOp = getCoarseLinOp(0);
	auto L0UnprecOp = getCoarseUnprecLinOp(0);

	// MR Smoother holds only references
	QDPIO::cout << " *********** Creating Wrapped Smoother ********" << std::endl;
	UnprecMRSmootherCoarseWrapper the_wrapped_smoother( std::make_shared<MRSmootherCoarse>(*L0LinOp,presmooth),L0LinOp);

	QDPIO::cout << " *********** Creating Non-Wrapped Wrapped Smoother ********" << std::endl;
	MRSmootherCoarse the_eo_smoother(*L0LinOp,presmooth);

	// Set up the CoarseSolver
	FGMRESParams coarse_solve_params;
	coarse_solve_params.MaxIter=200;
	coarse_solve_params.RsdTarget=0.1;
	coarse_solve_params.VerboseP = false;
	coarse_solve_params.NKrylov = 10;
	auto L1LinOp = getCoarseLinOp(1);


	// Bottom solver on level 1 is a wrapper.

	QDPIO::cout << " *********** Creating Non-Wrapped Bottom Solver ********" << std::endl;
	std::shared_ptr<const FGMRESSolverCoarse> eoprec_bottom_solver = std::make_shared<FGMRESSolverCoarse>(*(L1LinOp),coarse_solve_params);

	QDPIO::cout << " *********** Creating Wrapped Bottom Solver ********" << std::endl;
	UnprecFGMRESSolverCoarseWrapper wrapped_bottom_solver( eoprec_bottom_solver, L1LinOp);

	{
		// Create the 2 Level VCycle

		LinearSolverParamsBase vcycle_params;
		vcycle_params.MaxIter=50;                   // Single application
		vcycle_params.RsdTarget =1.0e-5;              // The desired reduction in || r ||
		vcycle_params.VerboseP = false;			   // Verbosity

		QDPIO::cout << "******* CREATING OLD EO VCYCLE "<< std::endl;
		VCycleCoarseEO vcycle(*(mg_levels.coarse_levels[1].info),
				mg_levels.coarse_levels[0].blocklist,
				mg_levels.coarse_levels[0].null_vecs,
				*(mg_levels.coarse_levels[0].M),
				the_wrapped_smoother,
				the_wrapped_smoother,
				wrapped_bottom_solver,
				vcycle_params);

		// info my_blocks, and vecs can probably be collected in a 'Transfer' class
		QDPIO::cout << "******* CREATING VCYCLE EO2 with wrapped bottom solver" << std::endl;

		VCycleCoarseEO2 vcycle_eo2( *(mg_levels.coarse_levels[1].info),
				mg_levels.coarse_levels[0].blocklist,
				mg_levels.coarse_levels[0].null_vecs,
				*(mg_levels.coarse_levels[0].M),
				the_eo_smoother,
				the_eo_smoother,
				wrapped_bottom_solver,
				vcycle_params);

		QDPIO::cout << "******* CREATING VCYCLE EO2 with unwrapped bottom solver" << std::endl;
		VCycleCoarseEO2 vcycle_eo3( *(mg_levels.coarse_levels[1].info),
				mg_levels.coarse_levels[0].blocklist,
				mg_levels.coarse_levels[0].null_vecs,
				*(mg_levels.coarse_levels[0].M),
				the_eo_smoother,
				the_eo_smoother,
				*eoprec_bottom_solver,
				vcycle_params);


		// Now need to do the coarse test
		CoarseSpinor psi_in(fine_info);
		CoarseSpinor chi_out(fine_info);
		ZeroVec(psi_in,SUBSET_EVEN);
		Gaussian(psi_in, SUBSET_ODD);
		ZeroVec(chi_out);

		double psi_norm = sqrt(Norm2Vec(psi_in,SUBSET_ODD));
		MasterLog(INFO, "psi_in has norm = %16.8e",psi_norm);

		QDPIO::cout << "********* Applying old Vcycle " << std::endl;
		double oldtime = -omp_get_wtime();
		LinearSolverResults res = vcycle(chi_out,psi_in);
		oldtime += omp_get_wtime();
		QDPIO::cout << res.n_count << " iterations" << std::endl;

//		ASSERT_EQ( res.n_count, 1);
//		ASSERT_EQ( res.resid_type, RELATIVE );
	//	ASSERT_LT( res.resid, 3.8e-1 );


		CoarseSpinor chi_out_eo2(fine_info);
		ZeroVec(chi_out_eo2);
		MasterLog(INFO, "psi_in has norm = %16.8e",psi_norm);
		QDPIO::cout << "********* Applying EO2 Vcycle " << std::endl;

		double newtime= -omp_get_wtime();
		res = vcycle_eo2(chi_out_eo2,psi_in);
		newtime += omp_get_wtime();
		QDPIO::cout << res.n_count << " iterations" << std::endl;

	//	ASSERT_EQ( res.n_count, 1);
	//	ASSERT_EQ( res.resid_type, RELATIVE );
	//	ASSERT_LT( res.resid, 3.8e-1 );


		CoarseSpinor chi_out_eo3(fine_info);
		ZeroVec(chi_out_eo3);
		MasterLog(INFO, "psi_in has norm = %16.8e",psi_norm);
		QDPIO::cout << "********* Applying EO3 Vcycle " << std::endl;

		double newtime2= -omp_get_wtime();
		res = vcycle_eo3(chi_out_eo3,psi_in);
		newtime2 += omp_get_wtime();
		QDPIO::cout << res.n_count << " iterations" << std::endl;

	//	ASSERT_EQ( res.n_count, 1);
	// Compute true residuum
		CoarseSpinor Ax(fine_info);
		(*(mg_levels.coarse_levels[0].M))(Ax,chi_out,LINOP_OP);
		double normdiff = sqrt(XmyNorm2Vec(Ax,psi_in, L0LinOp->GetSubset()));
		MasterLog(INFO, "Actual Relative Residuum After Old Vcycle with prec L0 operator = %16.8e", normdiff/psi_norm);
		//ASSERT_LT( (normdiff/psi_norm), 3.8e-1);
		MasterLog(INFO, "Old VCycle Took %16.8e sec", oldtime);


		(*(mg_levels.coarse_levels[0].M))(Ax,chi_out_eo2,LINOP_OP);
		normdiff = sqrt(XmyNorm2Vec(Ax,psi_in, L0LinOp->GetSubset()));
		MasterLog(INFO, "Actual Relative Residuum of New Vcycle with prec L0 operator= %16.8e", normdiff/psi_norm);
		//ASSERT_LT( (normdiff/psi_norm), 3.8e-1);
		MasterLog(INFO, "New VCycle Took %16.8e sec", newtime);

		(*(mg_levels.coarse_levels[0].M))(Ax,chi_out_eo3,LINOP_OP);
		normdiff = sqrt(XmyNorm2Vec(Ax,psi_in, L0LinOp->GetSubset()));
		MasterLog(INFO, "Actual Relative Residuum of Vcycle with unwrapped bottom with prec L0 operator= %16.8e", normdiff/psi_norm);
		//		ASSERT_LT( (normdiff/psi_norm), 3.8e-1);
		MasterLog(INFO, "New Unwrapped BottomVCycle Took %16.8e sec", newtime2);
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

	UnprecFGMRESSolverQPhiXWrapper fgmres_wrapper(M_fine_prec,fine_solve_params, &v_cycle);


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

TEST_F(VCycleEOTesting, TestLevelSetup2LevelEO2)
{
   // V Cycle parametere
	std::vector<VCycleParams> v_params(2);

	for(int level=0; level < mg_levels.n_levels-1; level++) {
		MasterLog(INFO,"Level = %d",level);

		v_params[level].pre_smoother_params.MaxIter=0;
		v_params[level].pre_smoother_params.RsdTarget = 0.1;
		v_params[level].pre_smoother_params.VerboseP = false;
		v_params[level].pre_smoother_params.Omega = 1.1;

		v_params[level].post_smoother_params.MaxIter=9;
		v_params[level].post_smoother_params.RsdTarget = 0.1;
		v_params[level].post_smoother_params.VerboseP = false;
		v_params[level].post_smoother_params.Omega = 1.1;

		v_params[level].bottom_solver_params.MaxIter=25;
		v_params[level].bottom_solver_params.NKrylov = 12;
		v_params[level].bottom_solver_params.RsdTarget= 0.1;
		v_params[level].bottom_solver_params.VerboseP = false;

		v_params[level].cycle_params.MaxIter=1;
		v_params[level].cycle_params.RsdTarget=0.1;
		v_params[level].cycle_params.VerboseP = false;
	}


	VCycleRecursiveQPhiXEO2 v_cycle(v_params,mg_levels);
	VCycleRecursiveQPhiX v_cycle_unprec(v_params,mg_levels_unprec);

	FGMRESParams fine_solve_params;
	fine_solve_params.MaxIter=2000;
	fine_solve_params.RsdTarget=1.0e-13;
	fine_solve_params.VerboseP = true;
	fine_solve_params.NKrylov = 12;

	UnprecFGMRESSolverQPhiXWrapper fgmres_wrapper(M_fine_prec,fine_solve_params, &v_cycle);


	FGMRESSolverQPhiX fgmres_unprec(*M_fine_unprec_full,fine_solve_params, &v_cycle_unprec);

	FGMRESSolverQPhiX totally_unprec(*M_fine_unprec_full,fine_solve_params, nullptr);

	MasterLog(INFO, "*** Recursive VCycle Structure + Solver Created");
	const LatticeInfo& fine_info = getFineInfo();

	QPhiXSpinor psi_in(fine_info);
	QPhiXSpinor chi_out(fine_info);
	QPhiXSpinor chi_out_unprec(fine_info);
	QPhiXSpinor chi_out_totunprec(fine_info);
	Gaussian(psi_in);
	ZeroVec(chi_out);
	ZeroVec(chi_out_unprec);
	ZeroVec(chi_out_totunprec);

	double psi_norm = sqrt(Norm2Vec(psi_in));
	MasterLog(INFO, "psi_in has norm = %16.8e",psi_norm);

	MasterLog(INFO, "About to do VCycleEO3 preconditioned solve");
	double stime_prec = omp_get_wtime();
	LinearSolverResults res=fgmres_wrapper(chi_out, psi_in);
	double etime_prec = omp_get_wtime();

	MasterLog(INFO, "About to do unpreconditioned solve");
	double stime_unprec = omp_get_wtime();
	LinearSolverResults res2=fgmres_unprec(chi_out_unprec, psi_in);
	double etime_unprec = omp_get_wtime();


	MasterLog(INFO, "About to do no-MG unpreconditioned solve");
	double stime_tunprec = omp_get_wtime();
	LinearSolverResults res3=totally_unprec(chi_out_totunprec, psi_in);
	double etime_tunprec = omp_get_wtime();
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

	{
		(*M_fine_unprec_full)(Ax,chi_out_totunprec,LINOP_OP);
		double diff = sqrt(XmyNorm2Vec(Ax,psi_in));
		double diff_rel = diff/psi_norm;
		MasterLog(INFO,"Totally unprec (No MG) Solution: || b - A x || = %16.8e", diff);
		MasterLog(INFO,"Totally unprec (No MG)  Solution: || b - A x ||/ || b || = %16.8e",diff_rel);

		ASSERT_EQ( res.resid_type, RELATIVE);
		ASSERT_LT( res.resid, 1.0e-13);
		ASSERT_LT( toDouble(diff_rel), 1.0e-13);
	}

	MasterLog(INFO, "Tot Unprec Solve Took: %16.8e sec", etime_tunprec-stime_tunprec);
	MasterLog(INFO, "Unprec Solve Took: %16.8e sec", etime_unprec-stime_unprec);
	MasterLog(INFO, "Prec Solve Took: %16.8e sec", etime_prec-stime_prec);

}

int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

void VCycleEOTesting::SetUp()
{
	latdims={{8,8,16,16}};
	initQDPXXLattice(latdims);

	LatticeInfo info(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	IndexArray node_orig=NodeInfo().NodeCoords();
	for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

	// Init Gauge field
	u.resize(n_dim);
	for(int mu=0; mu < n_dim; ++mu) {
		u[mu] = 1;
		QDP::LatticeColorMatrix g;
		gaussian(g);
		u[mu] += 0.1*g;
		reunit(u[mu]);
	}

	QDPIO::cout << "Creating Even Odd M_float" << std::endl;
	M_fine=std::make_shared<QPhiXWilsonCloverEOLinearOperatorF>(info,m_q, c_sw, t_bc,u);

	QDPIO::cout << "Creating Even Odd M_doube" << std::endl;
	M_fine_prec=std::make_shared<QPhiXWilsonCloverEOLinearOperator>(info,m_q, c_sw, t_bc,u);

	M_fine_unprec=std::make_shared<QPhiXWilsonCloverLinearOperatorF>(info,m_q, c_sw, t_bc,u);
	M_fine_unprec_full=std::make_shared<QPhiXWilsonCloverLinearOperator>(info,m_q, c_sw, t_bc,u);


	QDPIO::cout << "********* Calling Unprec Level setup" << std::endl;
	SetupQPhiXMGLevels(level_setup_params,
			mg_levels_unprec,
			M_fine_unprec);

	QDPIO::cout << "******** Calling EO Level setup" << std::endl;
	SetupQPhiXMGLevels(level_setup_params,
	 			  	  mg_levels,
					  M_fine);


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


