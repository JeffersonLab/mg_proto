/*
 * vcycle_recursive_qphix.h
 *
 *  Created on: Mar 21, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_QPHIX_VCYCLE_RECURSIVE_QPHIX_H_
#define INCLUDE_LATTICE_QPHIX_VCYCLE_RECURSIVE_QPHIX_H_

#include <vector>
#include <memory>
#include <MG_config.h>
#include "lattice/solver.h"
#include "lattice/qphix/vcycle_qphix_coarse.h"
#include "lattice/coarse/vcycle_coarse.h"
#include "lattice/qphix/mg_level_qphix.h"
#include "lattice/qphix/invmr_qphix.h"
#include "lattice/qphix/invfgmres_qphix.h"
#include "lattice/qphix/invbicgstab_qphix.h"
#include "lattice/coarse/invmr_coarse.h"
#include "lattice/coarse/invfgmres_coarse.h"
#include "lattice/coarse/coarse_deflation.h"

using namespace QDP;

namespace MG {
template<typename QPhiXMGLevelsT, typename Fine2CoarseVCycleT, typename Coarse2CoarseVCycleT, typename FineSmootherT, typename CoarseSmootherT, typename BottomSolverT>
class VCycleRecursiveQPhiXT :  public LinearSolver<QPhiXSpinor, QPhiXGauge>, public LinearSolver<QPhiXSpinorF, QPhiXGaugeF>

{
public:
	static std::vector<VCycleParams> hackVcycle(const std::vector<VCycleParams>& vcycle_params0) {
		std::vector<VCycleParams> vcycle_params(vcycle_params0);

#ifdef MG_HACK_QPHIX_TRANSFER
		// Hack vcycle_params
		VCycleParams dummy_vcycle_params;
		dummy_vcycle_params.pre_smoother_params.MaxIter=0;
		dummy_vcycle_params.pre_smoother_params.RsdTarget=1.0;
		dummy_vcycle_params.pre_smoother_params.VerboseP = false;
		dummy_vcycle_params.pre_smoother_params.Omega = 1.0;

		dummy_vcycle_params.post_smoother_params.MaxIter=0;
		dummy_vcycle_params.post_smoother_params.RsdTarget=1.0;
		dummy_vcycle_params.post_smoother_params.VerboseP = false;
		dummy_vcycle_params.post_smoother_params.Omega = 1.0;

		dummy_vcycle_params.bottom_solver_params.MaxIter= 1;
		dummy_vcycle_params.bottom_solver_params.NKrylov = 1;
		dummy_vcycle_params.bottom_solver_params.RsdTarget= 0.1;
		dummy_vcycle_params.bottom_solver_params.VerboseP = false;

		dummy_vcycle_params.cycle_params.MaxIter=1;
		dummy_vcycle_params.cycle_params.RsdTarget=0.1;
		dummy_vcycle_params.cycle_params.VerboseP = false;
		vcycle_params.insert(vcycle_params.begin(), dummy_vcycle_params);
#endif

		return vcycle_params;
	}

	VCycleRecursiveQPhiXT( const std::vector<VCycleParams>& vcycle_params0,
						  const QPhiXMGLevelsT& mg_levels )  : _vcycle_params(hackVcycle(vcycle_params0)), _mg_levels(mg_levels) {

		MasterLog(INFO, "Constructing Recursive EvenOdd VCycle");

		if ( _vcycle_params.size() != mg_levels.n_levels-1 ) {
			MasterLog(ERROR, "Params provided for %d levels, but with %d levels %d are needed",
						_vcycle_params.size(), mg_levels.n_levels, mg_levels.n_levels-1);
		}
		int n_levels = _mg_levels.n_levels;
		MasterLog(INFO, "There are %d levels", n_levels);

		_bottom_solver.resize(n_levels);  // No bottom level at the toplevel
		_coarse_presmoother.resize(n_levels -1); // No smoothers on bottom level
		_coarse_postsmoother.resize(n_levels -1 );
		_coarse_vcycle.resize(n_levels-1);




		MasterLog(INFO, "Entering Coarse Level Loop");
		for(int coarse_idx=n_levels-2; coarse_idx >= 0; --coarse_idx) {
			MasterLog(INFO, "Coarse_idx=%d",coarse_idx);
			auto this_level_linop = _mg_levels.coarse_levels[coarse_idx].M;

			if( coarse_idx == n_levels-2) {
				MasterLog(INFO, "Creating FGRMRES Solver Wrapper on Level %d", coarse_idx);

				// Bottom level There is only a bottom solver.
				_bottom_solver[coarse_idx] = std::make_shared< const BottomSolverT >(this_level_linop,_vcycle_params[coarse_idx].bottom_solver_params,nullptr);

				MasterLog(INFO, "Computing deflation for level %d", coarse_idx);
				//CoarseSpinor defl(this_level_linop->GetInfo(), 512);
				//computeDeflation(defl, *this_level_linop);
			}
			else{

				MasterLog(INFO, "Creating PreSmoother on Level %d using VCycleParams[%d]", coarse_idx+1, coarse_idx+1);
				// This becomes a wrapper
				_coarse_presmoother[coarse_idx] = std::make_shared<const CoarseSmootherT>(this_level_linop,_vcycle_params[coarse_idx+1].pre_smoother_params);

				MasterLog(INFO, "Creating PreSmoother on Level %d using VCycleParams[%d]", coarse_idx+1, coarse_idx+1);
				// This becomes a wrapper
				_coarse_postsmoother[coarse_idx] = std::make_shared<const CoarseSmootherT>(this_level_linop, _vcycle_params[coarse_idx+1].post_smoother_params);

				MasterLog(INFO, "Creating VCycle Between Levels: %d -> %d using VCycleParams[%d]", coarse_idx+1, coarse_idx+2,coarse_idx+1);
				_coarse_vcycle[coarse_idx] = std::make_shared< Coarse2CoarseVCycleT >(
						(*(_mg_levels.coarse_levels[coarse_idx+1].info)),
						(_mg_levels.coarse_levels[coarse_idx].blocklist),
						(_mg_levels.coarse_levels[coarse_idx].null_vecs),
						(*(_mg_levels.coarse_levels[coarse_idx].M)),
						(*(_coarse_presmoother[coarse_idx])),
						(*(_coarse_postsmoother[coarse_idx])),
						(*(_bottom_solver[coarse_idx+1])),
						(_vcycle_params[coarse_idx+1].cycle_params));

				MasterLog(INFO, "Creating Bottom Solver For level: %d, using VCycle Preconditioner from level %d", coarse_idx+1, coarse_idx+1);
				// This becomes a wrapper
				_bottom_solver[coarse_idx] = std::make_shared<const BottomSolverT>(this_level_linop,_vcycle_params[coarse_idx].bottom_solver_params,_coarse_vcycle[coarse_idx].get());



			}
		}

		MasterLog(INFO,"Creating Toplevel Smoothers");
		// The QPhiX Smoothers are already preconditioned so leave this as is.
		_pre_smoother = std::make_shared< const FineSmootherT >(_mg_levels.fine_level.M, _vcycle_params[0].pre_smoother_params);
		_post_smoother = std::make_shared< const FineSmootherT >(_mg_levels.fine_level.M, _vcycle_params[0].post_smoother_params);
		MasterLog(INFO,"Creating Toplevel VCycle");
		_toplevel_vcycle = std::make_shared< Fine2CoarseVCycleT >(
		    *(_mg_levels.fine_level.info), // Fine Info
		    *(_mg_levels.coarse_levels[0].info),  // Coarse info for first coarse level
		    (_mg_levels.fine_level.blocklist),   // Block List
		    (_mg_levels.fine_level.null_vecs),   // Null vecs
		    (*(_mg_levels.fine_level.M)),           // LinOp
		    (*_pre_smoother),                     //
		    (*_post_smoother),
		    (*(_bottom_solver[0])),
		    (_vcycle_params[0].cycle_params));

	}

	std::vector<LinearSolverResults> operator()(QPhiXSpinor& out, const QPhiXSpinor& in,
	    ResiduumType resid_type = RELATIVE ) const
		{
			return (*_toplevel_vcycle )( out, in, resid_type );
		}

	std::vector<LinearSolverResults> operator()(QPhiXSpinorF& out, const QPhiXSpinorF& in,
	    ResiduumType resid_type = RELATIVE ) const
		{
			return (*_toplevel_vcycle )( out, in, resid_type );
		}


	const LatticeInfo& GetInfo() const { return *_mg_levels.fine_level.info; }
	const CBSubset& GetSubset() const { return SUBSET_ALL; }
	void SetAntePostSmoother(Smoother<QPhiXSpinorF,QPhiXGaugeF>* s) { _toplevel_vcycle->SetAntePostSmoother(s); }

private:

	const std::vector<VCycleParams> _vcycle_params;
	const QPhiXMGLevelsT& _mg_levels;

	std::shared_ptr< const Smoother<QPhiXSpinorF,QPhiXGaugeF > > _pre_smoother;
	std::shared_ptr< const Smoother<QPhiXSpinorF,QPhiXGaugeF> > _post_smoother;
	std::shared_ptr< Fine2CoarseVCycleT > _toplevel_vcycle;

	std::vector< std::shared_ptr< const Smoother< CoarseSpinor, CoarseGauge > > >       _coarse_presmoother;
	std::vector< std::shared_ptr< const Smoother< CoarseSpinor, CoarseGauge > > >       _coarse_postsmoother;
	std::vector< std::shared_ptr< const LinearSolver< CoarseSpinor, CoarseGauge > > >   _coarse_vcycle;
	std::vector< std::shared_ptr< const LinearSolver< CoarseSpinor, CoarseGauge > > >   _bottom_solver;

};

using VCycleRecursiveQPhiX = VCycleRecursiveQPhiXT<QPhiXMultigridLevels,VCycleQPhiXCoarse2, VCycleCoarse, MRSmootherQPhiXF, MRSmootherCoarse,FGMRESSolverCoarse>;
using VCycleRecursiveQPhiXEO = VCycleRecursiveQPhiXT<QPhiXMultigridLevelsEO,VCycleQPhiXCoarseEO2, VCycleCoarseEO, MRSmootherQPhiXF, UnprecMRSmootherCoarseWrapper, UnprecFGMRESSolverCoarseWrapper>;
using VCycleRecursiveQPhiXEO2 = VCycleRecursiveQPhiXT<QPhiXMultigridLevelsEO,VCycleQPhiXCoarseEO3, VCycleCoarseEO2, FGMRESSmootherQPhiXF, MRSmootherCoarse, UnprecFGMRESSolverCoarseWrapper>;
//using VCycleRecursiveQPhiXEO2 = VCycleRecursiveQPhiXT<QPhiXMultigridLevelsEO,VCycleQPhiXCoarseEO3, VCycleCoarseEO2, MRSmootherQPhiXEOF, MRSmootherCoarse, UnprecFGMRESSolverCoarseWrapper>;
//using VCycleRecursiveQPhiXEO2 = VCycleRecursiveQPhiXT<QPhiXMultigridLevelsEO,VCycleQPhiXCoarseEO3, VCycleCoarseEO2, BiCBStabSmootherQPhiXEOF, MRSmootherCoarse, UnprecFGMRESSolverCoarseWrapper>;


};
#endif /* INCLUDE_LATTICE_QPHIX_VCYCLE_RECURSIVE_QPHIX_H_ */
