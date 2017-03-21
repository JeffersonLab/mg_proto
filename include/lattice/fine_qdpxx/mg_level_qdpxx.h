/*
 * mg_level_qdpxx.h
 *
 *  Created on: Mar 15, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_FINE_QDPXX_MG_LEVEL_QDPXX_H_
#define INCLUDE_LATTICE_FINE_QDPXX_MG_LEVEL_QDPXX_H_

#include "lattice/linear_operator.h"
#include "lattice/solver.h"
#include "lattice/coarse/block.h"
#include "lattice/mg_level_coarse.h"
#include "lattice/linear_operator.h"
#include "lattice/fine_qdpxx/wilson_clover_linear_operator.h"

#include "lattice/mr_params.h"
#include "lattice/fgmres_common.h"

#include "lattice/fine_qdpxx/vcycle_qdpxx_coarse.h"
#include "lattice/fine_qdpxx/invfgmres.h"
#include "lattice/fine_qdpxx/invmr.h"
#include "lattice/fine_qdpxx/invbicgstab.h"
#include "lattice/coarse/vcycle_coarse.h"
#include "lattice/invfgmres_coarse.h"
#include "lattice/invbicgstab_coarse.h"
#include "lattice/invmr_coarse.h"

#include "utils/print_utils.h"
#include <memory>

using std::vector;
using std::shared_ptr;
using std::make_shared;


		namespace MG {

struct MGLevelQDPXX {
	std::shared_ptr<const LatticeInfo> info;
	QDP::multi1d<QDP::LatticeFermion> null_vecs;
	std::shared_ptr< const LinearSolver< QDP::LatticeFermion,QDP::multi1d<QDP::LatticeColorMatrix> > > null_solver;           // Solver for NULL on this level
	std::shared_ptr< const QDPWilsonCloverLinearOperator > M;
	std::vector<Block> blocklist;

	~MGLevelQDPXX() {}
};

struct MultigridLevels {
	int n_levels;
	MGLevelQDPXX fine_level;
	std::vector<MGLevelCoarse> coarse_levels;
};


struct VCycleParams {
	// Pre Smoother Params
	MRSolverParams pre_smoother_params;
	FGMRESParams bottom_solver_params;
	MRSolverParams post_smoother_params;
	LinearSolverParamsBase cycle_params;
};

struct SetupParams {
	int n_levels;
	std::vector<int> n_vecs;
	IndexArray local_lattice_size;
	std::vector< IndexArray > block_sizes;
	std::vector< int > null_solver_max_iter;
	std::vector< double > null_solver_rsd_target;

};



void SetupQDPXXToCoarse(const SetupParams& p, std::shared_ptr<const QDPWilsonCloverLinearOperator> M_fine,
						MGLevelQDPXX& fine_level, MGLevelCoarse& coarse_level);

void SetupCoarseToCoarse(const SetupParams& p, std::shared_ptr< const CoarseWilsonCloverLinearOperator > M_fine, int fine_level_id,
						MGLevelCoarse& fine_level, MGLevelCoarse& coarse_level);

void SetupMGLevels(const SetupParams& p, MultigridLevels& mg_levels,
			std::shared_ptr<const QDPWilsonCloverLinearOperator> M_fine);

#if 1
class VCycleRecursiveQDPXX :  public LinearSolver<LatticeFermion, multi1d<LatticeColorMatrix> >
{
public:
	VCycleRecursiveQDPXX( const std::vector<VCycleParams>& vcycle_params,
						  const MultigridLevels& mg_levels ) : _vcycle_params(vcycle_params), _mg_levels(mg_levels) {

		MasterLog(INFO, "Constructing Recursive VCycle");

		if ( vcycle_params.size() != mg_levels.n_levels-1 ) {
			MasterLog(ERROR, "Params provided for %d levels, but with %d levels %d are needed",
						vcycle_params.size(), mg_levels.n_levels, mg_levels.n_levels-1);
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

			if( coarse_idx == n_levels-2) {
				MasterLog(INFO, "Creating FGRMRES SOlver on Level %d", coarse_idx);

				// Bottom level There is only a bottom solver.
				_bottom_solver[coarse_idx] = std::make_shared< const FGMRESSolverCoarse >(
						*(_mg_levels.coarse_levels[coarse_idx].M),
											_vcycle_params[coarse_idx].bottom_solver_params,nullptr);

			}
			else{
#if 1
				MasterLog(INFO, "Creating PreSmoother on Level %d using VCycleParams[%d]", coarse_idx+1, coarse_idx+1);
				_coarse_presmoother[coarse_idx] = std::make_shared<const MRSmootherCoarse >(*(_mg_levels.coarse_levels[coarse_idx].M),
							_vcycle_params[coarse_idx+1].pre_smoother_params);

				MasterLog(INFO, "Creating PreSmoother on Level %d using VCycleParams[%d]", coarse_idx+1, coarse_idx+1);
				_coarse_postsmoother[coarse_idx] = std::make_shared<const MRSmootherCoarse >(*(_mg_levels.coarse_levels[coarse_idx].M),
						_vcycle_params[coarse_idx+1].post_smoother_params);

				MasterLog(INFO, "Creating VCycle Between Levels: %d -> %d using VCycleParams[%d]", coarse_idx+1, coarse_idx+2,coarse_idx+1);
				_coarse_vcycle[coarse_idx] = std::make_shared< const VCycleCoarse >(
						(*(_mg_levels.coarse_levels[coarse_idx+1].info)),
						(_mg_levels.coarse_levels[coarse_idx].blocklist),
						(_mg_levels.coarse_levels[coarse_idx].null_vecs),
						(*(_mg_levels.coarse_levels[coarse_idx].M)),
						(*(_coarse_presmoother[coarse_idx])),
						(*(_coarse_postsmoother[coarse_idx])),
						(*(_bottom_solver[coarse_idx+1])),
						(_vcycle_params[coarse_idx+1].cycle_params));

				MasterLog(INFO, "Creating Bottom Solver For level: %d, using VCycle Preconditioner from level %d", coarse_idx+1, coarse_idx+1);
				_bottom_solver[coarse_idx] = std::make_shared< const FGMRESSolverCoarse >(
							*(_mg_levels.coarse_levels[coarse_idx].M),
							vcycle_params[coarse_idx].bottom_solver_params,
							_coarse_vcycle[coarse_idx].get());

#endif

			}
		}

		MasterLog(INFO,"Creating Toplevel Smoothers");
		_pre_smoother = std::make_shared< const MRSmoother >(*(_mg_levels.fine_level.M), _vcycle_params[0].pre_smoother_params);
		_post_smoother = std::make_shared< const MRSmoother >(*(_mg_levels.fine_level.M), _vcycle_params[0].post_smoother_params);
		MasterLog(INFO,"Creating Toplevel VCycle");
		_toplevel_vcycle = std::make_shared< const VCycleQDPCoarse2 >((*(_mg_levels.coarse_levels[0].info)),  // Coarse info for first coarse level
																(_mg_levels.fine_level.blocklist),   // Block List
																(_mg_levels.fine_level.null_vecs),   // Null vecs
																(*(_mg_levels.fine_level.M)),           // LinOp
																(*_pre_smoother),                     //
																(*_post_smoother),
																(*(_bottom_solver[0])),
																(_vcycle_params[0].cycle_params));



	}

	LinearSolverResults operator()(LatticeFermion& out, const LatticeFermion& in, ResiduumType resid_type = RELATIVE ) const
	{
		LinearSolverResults ret = (*_toplevel_vcycle )( out, in, resid_type );
		return ret;
	}


private:

	const std::vector<VCycleParams> _vcycle_params;
	const MultigridLevels& _mg_levels;

	std::shared_ptr< const Smoother<LatticeFermion,multi1d<LatticeColorMatrix> > > _pre_smoother;
	std::shared_ptr< const Smoother<LatticeFermion,multi1d<LatticeColorMatrix> > > _post_smoother;
	std::shared_ptr< const VCycleQDPCoarse2 > _toplevel_vcycle;

	std::vector< std::shared_ptr< const Smoother< CoarseSpinor, CoarseGauge > > >       _coarse_presmoother;
	std::vector< std::shared_ptr< const Smoother< CoarseSpinor, CoarseGauge > > >       _coarse_postsmoother;
	std::vector< std::shared_ptr< const LinearSolver< CoarseSpinor, CoarseGauge > > >   _coarse_vcycle;
	std::vector< std::shared_ptr< const LinearSolver< CoarseSpinor, CoarseGauge > > >   _bottom_solver;

};
#endif


}



#endif /* INCLUDE_LATTICE_FINE_QDPXX_MG_LEVEL_QDPXX_H_ */
