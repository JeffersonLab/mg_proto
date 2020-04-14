#include <lattice/coarse/invfgmres_coarse.h>
#include <lattice/coarse/invmr_coarse.h>
#include <lattice/fine_qdpxx/invfgmres_qdpxx.h>
#include <lattice/fine_qdpxx/invmr_qdpxx.h>
#include "lattice/fine_qdpxx/mg_params_qdpxx.h"
#include "lattice/fine_qdpxx/mg_level_qdpxx.h"
#include "lattice/fine_qdpxx/vcycle_recursive_qdpxx.h"
#include "lattice/solver.h"
#include "lattice/fine_qdpxx/vcycle_qdpxx_coarse.h"
#include "lattice/coarse/vcycle_coarse.h"

#include "utils/print_utils.h"

#include <vector>
#include <memory>

using std::shared_ptr;
using std::vector;

namespace MG
{
	VCycleRecursiveQDPXX::VCycleRecursiveQDPXX( const std::vector<VCycleParams>& vcycle_params,
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



			}
		}

		MasterLog(INFO,"Creating Toplevel Smoothers");
		_pre_smoother = std::make_shared< const MRSmootherQDPXX >(*(_mg_levels.fine_level.M), _vcycle_params[0].pre_smoother_params);
		_post_smoother = std::make_shared< const MRSmootherQDPXX >(*(_mg_levels.fine_level.M), _vcycle_params[0].post_smoother_params);
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

	std::vector<LinearSolverResults>
	VCycleRecursiveQDPXX::operator()(LatticeFermion& out, const LatticeFermion& in, ResiduumType resid_type ) const
	{
		return (*_toplevel_vcycle )( out, in, resid_type );
	}

}
