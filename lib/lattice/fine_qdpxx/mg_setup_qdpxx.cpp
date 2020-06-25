/*
 * mg_setup_qdpxx.cpp
 *
 *  Created on: Mar 20, 2017
 *      Author: bjoo
 */

#include <lattice/coarse/invbicgstab_coarse.h>
#include <lattice/coarse/coarse_eo_wilson_clover_linear_operator.h>
#include <lattice/coarse/coarse_wilson_clover_linear_operator.h>

#include <lattice/fine_qdpxx/invbicgstab_qdpxx.h>


#include "qdp.h"
#include "lattice/fine_qdpxx/mg_level_qdpxx.h"
#include "lattice/fine_qdpxx/mg_params_qdpxx.h"
#include "lattice/coarse/coarse_l1_blas.h"
#include "lattice/fine_qdpxx/aggregate_block_qdpxx.h"
#include "utils/print_utils.h"

#include <memory>

using namespace QDP;

namespace MG
{
void SetupQDPXXToCoarse(const SetupParams& p, std::shared_ptr<const QDPWilsonCloverLinearOperator> M_fine,
						MGLevelQDPXX& fine_level, MGLevelCoarse& coarse_level)
{
	SetupQDPXXToCoarseGenerateVecs(p, M_fine, fine_level, coarse_level);
	SetupQDPXXToCoarseVecsIn(p, M_fine, fine_level, coarse_level);
}

void SetupQDPXXToCoarseGenerateVecs(const SetupParams& p, std::shared_ptr<const QDPWilsonCloverLinearOperator> M_fine,
    MGLevelQDPXX& fine_level, MGLevelCoarse& coarse_level)
{
    (void)coarse_level;

    if( ! M_fine ) {
      MasterLog(ERROR, "%s: M_fine is null", __FUNCTION__);
    }
  // Null solver is BiCGStab. Let us make a parameter struct for it.
    fine_level.M = M_fine;
    LinearSolverParamsBase params = p.null_solver_params[0];

    fine_level.null_solver = std::make_shared<BiCGStabSolverQDPXX>(*M_fine, params);

    // Zero RHS
    LatticeFermion b=QDP::zero;

    // Generate the vectors
    int num_vecs = p.n_vecs[0];

    fine_level.null_vecs.resize(num_vecs);
    for(int k=0; k < num_vecs; ++k) {
      gaussian(fine_level.null_vecs[k]);
    }

    for(int k=0; k < num_vecs; ++k) {
       std::vector<LinearSolverResults> res = (*(fine_level.null_solver))(fine_level.null_vecs[k],b, ABSOLUTE);
       assert(res.size() == 1);
       QDPIO::cout << "BiCGStab Solver Took: " << res[0].n_count << " iterations"
          << std::endl;
    }
}

void SetupQDPXXToCoarseVecsIn(const SetupParams& p, std::shared_ptr<const QDPWilsonCloverLinearOperator> M_fine,
            MGLevelQDPXX& fine_level, MGLevelCoarse& coarse_level)
{
  // For sake of form
  IndexArray latdims = {{ QDP::Layout::subgridLattSize()[0],
              QDP::Layout::subgridLattSize()[1],
              QDP::Layout::subgridLattSize()[2],
              QDP::Layout::subgridLattSize()[3] }};


  if ( ! M_fine ) {
    MasterLog(ERROR, "%s M_fine is null", __FUNCTION__);
  }

  if ( ! fine_level.info ) {
    // FIXME: new NodeInfo is never free
    fine_level.info = std::make_shared<LatticeInfo>(latdims,4,3,*new NodeInfo());
  }

  if(! fine_level.M ) {
    fine_level.M = M_fine;
  }

  if( ! fine_level.null_solver ) {
    fine_level.null_solver = std::make_shared<BiCGStabSolverQDPXX>(*M_fine, p.null_solver_params[0]);
  }


  int num_vecs = fine_level.null_vecs.size();
  if (num_vecs != p.n_vecs[0] ) {
       MasterLog(ERROR, "QDPXX Setup is called without initing vectors, but initial vectors are not initialized");
  }

  IndexArray blocked_lattice_dims;
  IndexArray blocked_lattice_orig;
  CreateBlockList(fine_level.blocklist,
      blocked_lattice_dims,
      blocked_lattice_orig,
      latdims,
      p.block_sizes[0],
      fine_level.info->GetLatticeOrigin());


  // Orthonormalize the vectors -- I heard once that for GS stability is improved
  // if you do it twice.

  orthonormalizeBlockAggregatesQDPXX(fine_level.null_vecs,
                    fine_level.blocklist);

  orthonormalizeBlockAggregatesQDPXX(fine_level.null_vecs,
                    fine_level.blocklist);




  // Create the blocked Clover and Gauge Fields
  // This service needs the blocks, the vectors and is a convenience
    // Function of the M
  coarse_level.info = std::make_shared<const LatticeInfo>(blocked_lattice_orig,
                            blocked_lattice_dims,
                            2, num_vecs, fine_level.info->GetNodeInfo());

  coarse_level.gauge = std::make_shared<CoarseGauge>(*(coarse_level.info));


  M_fine->generateCoarse(fine_level.blocklist, fine_level.null_vecs, *(coarse_level.gauge));

  coarse_level.M = std::make_shared< const CoarseWilsonCloverLinearOperator>(coarse_level.gauge,1);

}

void SetupMGLevels(const SetupParams& p, MultigridLevels& mg_levels,
			std::shared_ptr<const QDPWilsonCloverLinearOperator> M_fine)
{
	 mg_levels.n_levels = p.n_levels;
	if( mg_levels.n_levels < 2 ){
		MasterLog(ERROR, "Number of Multigrid Levels < 2");
	}

	int n_coarse_levels = mg_levels.n_levels-1;
	mg_levels.coarse_levels.resize(n_coarse_levels);

	MasterLog(INFO, "Setup Level 0 and 1");
	SetupQDPXXToCoarse(p,M_fine, mg_levels.fine_level, mg_levels.coarse_levels[0]);

	for(int coarse_level=1; coarse_level < n_coarse_levels; ++coarse_level ) {

		MasterLog(INFO, "Setup Level %d and %d", coarse_level, coarse_level+1);
		SetupCoarseToCoarse(p,mg_levels.coarse_levels[coarse_level-1].M,
				coarse_level, mg_levels.coarse_levels[coarse_level-1],
				mg_levels.coarse_levels[coarse_level]);

	}


}


} // Namespace
