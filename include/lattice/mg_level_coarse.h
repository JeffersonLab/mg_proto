/*
 * mg_level_coarse.h
 *
 *  Created on: Mar 15, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_MG_LEVEL_COARSE_H_
#define INCLUDE_LATTICE_MG_LEVEL_COARSE_H_

#include <lattice/coarse/invbicgstab_coarse.h>
#include <memory>
#include <string>
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/block.h"
#include "lattice/coarse/coarse_op.h"
#include "lattice/solver.h"
#include "lattice/fine_qdpxx/mg_params_qdpxx.h"
#include "utils/timer.h"
#include "lattice/coarse/coarse_wilson_clover_linear_operator.h"
#include "lattice/coarse/coarse_eo_wilson_clover_linear_operator.h"

namespace MG {
  template<typename SolverT, typename LinOpT>
  struct MGLevelCoarseT {
  using Solver = SolverT;
  using LinOp = LinOpT;
  std::shared_ptr<const LatticeInfo> info;
  std::shared_ptr<CoarseGauge> gauge;
  std::vector<std::shared_ptr<CoarseSpinor> > null_vecs; // NULL Vectors
  std::vector<Block> blocklist;
  std::shared_ptr< const SolverT > null_solver;           // Solver for NULL on this level;
  std::shared_ptr< const LinOpT > M;

  ~MGLevelCoarseT() {}
  };

  // Unpreconditioned levels
  using MGLevelCoarse = MGLevelCoarseT<BiCGStabSolverCoarse , CoarseWilsonCloverLinearOperator>;

  // Preconditioned levels
  using MGLevelCoarseEO = MGLevelCoarseT< UnprecBiCGStabSolverCoarseWrapper , CoarseEOWilsonCloverLinearOperator>;

  template<typename CoarseLevelT>
  void SetupCoarseToCoarseT(const SetupParams& p,
              std::shared_ptr<const typename CoarseLevelT::LinOp > M_fine,
              int fine_level_id,
              CoarseLevelT& fine_level,
              CoarseLevelT& coarse_level)
  {
    // Info should already be created

    // Null solver is BiCGStab. Let us make a parameter struct for it.
    LinearSolverParamsBase params;
    params.MaxIter = p.null_solver_max_iter[fine_level_id];

    params.RsdTarget = p.null_solver_rsd_target[fine_level_id];
    params.VerboseP = p.null_solver_verboseP[fine_level_id];


    fine_level.null_solver = std::make_shared<typename CoarseLevelT::Solver>(M_fine, params);

    // Zero RHS
    const LatticeInfo& fine_info = *(fine_level.info);

    CoarseSpinor b(fine_info);
    ZeroVec(b);

    // Generate the vectors
    int num_vecs = p.n_vecs[fine_level_id];
    fine_level.null_vecs.resize(num_vecs);
    for(int k=0; k < num_vecs; ++k) {
      fine_level.null_vecs[k] = std::make_shared<CoarseSpinor>(fine_info);

        Gaussian(*(fine_level.null_vecs[k]));
      std::vector<LinearSolverResults> res = (*(fine_level.null_solver))((*fine_level.null_vecs[k]),b, ABSOLUTE);
      assert(res.size() == 1);
      MasterLog(INFO, "Level %d: Solver Took: %d iterations",fine_level_id, res[0].n_count);

    }

    IndexArray blocked_lattice_dims;
    IndexArray blocked_lattice_orig;
    CreateBlockList(fine_level.blocklist,
        blocked_lattice_dims,
        blocked_lattice_orig,
        fine_level.info->GetLatticeDimensions(),
        p.block_sizes[fine_level_id],
        fine_level.info->GetLatticeOrigin());

    // Orthonormalize the vectors -- I heard once that for GS stability is improved
    // if you do it twice.
    orthonormalizeBlockAggregates(fine_level.null_vecs,
                      fine_level.blocklist);

    orthonormalizeBlockAggregates(fine_level.null_vecs,
                      fine_level.blocklist);


    // Create the blocked Clover and Gauge Fields
    // This service needs the blocks, the vectors and is a convenience
      // Function of the M
    coarse_level.info = std::make_shared<LatticeInfo>(blocked_lattice_orig,
                              blocked_lattice_dims,
                              2, num_vecs, NodeInfo());

    coarse_level.gauge = std::make_shared<CoarseGauge>(*(coarse_level.info));

    M_fine->generateCoarse(fine_level.blocklist, fine_level.null_vecs, *(coarse_level.gauge));

    //FIXME: Insert inversion of coarse level gauge links... here?

    coarse_level.M = std::make_shared<const typename CoarseLevelT::LinOp>(coarse_level.gauge,fine_level_id+1);

    const char *coarse_prefix_name = std::getenv("MG_COARSE_FILENAME");
    if (coarse_prefix_name != nullptr && std::strlen(coarse_prefix_name) > 0) {
      std::string filename = std::string(coarse_prefix_name) + "_level" + std::to_string(fine_level_id+1) + ".bin";
      MasterLog(INFO, "CoarseEOCloverLinearOperator: Writing coarse operator in %s", filename.c_str());
      CoarseDiracOp::write(*(coarse_level.gauge), filename); 
    }


  }
  // These need to be moved into a .cc file. Right now they are with QDPXX (shriek!!!)
  void SetupCoarseToCoarse(const SetupParams& p, std::shared_ptr< const CoarseWilsonCloverLinearOperator > M_fine, int fine_level_id,
              MGLevelCoarse& fine_level, MGLevelCoarse& coarse_level);

  void SetupCoarseToCoarse(const SetupParams& p, std::shared_ptr< const CoarseEOWilsonCloverLinearOperator > M_fine, int fine_level_id,
                MGLevelCoarseEO& fine_level, MGLevelCoarseEO& coarse_level);


}


#endif /* INCLUDE_LATTICE_MG_LEVEL_COARSE_H_ */
