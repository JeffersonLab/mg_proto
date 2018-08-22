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
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/block.h"
#include "lattice/solver.h"
#include "lattice/fine_qdpxx/mg_params_qdpxx.h"
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

	using MGLevelCoarse = MGLevelCoarseT<BiCGStabSolverCoarse , CoarseWilsonCloverLinearOperator>;
	using MGLevelCoarseEO = MGLevelCoarseT< UnprecBiCGStabSolverCoarseWrapper , CoarseEOWilsonCloverLinearOperator>;

  void SetupCoarseToCoarse(const SetupParams& p, std::shared_ptr< const CoarseWilsonCloverLinearOperator > M_fine, int fine_level_id,
              MGLevelCoarse& fine_level, MGLevelCoarse& coarse_level);
  void SetupCoarseToCoarse(const SetupParams& p, std::shared_ptr< const CoarseEOWilsonCloverLinearOperator > M_fine, int fine_level_id,
                MGLevelCoarseEO& fine_level, MGLevelCoarseEO& coarse_level);

}


#endif /* INCLUDE_LATTICE_MG_LEVEL_COARSE_H_ */
