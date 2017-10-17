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

#include "lattice/coarse/coarse_wilson_clover_linear_operator.h"

namespace MG {
	struct MGLevelCoarse {
	std::shared_ptr<const LatticeInfo> info;
	std::shared_ptr<CoarseGauge> gauge;
	std::vector<std::shared_ptr<CoarseSpinor> > null_vecs; // NULL Vectors
	std::vector<Block> blocklist;
	std::shared_ptr< const BiCGStabSolverCoarse > null_solver;           // Solver for NULL on this level;
	std::shared_ptr< const CoarseWilsonCloverLinearOperator > M;

	~MGLevelCoarse() {}
};

}


#endif /* INCLUDE_LATTICE_MG_LEVEL_COARSE_H_ */
