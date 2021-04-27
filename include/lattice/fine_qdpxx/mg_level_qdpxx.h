/*
 * mg_level_qdpxx.h
 *
 *  Created on: Mar 15, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_FINE_QDPXX_MG_LEVEL_QDPXX_H_
#define INCLUDE_LATTICE_FINE_QDPXX_MG_LEVEL_QDPXX_H_

#include "lattice/coarse/block.h"
#include "lattice/fine_qdpxx/mg_params_qdpxx.h"
#include "lattice/fine_qdpxx/wilson_clover_linear_operator.h"
#include "lattice/linear_operator.h"
#include "lattice/mg_level_coarse.h"
#include "lattice/solver.h"
#include "utils/print_utils.h"
#include <memory>

using std::make_shared;
using std::shared_ptr;
using std::vector;

namespace MG {

    struct MGLevelQDPXX {
        std::shared_ptr<const LatticeInfo> info;
        QDP::multi1d<QDP::LatticeFermion> null_vecs;
        std::shared_ptr<const LinearSolver<QDP::LatticeFermion>>
            null_solver; // Solver for NULL on this level
        std::shared_ptr<const QDPWilsonCloverLinearOperator> M;
        std::vector<Block> blocklist;

        ~MGLevelQDPXX() {}
    };

    struct MultigridLevels {
        int n_levels;
        MGLevelQDPXX fine_level;
        std::vector<MGLevelCoarse> coarse_levels;
    };

    void SetupQDPXXToCoarseGenerateVecs(const SetupParams &p,
                                        std::shared_ptr<const QDPWilsonCloverLinearOperator> M_fine,
                                        MGLevelQDPXX &fine_level, MGLevelCoarse &coarse_level);

    void SetupQDPXXToCoarseVecsIn(const SetupParams &p,
                                  std::shared_ptr<const QDPWilsonCloverLinearOperator> M_fine,
                                  MGLevelQDPXX &fine_level, MGLevelCoarse &coarse_level);

    void SetupQDPXXToCoarse(const SetupParams &p,
                            std::shared_ptr<const QDPWilsonCloverLinearOperator> M_fine,
                            MGLevelQDPXX &fine_level, MGLevelCoarse &coarse_level);

    void SetupMGLevels(const SetupParams &p, MultigridLevels &mg_levels,
                       std::shared_ptr<const QDPWilsonCloverLinearOperator> M_fine);
}

#endif /* INCLUDE_LATTICE_FINE_QDPXX_MG_LEVEL_QDPXX_H_ */
