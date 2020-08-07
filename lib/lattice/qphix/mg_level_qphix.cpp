/*
 * mg_level_qphix.cpp
 *
 *  Created on: Oct 20, 2017
 *      Author: bjoo
 */

#include <lattice/qphix/mg_level_qphix.h>
#include <lattice/qphix/qphix_aggregate.h>
#include <lattice/qphix/qphix_blas_wrappers.h>
#include <lattice/qphix/qphix_qdp_utils.h>

namespace MG {

    void SetupQPhiXMGLevels(const SetupParams &p, QPhiXMultigridLevels &mg_levels,
                            const std::shared_ptr<const QPhiXWilsonCloverLinearOperatorF> &M_fine) {
        SetupQPhiXMGLevelsT<>(p, mg_levels, M_fine);
    }

    void
    SetupQPhiXMGLevels(const SetupParams &p, QPhiXMultigridLevelsEO &mg_levels,
                       const std::shared_ptr<const QPhiXWilsonCloverEOLinearOperatorF> &M_fine) {
        SetupQPhiXMGLevelsT<>(p, mg_levels, M_fine);
    }
}
