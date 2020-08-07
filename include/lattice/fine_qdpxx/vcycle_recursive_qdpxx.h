/*
 * vcycle_recursive_qdpxx.h
 *
 *  Created on: Mar 21, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_FINE_QDPXX_VCYCLE_RECURSIVE_QDPXX_H_
#define INCLUDE_LATTICE_FINE_QDPXX_VCYCLE_RECURSIVE_QDPXX_H_

#include "lattice/fine_qdpxx/mg_level_qdpxx.h"
#include "lattice/fine_qdpxx/mg_params_qdpxx.h"
#include "lattice/fine_qdpxx/vcycle_qdpxx_coarse.h"
#include "lattice/solver.h"
#include <memory>
#include <vector>

using namespace QDP;

namespace MG {

    class VCycleRecursiveQDPXX : public LinearSolver<LatticeFermion> {
    public:
        VCycleRecursiveQDPXX(const std::vector<VCycleParams> &vcycle_params,
                             const MultigridLevels &mg_levels);

        std::vector<LinearSolverResults>
        operator()(LatticeFermion &out, const LatticeFermion &in,
                   ResiduumType resid_type = RELATIVE,
                   InitialGuess guees = InitialGuessNotGiven) const;

        const LatticeInfo &GetInfo() const { return *_mg_levels.fine_level.info; }
        const CBSubset &GetSubset() const { return SUBSET_ALL; }

    private:
        const std::vector<VCycleParams> _vcycle_params;
        const MultigridLevels &_mg_levels;

        std::shared_ptr<const LinearSolver<LatticeFermion>> _pre_smoother;
        std::shared_ptr<const LinearSolver<LatticeFermion>> _post_smoother;
        std::shared_ptr<const VCycleQDPCoarse2> _toplevel_vcycle;

        std::vector<std::shared_ptr<const LinearSolver<CoarseSpinor>>> _coarse_presmoother;
        std::vector<std::shared_ptr<const LinearSolver<CoarseSpinor>>> _coarse_postsmoother;
        std::vector<std::shared_ptr<const LinearSolver<CoarseSpinor>>> _coarse_vcycle;
        std::vector<std::shared_ptr<const LinearSolver<CoarseSpinor>>> _bottom_solver;
    };

} // namespace MG
#endif /* INCLUDE_LATTICE_FINE_QDPXX_VCYCLE_RECURSIVE_QDPXX_H_ */
