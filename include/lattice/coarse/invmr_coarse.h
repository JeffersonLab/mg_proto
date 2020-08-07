/*! \file
 *  \brief Minimal-Residual (MR) for a generic fermion Linear Operator
 */
#ifndef INCLUDE_LATTICE_COARSE_INVMR_COARSE_H_
#define INCLUDE_LATTICE_COARSE_INVMR_COARSE_H_

#include "lattice/coarse/coarse_types.h"
#include "lattice/constants.h"
#include "lattice/linear_operator.h"
#include "lattice/mr_params.h"
#include "lattice/solver.h"

#include "lattice/unprec_solver_wrappers.h"

namespace MG {

    class MRSolverCoarse : public LinearSolverNoPrecon<CoarseSpinor> {
    public:
        /**
         * Constructor
         *
         * \param M: operator
         * \param params: linear solver params
         * \param prec: preconditioner (should be nullptr)
         */

        MRSolverCoarse(const LinearOperator<CoarseSpinor> &M, const LinearSolverParamsBase &params,
                       const LinearOperator<CoarseSpinor> *prec = nullptr)
            : LinearSolverNoPrecon<CoarseSpinor>(M, params, prec) {}

        /**
         * Compute the solution.
         *
         * \param[out] out: solution vector
         * \param in: input vector
         * \param resid_type: stopping criterion (RELATIVE or ABSOLUTE)
         * \param guess: Whether the initial is provided
         */

        std::vector<LinearSolverResults>
        operator()(Spinor &out, const Spinor &in, ResiduumType resid_type = RELATIVE,
                   InitialGuess guess = InitialGuessNotGiven) const override;
    };

    using UnprecMRSolverCoarseWrapper = UnprecLinearSolver<MRSolverCoarse>;
}

#endif /* TEST_QDPXX_INVMR_COARSE_H_ */
