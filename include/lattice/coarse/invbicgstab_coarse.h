/*! \file
 *  \brief Conjugate-Gradient algorithm for a generic Linear Operator
 */

#ifndef INCLUDE_LATTICE_COARSE_INVBICGSTAB_COARSE_H_
#define INCLUDE_LATTICE_COARSE_INVBICGSTAB_COARSE_H_

#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/subset.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"
#include <lattice/unprec_solver_wrappers.h>
#include <memory>

namespace MG {

    class BiCGStabSolverCoarse : public LinearSolverNoPrecon<CoarseSpinor> {
    public:
        /**
         * Constructor
         *
         * \param M: operator
         * \param params: linear solver params
         * \param prec: preconditioner (should be nullptr)
         */

        BiCGStabSolverCoarse(const LinearOperator<CoarseSpinor> &M,
                             const LinearSolverParamsBase &params,
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

    using UnprecBiCGStabSolverCoarseWrapper = UnprecLinearSolver<BiCGStabSolverCoarse>;

} // namespace MG

#endif /* TEST_QDPXX_INVBICGSTAB_H_ */
