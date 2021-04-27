/*! \file
 *  \brief Minimal-Residual (MR) for a generic fermion Linear Operator
 */

#ifndef INCLUDE_LATTICE_FINE_QDPXX_INVMR_QDPXX_H_
#define INCLUDE_LATTICE_FINE_QDPXX_INVMR_QDPXX_H_

#include "lattice/constants.h"
#include "lattice/linear_operator.h"
#include "lattice/mr_params.h"
#include "lattice/solver.h"
#include "qdp.h"

using namespace QDP;

namespace MG {

    class MRSolverQDPXX : public LinearSolverNoPrecon<QDP::LatticeFermion> {
    public:
        /**
         * Constructor
         *
         * \param M: operator
         * \param params: linear solver params
         * \param prec: preconditioner (should be nullptr)
         */

        MRSolverQDPXX(const LinearOperator<Spinor> &M, const LinearSolverParamsBase &params,
                      const LinearOperator<Spinor> *prec = nullptr)
            : LinearSolverNoPrecon<Spinor>(M, params, prec) {}

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
}

#endif /* TEST_QDPXX_INVMR_H_ */
