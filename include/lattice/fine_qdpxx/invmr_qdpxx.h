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

    class MRSolverQDPXX
        : public LinearSolver<QDP::LatticeFermion, QDP::multi1d<QDP::LatticeColorMatrix>> {
    public:
        MRSolverQDPXX(
            const LinearOperator<QDP::LatticeFermion, QDP::multi1d<QDP::LatticeColorMatrix>> &M,
            const MG::LinearSolverParamsBase &params);

        std::vector<LinearSolverResults> operator()(QDP::LatticeFermion &out,
                                                    const QDP::LatticeFermion &in,
                                                    ResiduumType resid_type = RELATIVE) const;

    private:
        const LinearOperator<QDP::LatticeFermion, QDP::multi1d<QDP::LatticeColorMatrix>> &_M;
        const LinearSolverParamsBase &_params;
    };

    class MRSmootherQDPXX
        : public Smoother<QDP::LatticeFermion, QDP::multi1d<QDP::LatticeColorMatrix>> {
    public:
        MRSmootherQDPXX(
            const LinearOperator<QDP::LatticeFermion, QDP::multi1d<QDP::LatticeColorMatrix>> &M,
            const MG::LinearSolverParamsBase &params);
        void operator()(QDP::LatticeFermion &out, const QDP::LatticeFermion &in) const;

    private:
        const LinearOperator<QDP::LatticeFermion, QDP::multi1d<QDP::LatticeColorMatrix>> &_M;
        const LinearSolverParamsBase &_params;
    };
}

#endif /* TEST_QDPXX_INVMR_H_ */
