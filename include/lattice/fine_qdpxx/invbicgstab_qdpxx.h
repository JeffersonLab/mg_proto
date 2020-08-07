/*! \file
 *  \brief Conjugate-Gradient algorithm for a generic Linear Operator
 */

#ifndef INCLUDE_LATTICE_FINE_QDPXX_INVBICGSTAB_QDPXX_H_
#define INCLUDE_LATTICE_FINE_QDPXX_INVBICGSTAB_QDPXX_H_

#include "lattice/constants.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"
#include "qdp.h"

using namespace QDP;

namespace MG {

    class BiCGStabSolverQDPXX
        : public LinearSolver<QDP::LatticeFermion, QDP::multi1d<QDP::LatticeColorMatrix>> {
    public:
        using Spinor = QDP::LatticeFermion;
        using Gauge = QDP::multi1d<QDP::LatticeColorMatrix>;
        BiCGStabSolverQDPXX(const LinearOperator<Spinor, Gauge> &M,
                            const LinearSolverParamsBase &params)
            : _M(M), _params(params) {}

        std::vector<LinearSolverResults> operator()(Spinor &out, const Spinor &in,
                                                    ResiduumType resid_type = RELATIVE) const;

        const LatticeInfo &GetInfo() const { return _M.GetInfo(); }
        const CBSubset &GetSubset() const { return _M.GetSubset(); }

    private:
        const LinearOperator<Spinor, Gauge> &_M;
        const LinearSolverParamsBase &_params;
    };

} // end namespace MGTEsting

#endif /* TEST_QDPXX_INVBICGSTAB_H_ */
