/*
 * invfgmres_qphix.h
 *
 *  Created on: Oct 17, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_QPHIX_INVFGMRES_QPHIX_H_
#define INCLUDE_LATTICE_QPHIX_INVFGMRES_QPHIX_H_

#include "lattice/invfgmres_generic.h"
#include "lattice/qphix/qphix_blas_wrappers.h"
#include "lattice/qphix/qphix_clover_linear_operator.h"
#include "lattice/qphix/qphix_eo_clover_linear_operator.h"
#include "lattice/qphix/qphix_types.h"
#include "lattice/unprec_solver_wrappers.h"
#include <memory>

namespace MG {

    using FGMRESSolverQPhiX = FGMRESGeneric::FGMRESSolverGeneric<QPhiXSpinor>;
    using FGMRESSolverQPhiXF = FGMRESGeneric::FGMRESSolverGeneric<QPhiXSpinorF>;

    template <typename Spinor>
    class FGMRESSmoother : public FGMRESGeneric::FGMRESSolverGeneric<Spinor> {
        static LinearSolverParamsBase setDefaults(LinearSolverParamsBase params) {
            if (params.NKrylov == 0) { params.NKrylov = params.MaxIter; }
            return params;
        }

    public:
        FGMRESSmoother(const LinearOperator<Spinor> &M_fine, const LinearSolverParamsBase &params,
                       const LinearOperator<Spinor> *prec = nullptr)
            : FGMRESGeneric::FGMRESSolverGeneric<Spinor>(M_fine, setDefaults(params), prec, "S") {}
    };

    using FGMRESSmootherQPhiXF = FGMRESSmoother<QPhiXSpinorF>;

    using UnprecFGMRESSolverQPhiXWrapper =
        UnprecLinearSolver<FGMRESGeneric::FGMRESSolverGeneric<QPhiXSpinor>>;
    using UnprecFGMRESSolverQPhiXFWrapper =
        UnprecLinearSolver<FGMRESGeneric::FGMRESSolverGeneric<QPhiXSpinorF>>;

    // Null space solvers
    // NOTE: Solve the original operator

    template <typename LinOp> class NullSolverFGMRES;

    template <typename FT>
    class NullSolverFGMRES<QPhiXWilsonCloverLinearOperatorT<FT>>
        : public FGMRESGeneric::FGMRESSolverGeneric<QPhiXSpinorT<FT>> {
    public:
        NullSolverFGMRES<QPhiXWilsonCloverLinearOperatorT<FT>>(
            const QPhiXWilsonCloverLinearOperatorT<FT> &M_fine,
            const LinearSolverParamsBase &params,
            const LinearOperator<QPhiXSpinorT<FT>> *prec = nullptr)
            : FGMRESGeneric::FGMRESSolverGeneric<QPhiXSpinorT<FT>>(M_fine, params, prec) {}
    };

    template <typename FT>
    class NullSolverFGMRES<QPhiXWilsonCloverEOLinearOperatorT<FT>>
        : public UnprecLinearSolver<FGMRESGeneric::FGMRESSolverGeneric<QPhiXSpinorT<FT>>> {
    public:
        NullSolverFGMRES<QPhiXWilsonCloverEOLinearOperatorT<FT>>(
            const QPhiXWilsonCloverEOLinearOperatorT<FT> &M_fine,
            const LinearSolverParamsBase &params,
            const LinearOperator<QPhiXSpinorT<FT>> *prec = nullptr)
            : UnprecLinearSolver<FGMRESGeneric::FGMRESSolverGeneric<QPhiXSpinorT<FT>>>(
                  M_fine, params, prec) {}
    };
} // namespace MG

#endif /* INCLUDE_LATTICE_QPHIX_INVFGMRES_QPHIX_H_ */
