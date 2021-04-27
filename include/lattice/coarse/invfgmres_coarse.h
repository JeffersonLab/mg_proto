/*
 * invfgmres_coarse.h
 *
 *  Created on: Oct 17, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_COARSE_INVFGMRES_COARSE_H_
#define INCLUDE_LATTICE_COARSE_INVFGMRES_COARSE_H_

#include "lattice/coarse/coarse_l1_blas.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/invfgmres_generic.h"
#include "lattice/unprec_solver_wrappers.h"
namespace MG {

    using FGMRESSolverCoarse = FGMRESGeneric::FGMRESSolverGeneric<CoarseSpinor>;

    class UnprecFGMRESSolverCoarseWrapper
        : public UnprecLinearSolver<FGMRESGeneric::FGMRESSolverGeneric<CoarseSpinor>> {
    public:
        UnprecFGMRESSolverCoarseWrapper(const EOLinearOperator<Spinor> &M_fine,
                                        const LinearSolverParamsBase &params,
                                        const LinearOperator<Spinor> *prec = nullptr)
            : UnprecLinearSolver<FGMRESGeneric::FGMRESSolverGeneric<Spinor>>(
                  M_fine, params, prec ? prec : (const LinearOperator<Spinor> *)&_op),
              _op(M_fine) {}

    private:
        M_oo_inv<Spinor> _op;
    };
}

#endif /* INCLUDE_LATTICE_COARSE_INVFGMRES_COARSE_H_ */
