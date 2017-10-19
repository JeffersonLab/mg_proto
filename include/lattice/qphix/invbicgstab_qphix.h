/*
 * invbicgstab_qphix.h
 *
 *  Created on: Oct 18, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_QPHIX_INVBICGSTAB_QPHIX_H_
#define INCLUDE_LATTICE_QPHIX_INVBICGSTAB_QPHIX_H_


#include "lattice/constants.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"
#include "lattice/qphix/qphix_types.h"
#include "lattice/qphix/qphix_clover_linear_operator.h"
#include "qphix/invbicgstab.h"
#include <memory>

namespace MG {

// Single Precision, for null space solving
template<typename FT>
class BiCGStabSolverQPhiXT : public LinearSolver<QPhiXSpinorT<FT>,QPhiXGaugeT<FT>> {
public:

  BiCGStabSolverQPhiXT(QPhiXWilsonCloverLinearOperatorT<FT>& M,
                       const LinearSolverParamsBase& params) : _M(M),_params(params),
                           bicg_solver( M.getQPhiXOp(),params.MaxIter),
                           solver_wrapper(bicg_solver,M.getQPhiXOp())


  {}

  LinearSolverResults operator()(QPhiXSpinorT<FT>& out,
                                const QPhiXSpinorT<FT>& in,
                                ResiduumType resid_type = RELATIVE ) const
  {
    const int isign= 1;
    int n_iters;
    double rsd_sq_final;
    unsigned long site_flops;
    unsigned long mv_apps;

    (solver_wrapper)(&(out.get()),
        &(in.get()),
        _params.RsdTarget,
        n_iters,
        rsd_sq_final,
        site_flops,
        mv_apps,
        isign,
        _params.VerboseP,
        ODD,
        resid_type == MG::RELATIVE ? QPhiX::RELATIVE : QPhiX::ABSOLUTE);

    LinearSolverResults ret_val;
    ret_val.n_count = n_iters;
    ret_val.resid = sqrt(rsd_sq_final);
    ret_val.resid_type = resid_type;
    return ret_val;

  }

 private:
    QPhiXWilsonCloverLinearOperatorT<FT>& _M;
    const LinearSolverParamsBase& _params;
    QPhiXBiCGStabT<FT> bicg_solver;
    QPhiXUnprecSolverT<FT> solver_wrapper;


 };

  using BiCGStabSolverQPhiX = BiCGStabSolverQPhiXT<double>;
  using BiCGStabSolverQPhiXF = BiCGStabSolverQPhiXT<float>;


}  // end namespace MGTEsting


#endif /* INCLUDE_LATTICE_QPHIX_INVBICGSTAB_QPHIX_H_ */
