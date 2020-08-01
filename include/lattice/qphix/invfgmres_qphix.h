/*
 * invfgmres_qphix.h
 *
 *  Created on: Oct 17, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_QPHIX_INVFGMRES_QPHIX_H_
#define INCLUDE_LATTICE_QPHIX_INVFGMRES_QPHIX_H_

#include "lattice/qphix/qphix_types.h"
#include "lattice/qphix/qphix_blas_wrappers.h"
#include "lattice/invfgmres_generic.h"
#include "lattice/unprec_solver_wrappers.h"
#include "lattice/qphix/qphix_clover_linear_operator.h"
#include "lattice/qphix/qphix_eo_clover_linear_operator.h"
#include <memory>

namespace MG {

  using FGMRESSolverQPhiX = FGMRESGeneric::FGMRESSolverGeneric<QPhiXSpinor,QPhiXGauge>;
  using FGMRESSolverQPhiXF = FGMRESGeneric::FGMRESSolverGeneric<QPhiXSpinorF,QPhiXGaugeF>;

  template<typename FT>
  class FGMRESSmootherQPhiXT : public FGMRESGeneric::FGMRESSolverGeneric<QPhiXSpinorT<FT>,QPhiXGaugeT<FT>>,
                               public Smoother<QPhiXSpinorT<FT>,QPhiXGaugeT<FT>>
  {
    static LinearSolverParamsBase setDefaults(LinearSolverParamsBase params) {
      if (params.NKrylov == 0) {
        params.NKrylov = params.MaxIter;
      }
      return params;
    }

    public:
    FGMRESSmootherQPhiXT(const std::shared_ptr<const QPhiXWilsonCloverEOLinearOperatorT<FT>>& M_fine, const LinearSolverParamsBase& params) :
      FGMRESGeneric::FGMRESSolverGeneric<QPhiXSpinorT<FT>,QPhiXGaugeT<FT>>(M_fine, setDefaults(params), nullptr, "S") {}

    void operator()(QPhiXSpinorT<FT>& out, const QPhiXSpinorT<FT>& in) const override {
      FGMRESGeneric::FGMRESSolverGeneric<QPhiXSpinorT<FT>,QPhiXGaugeT<FT>>::operator()(out, in);
    }
  };

  using FGMRESSmootherQPhiXF = FGMRESSmootherQPhiXT<float>;

  using UnprecFGMRESSolverQPhiXWrapper =  UnprecLinearSolverWrapper<QPhiXSpinor,QPhiXGauge,FGMRESGeneric::FGMRESSolverGeneric<QPhiXSpinor,QPhiXGauge>>;
  using UnprecFGMRESSolverQPhiXFWrapper =  UnprecLinearSolverWrapper<QPhiXSpinorF,QPhiXGaugeF,FGMRESGeneric::FGMRESSolverGeneric<QPhiXSpinor,QPhiXGauge>>;

  // Null space solvers
  template<typename LinOp> class NullSolverFGMRES;

  template<typename FT> class NullSolverFGMRES<QPhiXWilsonCloverLinearOperatorT<FT>> : public FGMRESGeneric::FGMRESSolverGeneric<QPhiXSpinorT<FT>,QPhiXGaugeT<FT>> {
    public:
    NullSolverFGMRES<QPhiXWilsonCloverLinearOperatorT<FT>>(const std::shared_ptr<const QPhiXWilsonCloverLinearOperatorT<FT>>& M_fine, const LinearSolverParamsBase& params) :
      FGMRESGeneric::FGMRESSolverGeneric<QPhiXSpinorT<FT>,QPhiXGaugeT<FT>>(M_fine, params) {}
  };

  template<typename FT> class NullSolverFGMRES<QPhiXWilsonCloverEOLinearOperatorT<FT>> : public UnprecLinearSolverWrapper<QPhiXSpinorT<FT>,QPhiXGaugeT<FT>,FGMRESGeneric::FGMRESSolverGeneric<QPhiXSpinorT<FT>,QPhiXGaugeT<FT>>> {
    public:
    NullSolverFGMRES<QPhiXWilsonCloverEOLinearOperatorT<FT>>(const std::shared_ptr<const QPhiXWilsonCloverEOLinearOperatorT<FT>>& M_fine, const LinearSolverParamsBase& params) :
      UnprecLinearSolverWrapper<QPhiXSpinorT<FT>,QPhiXGaugeT<FT>,FGMRESGeneric::FGMRESSolverGeneric<QPhiXSpinorT<FT>,QPhiXGaugeT<FT>>>(M_fine, params) {}
  };
}



#endif /* INCLUDE_LATTICE_QPHIX_INVFGMRES_QPHIX_H_ */
