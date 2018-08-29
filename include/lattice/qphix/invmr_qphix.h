/*
 * invmr_qphix.h
 *
 *  Created on: Oct 19, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_QPHIX_INVMR_QPHIX_H_
#define INCLUDE_LATTICE_QPHIX_INVMR_QPHIX_H_

#include "lattice/constants.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"
#include "lattice/qphix/qphix_types.h"
#include "lattice/qphix/qphix_clover_linear_operator.h"
#include "lattice/qphix/qphix_eo_clover_linear_operator.h"
#include "lattice/mr_params.h"
#include <qphix/invmr.h>
#include <memory>

namespace MG {

// Single Precision, for null space solving
template<typename FT>
class MRSolverQPhiXT : public LinearSolver<QPhiXSpinorT<FT>,QPhiXGaugeT<FT>> {
public:

  MRSolverQPhiXT(QPhiXWilsonCloverLinearOperatorT<FT>& M,
                       const MRSolverParams& params) : _params(params),
                           mr_solver( M.getQPhiXOp(),params.MaxIter,params.Omega),
                           solver_wrapper(mr_solver,M.getQPhiXOp()) {}

  MRSolverQPhiXT(QPhiXWilsonCloverEOLinearOperatorT<FT>& M,
                         const MRSolverParams& params) : _params(params),
                             mr_solver( M.getQPhiXOp(),params.MaxIter,params.Omega),
                             solver_wrapper(mr_solver,M.getQPhiXOp())


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

    const LinearSolverParamsBase& _params;
    QPhiXMRSolverT<FT> mr_solver;
    QPhiXUnprecSolverT<FT> solver_wrapper;
 };

  using MRSolverQPhiX = MRSolverQPhiXT<double>;
  using MRSolverQPhiXF = MRSolverQPhiXT<float>;

// Single Precision, for null space solving
template<typename FT>
class MRSmootherQPhiXT : public Smoother<QPhiXSpinorT<FT>,QPhiXGaugeT<FT>> {
public:

  MRSmootherQPhiXT(QPhiXWilsonCloverLinearOperatorT<FT>& M, const MRSolverParams& params) : _params(params),
                           mr_solver( M.getQPhiXOp(), params.MaxIter, params.Omega),
                           solver_wrapper(mr_solver, M.getQPhiXOp()) {}

  MRSmootherQPhiXT(std::shared_ptr<QPhiXWilsonCloverLinearOperatorT<FT>> M, const MRSolverParams& params) : _params(params),
          mr_solver( M->getQPhiXOp(), params.MaxIter, params.Omega),
          solver_wrapper(mr_solver, M->getQPhiXOp()) {}

  MRSmootherQPhiXT(QPhiXWilsonCloverEOLinearOperatorT<FT>& M, const MRSolverParams& params) : _params(params),
                           mr_solver( M.getQPhiXOp(), params.MaxIter, params.Omega),
                           solver_wrapper(mr_solver, M.getQPhiXOp()) {}

  MRSmootherQPhiXT(std::shared_ptr<QPhiXWilsonCloverEOLinearOperatorT<FT>> M, const MRSolverParams& params)  : _params(params),
          mr_solver( M->getQPhiXOp(), params.MaxIter, params.Omega),
          solver_wrapper(mr_solver, M->getQPhiXOp()) {}

  void operator()(QPhiXSpinorT<FT>& out,
                                const QPhiXSpinorT<FT>& in) const
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
        ODD);

  }

 private:

    const LinearSolverParamsBase& _params;

    QPhiXMRSmootherT<FT> mr_solver;
    QPhiXUnprecSolverT<FT> solver_wrapper;


 };

  using MRSmootherQPhiX  = MRSmootherQPhiXT<double>;
  using MRSmootherQPhiXF = MRSmootherQPhiXT<float>;

  // Single Precision, for null space solving
  template<typename FT>
  class MRSmootherQPhiXTEO : public Smoother<QPhiXSpinorT<FT>,QPhiXGaugeT<FT>> {
  public:

    MRSmootherQPhiXTEO(QPhiXWilsonCloverLinearOperatorT<FT>& M, const MRSolverParams& params) : _params(params),
                             mr_smoother( M.getQPhiXOp(), params.MaxIter, params.Omega) {}


    MRSmootherQPhiXTEO(QPhiXWilsonCloverEOLinearOperatorT<FT>& M, const MRSolverParams& params) : _params(params),
                             mr_smoother( M.getQPhiXOp(), params.MaxIter, params.Omega) {}

    MRSmootherQPhiXTEO(const std::shared_ptr<QPhiXWilsonCloverEOLinearOperatorT<FT>> M, const MRSolverParams& params) : _params(params),
                               mr_smoother( M->getQPhiXOp(), params.MaxIter, params.Omega) {}


    void operator()(QPhiXSpinorT<FT>& out,
                                  const QPhiXSpinorT<FT>& in) const
    {
      const int isign= 1;
      int n_iters;
      double rsd_sq_final;
      unsigned long site_flops;
      unsigned long mv_apps;

      (mr_smoother)(out.getCB(ODD).get(),
          in.getCB(ODD).get(),
          _params.RsdTarget,
          n_iters,
          rsd_sq_final,
          site_flops,
          mv_apps,
          isign,
          _params.VerboseP,
          ODD);

    }

   private:

      const LinearSolverParamsBase& _params;

      QPhiXMRSmootherT<FT> mr_smoother;

   };

  using MRSmootherQPhiXEO  = MRSmootherQPhiXTEO<double>;
   using MRSmootherQPhiXEOF = MRSmootherQPhiXTEO<float>;
}  // end namespace MGTEsting





#endif /* INCLUDE_LATTICE_QPHIX_INVMR_QPHIX_H_ */
