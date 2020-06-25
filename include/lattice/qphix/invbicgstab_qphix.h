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
#include "lattice/qphix/qphix_eo_clover_linear_operator.h"
#include "qphix/invbicgstab.h"
#include <memory>
#include <stdexcept>

namespace MG {

// Single Precision, for null space solving
template<typename FT>
class BiCGStabSolverQPhiXT : public LinearSolver<QPhiXSpinorT<FT>,QPhiXGaugeT<FT>> {
public:

  BiCGStabSolverQPhiXT(const QPhiXWilsonCloverLinearOperatorT<FT>& M,
                       const LinearSolverParamsBase& params) :
                           _info(M.GetInfo()), _params(params),
                           bicg_solver( M.getQPhiXOp(),params.MaxIter),
                           solver_wrapper(bicg_solver,M.getQPhiXOp())


  {}

  BiCGStabSolverQPhiXT(const QPhiXWilsonCloverEOLinearOperatorT<FT>& M,
                        const LinearSolverParamsBase& params) :
                            _info(M.GetInfo()), _params(params),
                            bicg_solver( M.getQPhiXOp(),params.MaxIter),
                            solver_wrapper(bicg_solver,M.getQPhiXOp())


   {}
  std::vector<LinearSolverResults> operator()(QPhiXSpinorT<FT>& out,
                                const QPhiXSpinorT<FT>& in,
                                ResiduumType resid_type = RELATIVE ) const
  {
    const int isign= 1;
    int n_iters;
    unsigned long site_flops;
    unsigned long mv_apps;
    assert(in.GetNCol() == out.GetNCol());
    IndexType ncol = in.GetNCol();
    std::vector<double> rsd_sq_final(ncol);

    for (int col=0; col < ncol; ++col) {
      (solver_wrapper)(&(out.get(col)),
          &(in.get(col)),
          _params.RsdTarget,
          n_iters,
          rsd_sq_final[col],
          site_flops,
          mv_apps,
          isign,
          _params.VerboseP,
          ODD,
          resid_type == MG::RELATIVE ? QPhiX::RELATIVE : QPhiX::ABSOLUTE);
    }

    std::vector<LinearSolverResults> ret_val(ncol);
    for (int col=0; col < ncol; ++col) {
       ret_val[col].n_count = n_iters;
       ret_val[col].resid = sqrt(rsd_sq_final[col]);
       ret_val[col].resid_type = resid_type;
    }
    return ret_val;

  }

  const LatticeInfo& GetInfo() const { return _info; }
  const CBSubset& GetSubset() const { return SUBSET_ALL; }

 private:

    const LatticeInfo _info;
    const LinearSolverParamsBase& _params;
    QPhiXBiCGStabT<FT> bicg_solver;
    QPhiXUnprecSolverT<FT> solver_wrapper;


 };

  using BiCGStabSolverQPhiX = BiCGStabSolverQPhiXT<double>;
  using BiCGStabSolverQPhiXF = BiCGStabSolverQPhiXT<float>;

  template<typename FT>
  class BiCGStabSolverQPhiXTEO : public LinearSolver<QPhiXSpinorT<FT>,QPhiXGaugeT<FT>> {
  public:

    BiCGStabSolverQPhiXTEO(const QPhiXWilsonCloverLinearOperatorT<FT>& M,
                         const LinearSolverParamsBase& params) : _params(params),
                             bicg_solver( M.getQPhiXOp(),params.MaxIter)    {}

    BiCGStabSolverQPhiXTEO(const QPhiXWilsonCloverEOLinearOperatorT<FT>& M,
                          const LinearSolverParamsBase& params) : _params(params),
                              bicg_solver( M.getQPhiXOp(),params.MaxIter)     {}

    std::vector<LinearSolverResults> operator()(QPhiXSpinorT<FT>& out,
                                  const QPhiXSpinorT<FT>& in,
                                  ResiduumType resid_type = RELATIVE ) const
    {
      const int isign= 1;
      int n_iters;
      unsigned long site_flops;
      unsigned long mv_apps;
      assert(in.GetNCol() == out.GetNCol());
      IndexType ncol = in.GetNCol();
      std::vector<double> rsd_sq_final(ncol);

      for (int col=0; col < ncol; ++col) {
        (bicg_solver)(out.getCB(col,ODD).get(),
            in.getCB(col,ODD).get(),
            _params.RsdTarget,
            n_iters,
            rsd_sq_final[col],
            site_flops,
            mv_apps,
            isign,
            _params.VerboseP,
            ODD,
            resid_type == MG::RELATIVE ? QPhiX::RELATIVE : QPhiX::ABSOLUTE);
      }

      std::vector<LinearSolverResults> ret_val(ncol);
      for (int col=0; col < ncol; ++col) {
        ret_val[col].n_count = n_iters;
        ret_val[col].resid = sqrt(rsd_sq_final[col]);
        ret_val[col].resid_type = resid_type;
      }
      return ret_val;

    }

   private:

      const LinearSolverParamsBase& _params;
      QPhiXBiCGStabT<FT> bicg_solver;

   };

    using BiCGStabSolverQPhiXEO = BiCGStabSolverQPhiXTEO<double>;
    using BiCGStabSolverQPhiXFEO = BiCGStabSolverQPhiXTEO<float>;

}  // end namespace MGTEsting


#endif /* INCLUDE_LATTICE_QPHIX_INVBICGSTAB_QPHIX_H_ */
