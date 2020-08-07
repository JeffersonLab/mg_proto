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
#include "lattice/mr_params.h"
#include "lattice/qphix/qphix_clover_linear_operator.h"
#include "lattice/qphix/qphix_eo_clover_linear_operator.h"
#include "lattice/qphix/qphix_types.h"
#include "lattice/solver.h"
#include <memory>
#include <qphix/invmr.h>

namespace MG {

    // Single Precision, for null space solving
    template <typename FT> class MRSolverQPhiXT : public LinearSolverNoPrecon<QPhiXSpinorT<FT>> {
    public:
        MRSolverQPhiXT(const QPhiXWilsonCloverLinearOperatorT<FT> &M,
                       const LinearSolverParamsBase &params)
            : LinearSolverNoPrecon<QPhiXSpinorT<FT>>(M, params),
              _params(params),
              mr_solver(M.getQPhiXOp(), params.MaxIter, params.Omega),
              solver_wrapper(mr_solver, M.getQPhiXOp()) {}

        MRSolverQPhiXT(const QPhiXWilsonCloverEOLinearOperatorT<FT> &M,
                       const LinearSolverParamsBase &params)
            : LinearSolverNoPrecon<QPhiXSpinorT<FT>>(M, params),
              _params(params),
              mr_solver(M.getQPhiXOp(), params.MaxIter, params.Omega),
              solver_wrapper(mr_solver, M.getQPhiXOp())

        {}
        std::vector<LinearSolverResults>
        operator()(QPhiXSpinorT<FT> &out, const QPhiXSpinorT<FT> &in,
                   ResiduumType resid_type = RELATIVE,
                   InitialGuess guess = InitialGuessNotGiven) const {
            (void)guess;

            const int isign = 1;
            int n_iters = 0;
            unsigned long site_flops = 0;
            unsigned long mv_apps = 0;
            assert(in.GetNCol() == out.GetNCol());
            IndexType ncol = in.GetNCol();
            std::vector<double> rsd_sq_final(ncol);

            if (_params.MaxIter <= 0) {
                CopyVec(out, in, SUBSET_ODD);
            } else {
                for (int col = 0; col < ncol; ++col) {
                    (solver_wrapper)(
                        &(out.get(col)), &(in.get(col)), _params.RsdTarget, n_iters,
                        rsd_sq_final[col], site_flops, mv_apps, isign, _params.VerboseP, ODD,
                        resid_type == MG::RELATIVE ? QPhiX::RELATIVE : QPhiX::ABSOLUTE);
                }
            }

            std::vector<LinearSolverResults> ret_val(ncol);
            for (int col = 0; col < ncol; ++col) {
                ret_val[col].n_count = n_iters;
                ret_val[col].resid = std::sqrt(rsd_sq_final[col]);
                ret_val[col].resid_type = resid_type;
            }
            return ret_val;
        }

    private:
        const LinearSolverParamsBase &_params;
        QPhiXMRSolverT<FT> mr_solver;
        QPhiXUnprecSolverT<FT> solver_wrapper;
    };

    using MRSolverQPhiX = MRSolverQPhiXT<double>;
    using MRSolverQPhiXF = MRSolverQPhiXT<float>;

    // Single Precision, for null space solving
    template <typename FT> class MRSmootherQPhiXT : public LinearSolverNoPrecon<QPhiXSpinorT<FT>> {
    public:
        MRSmootherQPhiXT(const QPhiXWilsonCloverLinearOperatorT<FT> &M,
                         const LinearSolverParamsBase &params)
            : LinearSolverNoPrecon<QPhiXSpinorT<FT>>(M, params),
              _params(params),
              mr_solver(M.getQPhiXOp(), params.MaxIter, params.Omega),
              solver_wrapper(mr_solver, M.getQPhiXOp()) {}

        MRSmootherQPhiXT(const QPhiXWilsonCloverEOLinearOperatorT<FT> &M,
                         const LinearSolverParamsBase &params)
            : LinearSolverNoPrecon<QPhiXSpinorT<FT>>(M, params),
              _params(params),
              mr_solver(M.getQPhiXOp(), params.MaxIter, params.Omega),
              solver_wrapper(mr_solver, M.getQPhiXOp()) {}

        std::vector<LinearSolverResults>
        operator()(QPhiXSpinorT<FT> &out, const QPhiXSpinorT<FT> &in,
                   ResiduumType resid_type = RELATIVE,
                   InitialGuess guess = InitialGuessNotGiven) const override {
            (void)guess;
            (void)resid_type;

            const int isign = 1;
            int n_iters = 0;
            double rsd_sq_final = 0;
            unsigned long site_flops = 0;
            unsigned long mv_apps = 0;
            assert(in.GetNCol() == out.GetNCol());
            IndexType ncol = in.GetNCol();

            if (_params.MaxIter <= 0) {
                ZeroVec(out, SUBSET_ODD);
                return std::vector<LinearSolverResults>(ncol, LinearSolverResults());
            } else {
                std::vector<LinearSolverResults> res(ncol);
                for (int col = 0; col < ncol; ++col) {
                    (solver_wrapper)(&(out.get(col)), &(in.get(col)), _params.RsdTarget, n_iters,
                                     rsd_sq_final, site_flops, mv_apps, isign, _params.VerboseP,
                                     ODD);
                    res[col].n_count = n_iters;
                    res[col].resid = sqrt(rsd_sq_final);
                    res[col].resid_type = resid_type;
                }
                return res;
            }
        }

    private:
        const LinearSolverParamsBase &_params;

        QPhiXMRSmootherT<FT> mr_solver;
        QPhiXUnprecSolverT<FT> solver_wrapper;
    };

    using MRSmootherQPhiX = MRSmootherQPhiXT<double>;
    using MRSmootherQPhiXF = MRSmootherQPhiXT<float>;

    // Single Precision, for null space solving
    template <typename FT>
    class MRSmootherQPhiXTEO : public LinearSolverNoPrecon<QPhiXSpinorT<FT>> {
    public:
        MRSmootherQPhiXTEO(QPhiXWilsonCloverLinearOperatorT<FT> &M,
                           const LinearSolverParamsBase &params)
            : LinearSolverNoPrecon<QPhiXSpinorT<FT>>(M, params),
              _params(params),
              mr_smoother(M.getQPhiXOp(), params.MaxIter, params.Omega) {}

        MRSmootherQPhiXTEO(QPhiXWilsonCloverEOLinearOperatorT<FT> &M,
                           const LinearSolverParamsBase &params)
            : LinearSolverNoPrecon<QPhiXSpinorT<FT>>(M, params),
              _params(params),
              mr_smoother(M.getQPhiXOp(), params.MaxIter, params.Omega) {}

        void operator()(QPhiXSpinorT<FT> &out, const QPhiXSpinorT<FT> &in,
                        ResiduumType resid_type = RELATIVE,
                        InitialGuess guess = InitialGuessNotGiven) const {
            (void)resid_type;
            (void)guess;

            const int isign = 1;
            int n_iters = 0;
            double rsd_sq_final = 0;
            unsigned long site_flops = 0;
            unsigned long mv_apps = 0;
            assert(out.GetNCol() == in.GetNCol());
            IndexType ncol = out.GetNCol();

            if (_params.MaxIter <= 0) {
                CopyVec(out, in, SUBSET_ODD);
            } else {
                for (int col = 0; col < ncol; ++col)
                    (mr_smoother)(out.getCB(col, ODD).get(), in.getCB(col, ODD).get(),
                                  _params.RsdTarget, n_iters, rsd_sq_final, site_flops, mv_apps,
                                  isign, _params.VerboseP, ODD);
            }
        }

    private:
        const LinearSolverParamsBase &_params;

        // QPhiXMRSmootherT<FT> mr_smoother;
        QPhiXMRSolverT<FT> mr_smoother;
    };

    using MRSmootherQPhiXEO = MRSmootherQPhiXTEO<double>;
    using MRSmootherQPhiXEOF = MRSmootherQPhiXTEO<float>;
} // namespace MG

#endif /* INCLUDE_LATTICE_QPHIX_INVMR_QPHIX_H_ */
