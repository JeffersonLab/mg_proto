/*
 * invfgmres.h
 *
 *  Created on: Jan 11, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_INVFGMRES_GENERIC_H_
#define INCLUDE_LATTICE_INVFGMRES_GENERIC_H_

#include "MG_config.h"
#include "lattice/array2d.h"
#include "lattice/constants.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"
#include "utils/timer.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#undef DEBUG_SOLVER

#include "lattice/coarse/subset.h"
#include "lattice/fgmres_common.h"
#include "lattice/givens.h"

namespace MG {

    namespace {
        /** negate
 *
 * 	Flip sign on all elements
 */
        template <typename T>
        std::vector<std::complex<float>> negate(const std::vector<std::complex<T>> &x) {
            std::vector<std::complex<float>> r(x.size());
            std::transform(x.begin(), x.end(), r.begin(),
                           [](const std::complex<T> &f) { return -f; });
            return r;
        }
    } // namespace

    namespace FGMRESGeneric {

        inline void showConvergence(const std::vector<double> &residuals,
                                    const std::vector<double> &targets, std::string level, int iter,
                                    int n_cycles = -1) {
            const int ncol = residuals.size();
            int num_converged = 0;
            double avg_residual = 0.0, avg_target = 0;
            for (int col = 0; col < ncol; ++col) {
                avg_residual += std::log(residuals[col]);
                avg_target += std::log(targets[col]);
                if (residuals[col] <= targets[col]) num_converged++;
            }
            avg_residual = std::exp(avg_residual / ncol);
            avg_target = std::exp(avg_target / ncol);
            if (n_cycles < 0) {
                MasterLog(INFO,
                          "FLEXIBLE ARNOLDI: level=%s Iter=%d avg || r ||=%4.2e converged=%d avg "
                          "Target=%16.8e",
                          level.c_str(), iter, avg_residual, num_converged, avg_target);
            } else {
                MasterLog(INFO,
                          "FLEXIBLE ARNOLDI: level=%s Cycles=%d Iter=%d avg || r ||=%4.2e "
                          "converged=%d avg "
                          "Target=%16.8e",
                          level.c_str(), n_cycles, iter, avg_residual, num_converged, avg_target);
            }
        }

        template <typename ST>
        void FlexibleArnoldiT(
            int n_krylov, const std::vector<double> &rsd_target,
            const LinearOperator<ST> &A, // Operator
            const LinearOperator<ST> *M, // Preconditioner
            std::vector<ST *>
                &V, // Nuisance: Need constructor free way to make these. Init functions?
            std::vector<ST *> &Z, ST &w, std::vector<Array2d<std::complex<double>>> &H,
            std::vector<std::vector<Givens *>> &givens_rots,
            std::vector<std::vector<std::complex<double>>> &c, int &ndim_cycle,
            ResiduumType resid_type, bool VerboseP, std::string level)

        {
            (void)resid_type;

            ndim_cycle = 0;
            const CBSubset &subset = A.GetSubset();
            IndexType ncol = V[0]->GetNCol();
            assert(ncol == w.GetNCol());
            assert((unsigned int)ncol == H.size());
            assert((unsigned int)ncol == givens_rots.size());
            assert((unsigned int)ncol == c.size());

            if (VerboseP) {
                MasterLog(INFO,
                          "FLEXIBLE ARNOLDI: level=%s Flexible Arnoldi Cycle: ", level.c_str());
            }

            // Work by columns:
            for (int j = 0; j < n_krylov; ++j) {

                // Check convention... Z=solution, V=source
                // Here we have an opportunity to not precondition...
                // If M is a nullpointer.
                if (M != nullptr) {

                    // solve z_j = M^{-1} v_j
                    //
                    // solve into tmpsolve in case the solution process changes the
                    // The Z_j are pre-zeroed
                    // The V_js are only changed on the subset so off-subset should be zeroed
                    // non-subset part.
                    // But I will go through a tmpsolve temporary because
                    // a proper unprec solver may overwrite the off checkerboard parts with a reconstruct
                    // etc.
                    //
                    Timer::TimerAPI::startTimer("FGMRESSolverGeneric/preconditioner/level" + level);

                    (*M)(*Z[j], *(V[j])); // z_j = M^{-1} v_j

                    Timer::TimerAPI::stopTimer("FGMRESSolverGeneric/preconditioner/level" + level);
                } else {
                    CopyVec(*(Z[j]), *(V[j]), subset); // Vector assignment " copy "
                }

#ifdef DEBUG_SOLVER
                {
                    MasterLog(
                        DEBUG,
                        "FLEXIBLE ARNOLDI: level=%s norm of Z_j = %16.8e norm of V_j = %16.8e",
                        level.c_str(), Norm2Vec(*(Z[j]), subset), Norm2Vec(*(V[j]), subset));
                }
#endif

                Timer::TimerAPI::startTimer("FGMRESSolverGeneric/operatorA/level" + level);
                A(w, *(Z[j]), LINOP_OP); // w  = A z_
                Timer::TimerAPI::stopTimer("FGMRESSolverGeneric/operatorA/level" + level);

                // Fill out column j
                for (int i = 0; i <= j; ++i) {
                    std::vector<std::complex<double>> hji =
                        InnerProductVec(*(V[i]), w, subset);                      //  Inner product
                    for (int col = 0; col < ncol; ++col) H[col](j, i) = hji[col]; //  Inner product

                    // w[s] -= H(j,i)* V[i];                     // y = y - alpha x = CAXPY
                    AxpyVec(negate(hji), *(V[i]), w, subset);
                }

                std::vector<double> wnorm = aux::sqrt(Norm2Vec(w, subset)); //  NORM
#ifdef DEBUG_SOLVER
                for (int col = 0; col < ncol; ++col)
                    MasterLog(DEBUG, "FLEXIBLE ARNOLDI: level=%s j=%d wnorm=%16.8e\n",
                              level.c_str(), j, wnorm[col]);
#endif

                for (int col = 0; col < ncol; ++col)
                    H[col](j, j + 1) = std::complex<double>(wnorm[col], 0);

                // In principle I should check w_norm to be 0, and if it is I should
                // terminate: Code Smell -- get rid of 1.0e-14.
                for (int col = 0; col < ncol; ++col) {
                    if (wnorm[col] < 1.0e-14) {

                        // If wnorm = 0 exactly, then we have converged exactly
                        // Replay Givens rots here, how to test?
                        if (VerboseP) {
                            MasterLog(INFO, "FLEXIBLE ARNOLDI: level=%s Converged at iter = %d ",
                                      level.c_str(), j + 1);
                        }
                        ndim_cycle = j;
                        return;
                    }
                }

                std::vector<double> invwnorm(ncol);
                for (int col = 0; col < ncol; ++col) invwnorm[col] = 1.0 / wnorm[col];
                // V[j+1] = invwnorm*w;                           // SCAL
                ZeroVec(*(V[j + 1]), subset);
                AxpyVec(invwnorm, w, *(V[j + 1]), subset);

                // Apply Existing Givens Rotations to this column of H
                for (int col = 0; col < ncol; ++col) {
                    for (int i = 0; i < j; ++i) { (*givens_rots[col][i])(j, H[col]); }
                }

                // Compute next Givens Rot for this column
                for (int col = 0; col < ncol; ++col) givens_rots[col][j] = new Givens(j, H[col]);

                for (int col = 0; col < ncol; ++col) {
                    (*givens_rots[col][j])(j, H[col]); // Apply it to H
                    (*givens_rots[col][j])(c[col]);    // Apply it to the c vector
                }

                // j-ndeflate is the 0 based iteration count
                // j-ndeflate+1 is the 1 based human readable iteration count

                ndim_cycle = j + 1;
                bool all_converged = true;
                std::vector<double> residuals(ncol);
                for (int col = 0; col < ncol; ++col) {
                    double accum_resid = fabs(real(c[col][j + 1]));
                    residuals[col] = accum_resid;
                    if (accum_resid > rsd_target[col]) all_converged = false;
                }
                if (VerboseP) { showConvergence(residuals, rsd_target, level, j + 1); }
                if (all_converged) return;
            } // while
        }     // function

        template <typename ST> class FGMRESSolverGeneric : public LinearSolver<ST> {
        public:
            using LinearSolver<ST>::_M;
            using LinearSolver<ST>::_params;
            using LinearSolver<ST>::_prec;
            using AuxiliarySpinors<ST>::tmp;

            /**
             * Constructor
             *
             * \param M: operator
             * \param params: linear solver params
             * \param prec: preconditioner
             */

            FGMRESSolverGeneric(const LinearOperator<ST> &M, const LinearSolverParamsBase &params,
                                const LinearOperator<ST> *prec = nullptr, std::string prefix = "")
                : LinearSolver<ST>(M, set_params_defaults(params), prec) {

                initialize(0);

                _prefix = std::to_string(M.GetLevel()) + prefix;
                Timer::TimerAPI::addTimer("FGMRESSolverGeneric/operator()/level" + _prefix);
                Timer::TimerAPI::addTimer("FGMRESSolverGeneric/operatorA/level" + _prefix);
                Timer::TimerAPI::addTimer("FGMRESSolverGeneric/preconditioner/level" + _prefix);
            }

            ~FGMRESSolverGeneric() { destroy(); }

        private:
            static LinearSolverParamsBase
            set_params_defaults(const LinearSolverParamsBase &params) {
                LinearSolverParamsBase p(params);
                if (p.MaxIter < 0) p.MaxIter = 100;
                if (p.NKrylov <= 0) p.NKrylov = 8;
                if (p.RsdTarget < 0) p.RsdTarget = 1e-3;
                return p;
            }

            void initialize(IndexType ncol) const {
                if ((int)H_.size() == ncol) return;

                destroy();

                H_.resize(ncol);
                givens_rots_.resize(ncol);
                c_.resize(ncol);
                eta_.resize(ncol);

                V_.resize(_params.NKrylov + 1);
                Z_.resize(_params.NKrylov + 1);

                for (int i = 0; i < _params.NKrylov + 1; ++i) {
                    V_[i] = new ST(_M.GetInfo(), ncol);
                    Z_[i] = new ST(_M.GetInfo(), ncol);
                }

                for (int row = 0; row < _params.NKrylov + 1; row++) {
                    ZeroVec(*(V_[row]), SUBSET_ALL); // BLAS ZERO
                    ZeroVec(*(Z_[row]), SUBSET_ALL); // BLAS ZERO
                }

                for (int col = 0; col < ncol; ++col) {
                    H_[col].resize(_params.NKrylov,
                                   _params.NKrylov + 1); // This is odd. Shouldn't it be

                    givens_rots_[col].resize(_params.NKrylov + 1);
                    c_[col].resize(_params.NKrylov + 1);
                    eta_[col].resize(_params.NKrylov);

                    for (int i = 0; i < _params.NKrylov; i++) {
                        for (int row = 0; row < _params.NKrylov + 1; row++) {
                            H_[col](i, row) = std::complex<double>(0, 0); // COMPLEX ZERO
                        }
                    }

                    for (int row = 0; row < _params.NKrylov + 1; row++) {
                        c_[col][row] = std::complex<double>(0, 0); // COMPLEX ZERO
                        givens_rots_[col][row] = nullptr;
                    }

                    for (int row = 0; row < _params.NKrylov; row++) {
                        eta_[col][row] = std::complex<double>(0, 0);
                    }
                }
            }

            void destroy() const {
                for (unsigned int i = 0; i < V_.size(); ++i) {
                    delete V_[i];
                    delete Z_[i];
                    for (unsigned int col = 0; col < H_.size(); ++col) {
                        if (givens_rots_[col][i] != nullptr) delete givens_rots_[col][i];
                    }
                }
            }

        public:
            std::vector<LinearSolverResults>
            operator()(ST &out, const ST &in, ResiduumType resid_type = RELATIVE,
                       InitialGuess guess = InitialGuessNotGiven) const override {

                IndexType ncol = in.GetNCol();
                if (_params.MaxIter <= 0)
                    return std::vector<LinearSolverResults>(ncol, LinearSolverResults());

                Timer::TimerAPI::startTimer("FGMRESSolverGeneric/operator()/level" + _prefix);

                assert(in.GetNCol() == out.GetNCol());

                const CBSubset &subset = _M.GetSubset();

                initialize(ncol);
                std::vector<LinearSolverResults> res(ncol); // Value to return
                for (int col = 0; col < ncol; ++col) res[col].resid_type = resid_type;

                std::vector<double> norm_rhs =
                    aux::sqrt(Norm2Vec(in, subset)); //  || b || BLAS: NORM2
                std::vector<double> target(ncol, _params.RsdTarget);

                if (resid_type == RELATIVE) {
                    for (int col = 0; col < ncol; ++col)
                        target[col] *= norm_rhs[col]; // Target  || r || < || b || RsdTarget
                }

                const LatticeInfo &in_info = in.GetInfo();
                const LatticeInfo &out_info = out.GetInfo();
                AssertCompatible(in_info, _M.GetInfo());
                AssertCompatible(out_info, _M.GetInfo());

                // Temporaries - passed into flexible Arnoldi
                std::shared_ptr<ST> w = tmp(in_info, ncol);

                // Compute ||r||
                std::shared_ptr<ST> r = tmp(in_info, ncol);
                ZeroVec(*r, subset); // BLAS: ZERO
                std::shared_ptr<ST> t = tmp(in_info, ncol);
                ZeroVec(*t, subset); // BLAS: COPY
                CopyVec(*r, in, subset);

                std::vector<double> r_norm;
                if (guess == InitialGuessGiven) {
                    (_M)(*t, out, LINOP_OP);

                    // r[s] -=t
                    // r_norm = sqrt(norm2(r,s))
                    r_norm = aux::sqrt(XmyNorm2Vec(*r, *t, subset));
                } else {
                    r_norm = norm_rhs;
                    ZeroVec(out, subset);
                }

                // Initialize iterations
                int iters_total = 0;
                if (_params.VerboseP) { showConvergence(r_norm, target, _prefix, iters_total); }

                bool all_converged = true;
                for (int col = 0; col < ncol; ++col) {
                    if (r_norm[col] < target[col]) {
                        res[col].n_count = 0;
                        res[col].resid = r_norm[col];
                    } else {
                        all_converged = false;
                    }
                }

                if (all_converged) {
                    Timer::TimerAPI::stopTimer("FGMRESSolverGeneric/operator()/level" + _prefix);
                    return res;
                }

                int n_cycles = 0;

                // We keep executing cycles until we are finished
                while (iters_total < _params.MaxIter) {

                    // If not finished, we should do another cycle with RHS='r' to find dx to add to psi.
                    ++n_cycles;

                    int dim; // dim at the end of cycle (in case we terminate in-cycle
                    int n_krylov = _params.NKrylov;

                    // We are either first cycle, or
                    // We have no deflation subspace ie we are regular FGMRES
                    // and we are just restarting
                    //
                    // Set up initial vector c = [ beta, 0 ... 0 ]^T
                    // In this case beta should be the r_norm, ie || r || (=|| b || for the first cycle)
                    //
                    // NB: We will have a copy of this called 'g' onto which we will
                    // apply Givens rotations to get an inline estimate of the residuum
                    for (int col = 0; col < ncol; ++col) {
                        for (unsigned int j = 0; j < c_[col].size(); ++j) {
                            c_[col][j] = std::complex<double>(0);
                        }
                        c_[col][0] = r_norm[col];
                    }

                    // Set up initial V[0] = rhs / || r^2 ||
                    // and since we are solving for A delta x = r
                    // the rhs is 'r'
                    //
                    std::vector<double> beta_inv(ncol);
                    for (int col = 0; col < ncol; ++col) beta_inv[col] = 1.0 / r_norm[col];
                    //  V_[0] = beta_inv * r;                       // BLAS: VSCAL
                    ZeroVec(*(V_[0]), subset);
                    AxpyVec(beta_inv, *r, *(V_[0]), subset);

                    // Carry out Flexible Arnoldi process for the cycle

                    // We are solving for the defect:   A dx = r
                    // so the RHS in the Arnoldi process is 'r'
                    // NB: We recompute a true 'r' after every cycle
                    // So in the cycle we could in principle
                    // use reduced precision... TBInvestigated.
                    FlexibleArnoldi(n_krylov, target, V_, Z_, *w, H_, givens_rots_, c_, dim,
                                    resid_type, _prefix);

                    int iters_this_cycle = dim;
                    assert(dim > 0);
                    if (dim > 0)
                        for (int col = 0; col < ncol; ++col)
                            LeastSquaresSolve(H_[col], c_[col], eta_[col],
                                              dim); // Solve Least Squares System

                    // Compute the correction dx = sum_j  eta_j Z_j
                    // LatticeFermion dx = zero;                         // BLAS: ZERO
                    for (int j = 0; j < dim; ++j) {

                        std::vector<std::complex<float>> alpha(ncol);
                        for (int col = 0; col < ncol; ++col)
                            alpha[col] = std::complex<float>(std::real(eta_[col][j]),
                                                             std::imag(eta_[col][j]));
                        AxpyVec(alpha, *(Z_[j]), out, subset); // Y = Y + AX => BLAS AXPY
                    }

                    // Update total iters
                    iters_total += iters_this_cycle;

                    // Recompute r
                    if (iters_total < _params.MaxIter || _params.RsdTarget > 0.0) {
                        CopyVec(*r, in, subset);
                        (_M)(*t, out, LINOP_OP);
                        r_norm = aux::sqrt(XmyNorm2Vec(*r, *t, subset));
                    }

                    if (_params.VerboseP) { showConvergence(r_norm, target, _prefix, iters_total); }

                    // Init matrices should've initialized this but just in case this is e.g. a second call
                    // or something.
                    for (int col = 0; col < ncol; ++col) {
                        for (int j = 0; j < _params.NKrylov; ++j) {
                            if (givens_rots_[col][j] != nullptr) {
                                delete givens_rots_[col][j];
                                givens_rots_[col][j] = nullptr;
                            }
                        }
                    }

                    // Check if all columns are converged
                    bool finished = true;
                    for (int col = 0; col < ncol; ++col)
                        if (r_norm[col] > target[col]) finished = false;
                    if (finished) break;

                } // Next Cycle...

                if (_params.VerboseP) {
                    showConvergence(r_norm, target, _prefix, iters_total, n_cycles);
                }

                // Either we've exceeded max iters, or we have converged in either case set res:
                for (int col = 0; col < ncol; ++col) {
                    res[col].n_count = iters_total;
                    res[col].resid = r_norm[col];
                    if (resid_type == RELATIVE) { res[col].resid /= norm_rhs[col]; }
                }
                Timer::TimerAPI::stopTimer("FGMRESSolverGeneric/operator()/level" + _prefix);
                return res;
            }

            void FlexibleArnoldi(int n_krylov, const std::vector<double> &rsd_target,
                                 std::vector<ST *> &V, std::vector<ST *> &Z, ST &w,
                                 std::vector<Array2d<std::complex<double>>> &H,
                                 std::vector<std::vector<Givens *>> &givens_rots,
                                 std::vector<std::vector<std::complex<double>>> &c, int &ndim_cycle,
                                 ResiduumType resid_type, std::string prefix) const {

                FlexibleArnoldiT<ST>(n_krylov, rsd_target, _M, _prec, V, Z, w, H, givens_rots, c,
                                     ndim_cycle, resid_type, _params.VerboseP, prefix);
            }

            void LeastSquaresSolve(const Array2d<std::complex<double>> &H,
                                   const std::vector<std::complex<double>> &rhs,
                                   std::vector<std::complex<double>> &eta, int n_cols) const {
                /* Assume here we have a square matrix with an extra row.
             Hence the loop counters are the columns not the rows.
             NB: For an augmented system this will change */
                eta[n_cols - 1] = rhs[n_cols - 1] / H(n_cols - 1, n_cols - 1);
                for (int row = n_cols - 2; row >= 0; --row) {
                    eta[row] = rhs[row];
                    for (int col = row + 1; col < n_cols; ++col) {
                        eta[row] -= H(col, row) * eta[col];
                    }
                    eta[row] /= H(row, row);
                }
            }

        private:
            std::string _prefix;

            // These can become state variables, as they will need to be
            // handed around
            mutable std::vector<Array2d<std::complex<double>>> H_; // The H matrix
            mutable std::vector<ST *> V_;                          // K(A)
            mutable std::vector<ST *> Z_;                          // K(MA)
            mutable std::vector<std::vector<Givens *>> givens_rots_;

            // This is the c = V^H_{k+1} r vector (c is frommers Notation)
            // For regular FGMRES I need to keep only the basis transformed
            // version of it, for rotating by the Q's (I call this 'g')
            // However, for FGMRES-DR I need this because I need
            //  || c - H eta || to extend the G_k matrix to form V_{k+1}
            // it is made of the previous v_{k+1} by explicitly evaluating c
            // except at the end of the first cycle when the V_{k+1} are not
            // yet available

            // Once G_k+1 is available, I can form the QR decomposition
            // and set  my little g = Q^H and then as I do the Arnoldi
            // I can then work it with the Givens rotations...

            mutable std::vector<std::vector<std::complex<double>>> c_;
            mutable std::vector<std::vector<std::complex<double>>> eta_;
        };

    } // namespace FGMRESGeneric
} // namespace MG

#endif /* TEST_QDPXX_INVFGMRES_H_ */
