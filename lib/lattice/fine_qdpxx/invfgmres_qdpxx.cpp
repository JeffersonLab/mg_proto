/*
 * invfgmres.cpp
 *
 *  Created on: Mar 21, 2017
 *      Author: bjoo
 */

#include <lattice/fine_qdpxx/invfgmres_qdpxx.h>

namespace MG {
    namespace QDPFGMRES {
        // Givens rotation.
        //   There are a variety of ways to choose the rotations
        //   which can do the job. I employ the method given by Saad
        //   in Iterative Methods (Sec. 6.5.9 (eq. 6.80)
        //
        //  [  conj(c) conj(s)  ] [ h_jj    ] = [ r ]
        //  [    -s       c     ] [ h_j+1,j ]   [ 0 ]
        //
        //  We know that h_j+1,j is a vector norm so imag(h_{j+1,j} = 0)
        //
        //  we have: s = h_{j+1,j} / t
        //           c = h_jj / t
        //
        //   t=sqrt( norm2(h_jj) + h_{j+1,j}^2 ) is real and nonnegative
        //
        //  so in this case s, is REAL and nonnegative (since t is and h_{j+1,j} is
        //  but  c is in general complex valued.
        //
        //
        //  using this we find r = conj(c) h_jj + conj(s) h_{j+1,j}
        //                        = (1/t)* [  conj(h_jj)*h_jj + h_{j+1,j}*h_{j+1,h} ]
        //                        = (1/t)* [  norm2(h_jj) + h_{j+1,j}^2 ]
        //                        = (1/t)* [ t^2 ] = t
        //
        //  Applying this to a general 2 vector
        //
        //   [ conj(c) conj(s) ] [ a ] = [ r_1 ]
        //   [   -s      c     ] [ b ]   [ r_2 ]
        //
        //   we have r_1 = conj(c)*a + conj(s)*b  = conj(c)*a + s*b  since s is real, nonnegative
        //      and  r_2 = -s*a + c*b
        //
        //  NB: In this setup we choose the sine 's' to be real in the rotation.
        //      This is in contradistinction from LAPACK which typically chooses the cosine 'c' to be real
        //
        //
        // There are some special cases:
        //   if  h_jj and h_{j+1,j} are both zero we can choose any s and c as long as |c|^2 + |s|^2 =1
        //   Keeping with the notation that s is real and nonnegative we choose
        //   if    h_jj != 0 and h_{j+1,h) == 0 => c = sgn(conj(h_jj)), s = 0, r = | h_jj |
        //   if    h_jj == 0 and h_{j+1,j} == 0 => c = 0, s = 1,  r = 0 = h_{j+1,j}
        //   if    h_jj == 0 and h_{j+1,j} != 0 => c = 0, s = 1,  r = h_{j+1,j} = h_{j+1,j}
        //   else the formulae we computed.

        /*! Given  a marix H, construct the rotator so that H(row,col) = r and H(row+1,col) = 0
 *
 *  \param col  the column Input
 *  \param  H   the Matrix Input
 */

        Givens::Givens(int col, const multi2d<DComplex> &H) : col_(col) {
            DComplex f = H(col_, col_);
            DComplex g = H(col_, col_ + 1);

            if (toBool(real(f) == Double(0) && imag(f) == Double(0))) {

                // h_jj is 0
                c_ = DComplex(0);
                s_ = DComplex(1);
                r_ = g; // Handles the case when g is also zero
            } else {
                if (toBool(real(g) == Double(0) && imag(g) == Double(0))) {
                    s_ = DComplex(0);
                    c_ = conj(f) /
                         sqrt(norm2(
                             f)); //   sgn( conj(f) ) = conj(f) / | conj(f) |  = conj(f) / | f |
                    r_ = DComplex(sqrt(norm2(f)));
                } else {
                    // Revisit this with
                    Double t = sqrt(norm2(f) + norm2(g));
                    r_ = t;
                    c_ = f / t;
                    s_ = g / t;
                }
            }
        }

        /*! Apply the rotation to column col of the matrix H. The
 *  routine affects col and col+1.
 *
 *  \param col  the columm
 *  \param  H   the matrix
 */

        void Givens::operator()(int col, multi2d<DComplex> &H) {
            if (col == col_) {
                // We've already done this column and know the answer
                H(col_, col_) = r_;
                H(col_, col_ + 1) = 0;
            } else {
                int row = col_; // The row on which the rotation was defined
                DComplex a = H(col, row);
                DComplex b = H(col, row + 1);
                H(col, row) = conj(c_) * a + conj(s_) * b;
                H(col, row + 1) = -s_ * a + c_ * b;
            }
        }

        /*! Apply rotation to Column Vector v */
        void Givens::operator()(multi1d<DComplex> &v) {
            DComplex a = v(col_);
            DComplex b = v(col_ + 1);
            v(col_) = conj(c_) * a + conj(s_) * b;
            v(col_ + 1) = -s_ * a + c_ * b;
        }

        void FlexibleArnoldiT(int n_krylov, const Real &rsd_target,
                              const LinearOperator<LatticeFermion> &A, // Operator
                              const LinearOperator<LatticeFermion> *M, // Preconditioner
                              multi1d<LatticeFermion> &V, multi1d<LatticeFermion> &Z,
                              multi2d<DComplex> &H, multi1d<Givens *> &givens_rots,
                              multi1d<DComplex> &c, int &ndim_cycle, ResiduumType resid_type,
                              bool VerboseP)

        {
            (void)resid_type;
            const Subset &s = QDP::all; // Linear Operator Subset
            ndim_cycle = 0;

            int level = A.GetLevel();

            // Work by columns:
            for (int j = 0; j < n_krylov; ++j) {

                // Check convention... Z=solution, V=source
                // Here we have an opportunity to not precondition...
                // If M is a nullpointer.
                if (M != nullptr) {
                    (*M)(Z[j], V[j]); // z_j = M^{-1} v_j
                } else {
                    Z[j] = V[j]; // Vector assignment " copy "
                }

                LatticeFermion w;
                A(w, Z[j], LINOP_OP); // w  = A z_

                // Fill out column j
                for (int i = 0; i <= j; ++i) {
                    H(j, i) = innerProduct(V[i], w, s); //  Inner product
                    w[s] -= H(j, i) * V[i];             // y = y - alpha x = CAXPY
                }

                Double wnorm = sqrt(norm2(w, s)); //  NORM
                H(j, j + 1) = DComplex(wnorm);

                // In principle I should check w_norm to be 0, and if it is I should
                // terminate: Code Smell -- get rid of 1.0e-14.
                if (toBool(fabs(wnorm) < Double(1.0e-14))) {

                    // If wnorm = 0 exactly, then we have converged exactly
                    // Replay Givens rots here, how to test?

                    ndim_cycle = j;
                    return;
                }

                Double invwnorm = Double(1) / wnorm;
                V[j + 1] = invwnorm * w; // SCAL

                // Apply Existing Givens Rotations to this column of H
                for (int i = 0; i < j; ++i) { (*givens_rots[i])(j, H); }

                // Compute next Givens Rot for this column
                givens_rots[j] = new Givens(j, H);

                (*givens_rots[j])(j, H); // Apply it to H
                (*givens_rots[j])(c);    // Apply it to the c vector

                Double accum_resid = fabs(real(c[j + 1]));

                // j-ndeflate is the 0 based iteration count
                // j-ndeflate+1 is the 1 based human readable iteration count

                if (VerboseP) {

                    MasterLog(INFO,
                              "FLEXIBLE ARNOLDI: level=%d Iter=%d  || r ||=%16.8e  Target=%16.8e",
                              level, j + 1, toDouble(accum_resid), toDouble(rsd_target));
                }
                ndim_cycle = j + 1;
                if (toBool(accum_resid <= rsd_target)) { return; }
            }
        }
    } // namespace QDPFGMRES

    FGMRESSolverQDPXX::FGMRESSolverQDPXX(const LinearOperator<LatticeFermion> &M,
                                         const MG::LinearSolverParamsBase &params,
                                         const LinearOperator<LatticeFermion> *prec)
        : LinearSolver<LatticeFermion>(M, params, prec) {
        // Initialize stuff
        InitMatrices();
    }

    //! Initialize the internal matrices
    void FGMRESSolverQDPXX::InitMatrices() {

        H_.resize(_params.NKrylov, _params.NKrylov + 1); // This is odd. Shouldn't it be

        V_.resize(_params.NKrylov + 1);
        Z_.resize(_params.NKrylov + 1);
        givens_rots_.resize(_params.NKrylov + 1);
        c_.resize(_params.NKrylov + 1);
        eta_.resize(_params.NKrylov);

        for (int col = 0; col < _params.NKrylov; col++) {
            for (int row = 0; row < _params.NKrylov + 1; row++) {
                H_(col, row) = zero; // COMPLEX ZERO
            }
        }

        for (int row = 0; row < _params.NKrylov + 1; row++) {
            V_[row] = zero; // BLAS ZERO
            Z_[row] = zero; // BLAS ZERO
            c_[row] = zero; // COMPLEX ZERO
            givens_rots_[row] = nullptr;
        }

        for (int row = 0; row < _params.NKrylov; row++) { eta_[row] = zero; }
    }

    std::vector<LinearSolverResults> FGMRESSolverQDPXX::operator()(LatticeFermion &out,
                                                                   const LatticeFermion &in,
                                                                   ResiduumType resid_type,
                                                                   InitialGuess guess) const {
        (void)guess;

        LinearSolverResults res; // Value to return
        res.resid_type = resid_type;
        int level = _M.GetLevel();

        const Subset s = QDP::all;

        Double norm_rhs = sqrt(norm2(in, s)); //  || b ||                      BLAS: NORM2
        Double target = _params.RsdTarget;

        if (resid_type == RELATIVE) {
            target *= norm_rhs; // Target  || r || < || b || RsdTarget
        }

        // Compute ||r||
        LatticeFermion r = zero;   // BLAS: ZERO
        LatticeFermion tmp = zero; // BLAS: COPY
        r[s] = in;
        (_M)(tmp, out, LINOP_OP);
        r[s] -= tmp; // BLAS: X=X-Y

        // The current residuum
        Double r_norm = sqrt(norm2(r, s)); // BLAS: NORM

        // Initialize iterations
        int iters_total = 0;
        if (_params.VerboseP) {
            MasterLog(INFO, "FGMRES: level=%d iters=%d  || r ||=%16.8e  Target=%16.8e", level,
                      iters_total, toDouble(r_norm), toDouble(target));
        }

        if (toBool(r_norm < target)) {
            res.n_count = 0;
            res.resid = toDouble(r_norm);
            if (resid_type == ABSOLUTE) {
                if (_params.VerboseP) {
                    MasterLog(INFO,
                              "FGMRES: level=%d Solve Converged: iters=%d  Final (absolute) || r "
                              "||=%16.8e",
                              level, res.resid);
                }
            } else {
                res.resid /= toDouble(norm_rhs);
                if (_params.VerboseP) {
                    MasterLog(
                        INFO,
                        "FGMRES: level=%d Solve Converged: iters=%d  Final (relative) || r ||/|| "
                        "b ||=%16.8e",
                        level, res.resid);
                }
            }
            return std::vector<LinearSolverResults>(1, res);
        }

        int n_cycles = 0;

        // We are done if norm is sufficiently accurate,
        bool finished = toBool(r_norm <= target);

        // We keep executing cycles until we are finished
        while (!finished) {

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
            for (int j = 0; j < c_.size(); ++j) { c_[j] = DComplex(0); }
            c_[0] = r_norm;

            // Set up initial V[0] = rhs / || r^2 ||
            // and since we are solving for A delta x = r
            // the rhs is 'r'
            //
            Double beta_inv = Double(1) / r_norm;
            V_[0][s] = beta_inv * r; // BLAS: VSCAL

            // Carry out Flexible Arnoldi process for the cycle

            // We are solving for the defect:   A dx = r
            // so the RHS in the Arnoldi process is 'r'
            // NB: We recompute a true 'r' after every cycle
            // So in the cycle we could in principle
            // use reduced precision... TBInvestigated.

            FlexibleArnoldi(n_krylov, target, V_, Z_, H_, givens_rots_, c_, dim, resid_type);

            int iters_this_cycle = dim;
            LeastSquaresSolve(H_, c_, eta_, dim); // Solve Least Squares System

            // Compute the correction dx = sum_j  eta_j Z_j
            LatticeFermion dx = zero; // BLAS: ZERO
            for (int j = 0; j < dim; ++j) {
                dx[s] += eta_[j] * Z_[j]; // Y = Y + AX => BLAS AXPY
            }

            // Update psi
            out[s] += dx; // BLAS: Y=Y+X => APY

            // Recompute r
            r[s] = in; // BLAS: COPY
            (_M)(tmp, out, LINOP_OP);
            r[s] -= tmp; // This 'r' will be used in next cycle as the || rhs ||: Y=Y-X

            // Recompute true norm
            r_norm = sqrt(norm2(r, s)); // BLAS: NORM2

            // Update total iters
            iters_total += iters_this_cycle;
            if (_params.VerboseP) {
                MasterLog(INFO, "FGMRES: level=%d iter=%d || r ||=%16.8e target=%16.8e", level,
                          iters_total, toDouble(r_norm), toDouble(target));
            }

            // Check if we are done either via convergence, or runnign out of iterations
            finished = toBool(r_norm <= target) || (iters_total >= _params.MaxIter);

            // Init matrices should've initialized this but just in case this is e.g. a second call or
            // something.
            for (int j = 0; j < _params.NKrylov; ++j) {
                if (givens_rots_[j] != nullptr) {
                    delete givens_rots_[j];
                    givens_rots_[j] = nullptr;
                }
            }
        }

        // Either we've exceeded max iters, or we have converged in either case set res:
        res.n_count = iters_total;
        res.resid = toDouble(r_norm);
        if (resid_type == ABSOLUTE) {

            MasterLog(INFO, "FGMRES: level=%d  Solve Done. Cycles=%d, Iters=%d || r ||=%16.8e",
                      level, n_cycles, iters_total, res.resid, _params.RsdTarget);
        } else {
            res.resid /= toDouble(norm_rhs);
            MasterLog(INFO,
                      "FGMRES: level=%d  Solve Done. Cycles=%d, Iters=%d || r ||/|| b ||=%16.8e",
                      level, n_cycles, iters_total, res.resid, _params.RsdTarget);
        }
        return std::vector<LinearSolverResults>(1, res);
    }

    void FGMRESSolverQDPXX::FlexibleArnoldi(int n_krylov, const Real &rsd_target,
                                            multi1d<LatticeFermion> &V, multi1d<LatticeFermion> &Z,
                                            multi2d<DComplex> &H,
                                            multi1d<QDPFGMRES::Givens *> &givens_rots,
                                            multi1d<DComplex> &c, int &ndim_cycle,
                                            ResiduumType resid_type) const {

        QDPFGMRES::FlexibleArnoldiT(n_krylov, rsd_target, _M, _prec, V, Z, H, givens_rots, c,
                                    ndim_cycle, resid_type, _params.VerboseP);
    }

    void FGMRESSolverQDPXX::LeastSquaresSolve(const multi2d<DComplex> &H,
                                              const multi1d<DComplex> &rhs, multi1d<DComplex> &eta,
                                              int n_cols) const

    {
        /* Assume here we have a square matrix with an extra row.
       Hence the loop counters are the columns not the rows.
       NB: For an augmented system this will change */
        eta[n_cols - 1] = rhs[n_cols - 1] / H(n_cols - 1, n_cols - 1);
        for (int row = n_cols - 2; row >= 0; --row) {
            eta[row] = rhs[row];
            for (int col = row + 1; col < n_cols; ++col) { eta[row] -= H(col, row) * eta[col]; }
            eta[row] /= H(row, row);
        }
    }

} // namespace MG
