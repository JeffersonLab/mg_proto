/*
 * solver.h
 *
 *  Created on: Jan 10, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_SOLVER_H_
#define INCLUDE_LATTICE_SOLVER_H_

#include "lattice/coarse/subset.h"
#include "lattice/lattice_info.h"
#include "lattice/linear_operator.h"
#include "utils/auxiliary.h"
#include <stdexcept>
#include <vector>

namespace MG {

    /**
     * Stopping criterion for solving linear systems.
     */

    enum ResiduumType {
        /// ||M*x-b|| <= RsdTol
        ABSOLUTE,

        /// ||M*x-b|| <= RsdTol * ||b||
        RELATIVE,

        /// ???
        INVALID
    };

    /**
     * Whether an initial is provided.
     */

    enum InitialGuess {
        /// No initial guess given
        InitialGuessNotGiven,

        // Initial guess given
        InitialGuessGiven
    };

    /**
     * Linear solver performance results
     */

    struct LinearSolverResults {
        /// Stopping criterion used
        ResiduumType resid_type;

        /// Number of iterations
        int n_count;

        /// Residual norm of the solution ||M*x-b||
        double resid;
    };

    /**
     * Linear solver parameters
     */

    class LinearSolverParamsBase {
    public:
        /// Tolerance for the stopping criterion
        double RsdTarget;
        /// Maximum number of iterations
        int MaxIter;
        /// Print convergence information
        bool VerboseP;
        /// Maximum size of the Krylov space (used by FGMRES)
        int NKrylov;
        /// Relaxation parameter (used by MR)
        double Omega;

        LinearSolverParamsBase() {
            RsdTarget = 0.0;
            MaxIter = -1;
            VerboseP = false;
            NKrylov = 0;
            Omega = 0.0;
        }
    };

    /**
     * Solver for linear operators
     *
     * \param Spinor: vector class
     *
     * operator(x,b) set x so that M*x approximates b.
     */

    template <typename Spinor_t> class LinearSolver : public LinearOperator<Spinor_t> {
    public:
        /// Vector class
        using Spinor = Spinor_t;

        /**
         * Compute the solution.
         *
         * \param[out] out: solution vector
         * \param in: input vector
         * \param resid_type: stopping criterion (RELATIVE or ABSOLUTE)
         * \param guess: Whether the initial is provided
         */

        virtual std::vector<LinearSolverResults>
        operator()(Spinor &out, const Spinor &in, ResiduumType resid_type = RELATIVE,
                   InitialGuess guess = InitialGuessNotGiven) const = 0;

        /**
         * Compute the solution of a linear system as an operator
         *
         * \overload
         */

        void operator()(Spinor &out, const Spinor &in, IndexType type) const override {
            if (type != LINOP_OP) throw std::runtime_error("Unsupported type!");

            // A linear solver stopping at ||M*x-b|| <= ||b||*eps behaves as a linear operator of
            // M^{-1} with error level eps
            operator()(out, in, RELATIVE, InitialGuessNotGiven);
        }

        /**
         * Set a preconditioner (optional)
         *
         * \param prec: new preconditioner
         *
         * \throw std::runtime_error if the solver does not allow preconditioning
         */

        virtual void SetPrec(const LinearOperator<Spinor> *prec = nullptr) const { _prec = prec; }

        /**
         * Get the current preconditioner
         */

        virtual const LinearOperator<Spinor> *GetPrec() const { return _prec; }

        /**
         * Return the lattice information
         */

        const LatticeInfo &GetInfo() const override { return _M.GetInfo(); }

        /**
         * Return the support of the operator (SUBSET_EVEN, SUBSET_ODD, SUBSET_ALL)
         */

        const CBSubset &GetSubset() const override { return _M.GetSubset(); }

        virtual ~LinearSolver() {}

    protected:
        /**
         * Constructor
         *
         * \param M: operator
         * \param params: linear solver params
         * \param prec: preconditioner
         */

        LinearSolver(const LinearOperator<Spinor> &M, const LinearSolverParamsBase &params,
                     const LinearOperator<Spinor> *prec = nullptr)
            : _M(M), _params(params), _prec(prec) {

            // Auxiliary vectors are managed by M
            AuxiliarySpinors<Spinor>::subrogateTo(&M);
        }

    public:
        /// Linear operator
        const LinearOperator<Spinor> &_M;
        /// Linear solver params
        const LinearSolverParamsBase _params;
        /// Preconditioner
        mutable const LinearOperator<Spinor> *_prec;
    };

    /**
     * Solver for linear operators with no support for preconditioners
     *
     * \param Spinor: vector class
     *
     * operator(x,b) set x so that M*x approximates b.
     */

    template <typename Spinor_t> class LinearSolverNoPrecon : public LinearSolver<Spinor_t> {
    public:
        /// Vector class
        using Spinor = Spinor_t;

        /**
         * Constructor
         *
         * \param M: operator
         * \param params: linear solver params
         * \param prec: preconditioner (should be nullptr)
         */

        LinearSolverNoPrecon(const LinearOperator<Spinor> &M, const LinearSolverParamsBase &params,
                             const LinearOperator<Spinor> *prec = nullptr)
            : LinearSolver<Spinor>(M, params) {
            SetPrec(prec);
        }

        /**
         * Set a preconditioner (optional)
         *
         * \param prec: new preconditioner (should be nullptr)
         *
         * \throw std::runtime_error if prec isn't nullptr
         */

        void SetPrec(const LinearOperator<Spinor> *prec = nullptr) const override {
            if (prec != nullptr) throw std::runtime_error("Preconditioning is not supported!");
        }
    };

    /**
     * Solver for implicit linear operators
     *
     * \param Spinor: vector class
     *
     * operator(x,b) set x so that M*x approximates b.
     */

    template <typename Spinor_t>
    class ImplicitLinearSolver : public LinearSolverNoPrecon<Spinor_t> {
    public:
        /// Vector class
        using Spinor = Spinor_t;

        const LatticeInfo &GetInfo() const override { return _M.GetInfo(); }

        /**
         * Return the support of the operator (SUBSET_EVEN, SUBSET_ODD, SUBSET_ALL)
         */

        const CBSubset &GetSubset() const override { return _M.GetSubset(); }

        virtual ~ImplicitLinearSolver() {}

    protected:
        /**
         * Constructor
         *
         * \param M: operator
         * \param params: linear solver params
         * \param prec: preconditioner
         */

        ImplicitLinearSolver(const LatticeInfo &info, const CBSubset subset = SUBSET_ALL,
                             const LinearSolverParamsBase &params = LinearSolverParamsBase())
            : LinearSolverNoPrecon<Spinor_t>(_M, params), _M(info, subset), _params(params) {}

    public:
        /// Linear operator
        const ImplicitLinearOperator<Spinor> _M;
        /// Linear solver params
        const LinearSolverParamsBase _params;
    };

    /**
     * Solve the linear system using the even-odd approach.
     */

    template <typename Solver> class UnprecLinearSolver : public Solver {
    public:
        /// Vector class
        using Spinor = typename Solver::Spinor;

        /**
         * Constructor
         *
         * \param M: operator
         * \param params: linear solver params
         * \param prec: preconditioner
         */

        UnprecLinearSolver(const EOLinearOperator<Spinor> &M, const LinearSolverParamsBase &params,
                           const LinearOperator<Spinor> *prec = nullptr)
            : Solver(M, params, prec), _M(M) {}

        virtual ~UnprecLinearSolver() {}

        /**
         * Compute the solution.
         *
         * \param[out] out: solution vector
         * \param in: input vector
         * \param resid_type: stopping criterion (RELATIVE or ABSOLUTE)
         * \param guess: Whether the initial is provided
         */

        virtual std::vector<LinearSolverResults>
        operator()(Spinor &out, const Spinor &in, ResiduumType resid_type = RELATIVE,
                   InitialGuess guess = InitialGuessNotGiven) const {

            std::vector<LinearSolverResults> ret_val;
            std::shared_ptr<Spinor> tmp_src = this->tmp(in);
            std::shared_ptr<Spinor> tmp_out = this->tmp(in);

            // Prepare the source: L^{-1} in
            _M.leftInvOp(*tmp_src, in);

            // Prepare the initial guess
            if (guess == InitialGuessGiven) {
                _M.rightOp(*tmp_out, out);
            } else {
                ZeroVec(*tmp_out, _M.GetSubset());
            }

            // Solve odd part
            ret_val = Solver::operator()(*tmp_out, *tmp_src, resid_type, guess);

            // Solve even part
            _M.M_ee_inv(*tmp_out, *tmp_src);

            // Reconstruct the result
            _M.rightInvOp(out, *tmp_out);

            return ret_val;
        }

        const CBSubset &GetSubset() const override { return SUBSET_ALL; }

    private:
        /// Linear operator
        const EOLinearOperator<Spinor> &_M;
    };

} // namespace MG

#endif /* INCLUDE_LATTICE_SOLVER_H_ */
