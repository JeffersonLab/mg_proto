/*
 * linear_operator.h
 *
 *  Created on: Jan 9, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_LINEAR_OPERATOR_H_
#define INCLUDE_LATTICE_LINEAR_OPERATOR_H_

#include "lattice/coarse/subset.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "utils/auxiliary.h"
#include "utils/print_utils.h"
#include <stdexcept>

namespace MG {

    /**
     * Abstract linear operator class
     *
     * \param Spinor_t: vector class
     *
     * LinearOpeartor<V> represents an operator such that y0+y1 approximates y2 after calling
     * operator()(y0,x0), operator()(y1,x1), and operator(y2,x0+x1) for any vectors V x0, x1,
     * such that ||y0+y1 - y2|| <= error * ||y2||.
     */

    template <typename Spinor_t> class LinearOperator : public AuxiliarySpinors<Spinor_t> {
    public:
        using Spinor = Spinor_t;

        /**
         * Apply the operator
         *
         * \param[out] out: the result of applying the operator on "in"
         * \param in: the input vector
         * \param type: whether to apply the operator directly (LINOP_OP) or conjugate transposed
         *             (LINOP_DAGGER)
         */

        virtual void operator()(Spinor &out, const Spinor &in, IndexType type = LINOP_OP) const = 0;

        virtual ~LinearOperator() {}

        /**
         * Return the multigrid level; 0 is top level (fine operator)
         */

        inline int GetLevel(void) const { return GetInfo().GetLevel(); }

        /**
         * Return the lattice information
         */

        virtual const LatticeInfo &GetInfo() const = 0;

        /**
         * Return the support of the operator (SUBSET_EVEN, SUBSET_ODD, SUBSET_ALL)
         */

        virtual const CBSubset &GetSubset() const = 0;
    };

    /**
     * Abstract Even-Odd linear operator class
     *
     * \param Spinor_t: vector class
     *
     * EOLinearOpeartor<V>::operator() usually represents the Schur complement of the original
     * operator (unprecOp).
     *
     * Given the original operator split into the even and odd sites, it is factorized as follows
     *
     *   M = L * D * R,
     *
     * that is,
     *
     *   [M_oo  M_oe] = [1  M_oe*M_ee^{-1}] * [M_oo-M_oe*M_ee^{-1}*M_eo    0 ] * [       1        0]
     *   [M_eo  M_ee]   [0        1       ]   [           0              M_ee]   [M_ee^{-1}*M_eo  1]
     *
     * M_oo and M_ee are (block) diagonal on operators with only near neighbor connections, making
     * them efficiently invertible. Instead of solving the linear system
     *
     *   M * x = b,
     *
     * it is solved
     *
     *   L^{-1}*M*R^{-1} * R*x = L^{-1}*b -> D * y = c.
     *
     * Most of the effort is on solving D_oo * y_o = c_o. Note that despite using left
     * preconditioning, the residual norm of a solution for D_oo matches the residual norm of
     * the solution R^{-1}*y for M if D_ee * y_e = c_e is solved exactly. (Given an approximate
     * solution for D, the residual vector for M is L*(D*y-c), and L*[z; 0] = [z; 0].)
     */

    template <typename Spinor_t> class EOLinearOperator : public LinearOperator<Spinor_t> {
    public:
        using Spinor = Spinor_t;

        /**
         * Apply the original operator
         *
         * \param[out] out: the result of applying the operator on "in"
         * \param in: the input vector
         * \param type: whether to apply the operator directly (LINOP_OP) or conjugate transposed
         * (LINOP_DAGGER)
         */

        virtual void unprecOp(Spinor &out, const Spinor &in, IndexType type = LINOP_OP) const = 0;

        /**
         * Apply the left preconditioning inverted, L
         *
         * \param[out] out: the result of applying the operator on "in"
         * \param in: the input vector
         */

        virtual void leftOp(Spinor &out, const Spinor &in) const = 0;

        /**
         * Apply the left preconditioning, L^{-1}
         *
         * \param[out] out: the result of applying the operator on "in"
         * \param in: the input vector
         */

        virtual void leftInvOp(Spinor &out, const Spinor &in) const = 0;

        /**
         * Apply the right preconditioning inverted, R
         *
         * \param[out] out: the result of applying the operator on "in"
         * \param in: the input vector
         */

        virtual void rightOp(Spinor &out, const Spinor &in) const = 0;

        /**
         * Apply the right preconditioning, R^{-1}
         *
         * \param[out] out: the result of applying the operator on "in"
         * \param in: the input vector
         */

        virtual void rightInvOp(Spinor &out, const Spinor &in) const = 0;

        /**
         * Apply M_oo^{-1}
         *
         * \param[out] out: the result of applying the operator on "in"
         * \param in: the input vector
         */

        virtual void M_oo_inv(Spinor &out, const Spinor &in, IndexType type = LINOP_OP) const = 0;


        /**
         * Apply M_ee^{-1}
         *
         * \param[out] out: the result of applying the operator on "in"
         * \param in: the input vector
         */

        virtual void M_ee_inv(Spinor &out, const Spinor &in, IndexType type = LINOP_OP) const = 0;

        /**
         * Apply either M_ee or M_oo
         *
         * \param[out] out: the result of applying the operator on "in"
         * \param in: the input vector
         * \param cb: EVEN for applying M_ee and ODD for applying M_oo
         */

        virtual void M_diag(Spinor &out, const Spinor &in, int cb) const = 0;
    };

     /**
     * Implicit linear operator class
     *
     * \param Spinor_t: vector class
     *
     * LinearOperator that the dimensions are only exposed.
     */

    template <typename Spinor_t> class ImplicitLinearOperator : public LinearOperator<Spinor_t> {
    public:
        using Spinor = Spinor_t;

        /**
         * Constructor
         *
         * \param info: lattice info
         * \param subset: subset (default SUBSET_ALL)
         */

        ImplicitLinearOperator(const LatticeInfo &info, const CBSubset subset = SUBSET_ALL)
            : _info(info), _subset(subset) {}

        /**
         * Apply the operator (not available)
         *
         * \param[out] out: the result of applying the operator on "in"
         * \param in: the input vector
         * \param type: whether to apply the operator directly (LINOP_OP) or conjugate transposed
         *             (LINOP_DAGGER)
         *
         * \throw std::runtime_error always
         */

        void operator()(Spinor &out, const Spinor &in, IndexType type = LINOP_OP) const override {
            throw std::runtime_error("Implicit operator! operator() is not available!");
        }

        virtual ~ImplicitLinearOperator() {}

        /**
         * Return the lattice information
         */

        const LatticeInfo &GetInfo() const override { return _info; }

        /**
         * Return the support of the operator (SUBSET_EVEN, SUBSET_ODD, SUBSET_ALL)
         */

        const CBSubset &GetSubset() const override { return _subset; }

    private:
        const LatticeInfo _info;
        const CBSubset _subset;
    };

    /**
     * Linear operator for M_oo^{-1}
     *
     * \param Spinor_t: vector class
     *
     * EOLinearOpeartor<V>::operator() applies M_oo^{-1}
     */

    template <typename Spinor_t> class M_oo_inv : public ImplicitLinearOperator<Spinor_t> {
    public:
        using Spinor = Spinor_t;
        M_oo_inv(const EOLinearOperator<Spinor_t> &op)
            : ImplicitLinearOperator<Spinor_t>(op.GetInfo(), SUBSET_ODD), _op(op) {}

        void operator()(Spinor &out, const Spinor &in, IndexType type = LINOP_OP) const {
            if (type != LINOP_OP) throw std::runtime_error("type not supported");
            _op.M_oo_inv(out, in);
        }

    private:
        const EOLinearOperator<Spinor_t> &_op;
};


} // namespace MG

#endif /* INCLUDE_LATTICE_LINEAR_OPERATOR_H_ */
