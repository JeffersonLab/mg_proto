/*
 * coarse_wilson_clover_linear_operator.h
 *
 *  Created on: Jan 12, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_COARSE_COARSE_EO_WILSON_CLOVER_LINEAR_OPERATOR_H_
#define INCLUDE_LATTICE_COARSE_COARSE_EO_WILSON_CLOVER_LINEAR_OPERATOR_H_

#include "lattice/coarse/aggregate_block_coarse.h"
#include "lattice/coarse/coarse_l1_blas.h"
#include "lattice/coarse/coarse_op.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/linear_operator.h"
#include "utils/auxiliary.h"
#include "utils/print_utils.h"
#include <memory>
#include <omp.h>
#include <vector>

namespace MG {

    class CoarseEOWilsonCloverLinearOperator : public EOLinearOperator<CoarseSpinor> {
    public:
        // Hardwire n_smt=1 for now.
        CoarseEOWilsonCloverLinearOperator(const std::shared_ptr<CoarseGauge> &gauge_in)
            : _u(gauge_in), _the_op(gauge_in->GetInfo(), 1) {
            subrogateTo(&_the_op);
            MasterLog(INFO, "Creating Coarse CoarseEOWilsonCloverLinearOperator LinOp");
        }

        ~CoarseEOWilsonCloverLinearOperator() {}

        const CBSubset &GetSubset() const override { return SUBSET_ODD; }

        /**
         * y_o = dagger(M_oo - M_oe * M_ee^{-1} * M_eo) * x_o
         *
         * \param out: (in/out) return vector
         * \param in: input vector
         * \param type: apply direct (LINOP_OP) or conjugate-transposed (LINOP_DAGGER)
         */

        void operator()(Spinor &out, const Spinor &in, IndexType type = LINOP_OP) const override {
            _the_op.EOPrecOp(out, *_u, in, ODD, type);
        }

        /**
         * y = dagger(M) * x
         *
         * \param out: (in/out) return vector
         * \param in: input vector
         * \param type: apply direct (LINOP_OP) or conjugate-transposed (LINOP_DAGGER)
         */

        void unprecOp(Spinor &out, const Spinor &in, IndexType type = LINOP_OP) const override {
#pragma omp parallel
            {
                int tid = omp_get_thread_num();
                _the_op.unprecOp(out, (*_u), in, EVEN, type, tid);
                _the_op.unprecOp(out, (*_u), in, ODD, type, tid);
            }
        }

        /**
         * out = L * in
         *
         * \param out: (out) return vector
         * \param in: input vector
         *
         * In matrix form,
         *   out = [ 1  M_oe*M_ee^{-1} ] * in
         *         [ 0           1     ]
         */

        void leftOp(Spinor &out, const Spinor &in) const override {
            _the_op.L_matrix(out, *_u, in);
        }

        /**
         * out = L^{-1} * in
         *
         * \param out: (out) return vector
         * \param in: input vector
         *
         * In matrix form,
         *   out = [ 1  -M_oe*M_ee^{-1} ] * in
         *         [ 0            1     ]
         */

        void leftInvOp(Spinor &out, const Spinor &in) const override {
            _the_op.L_inv_matrix(out, *_u, in);
        }

        void M_diag(Spinor &out, const Spinor &in, int cb) const override {
#pragma omp parallel
            {
                int tid = omp_get_thread_num();

                _the_op.M_diag(out, (*_u), in, cb, LINOP_OP, tid);
            }
        }

        /**
         * out = R * in
         *
         * \param out: (out) return vector
         * \param in: input vector
         *
         * In matrix form,
         *   out = [ 1               0 ] * in
         *         [ M_ee^{-1}*M_eo  1 ]
         */

        void rightOp(Spinor &out, const Spinor &in) const override {
            _the_op.R_matrix(out, (*_u), in);
        }

        /**
         * out = R^{-1} * in
         *
         * \param out: (out) return vector
         * \param in: input vector
         *
         * In matrix form,
         *   out = [  1               0 ] * in
         *         [ -M_ee^{-1}*M_eo  1 ]
         */

        void rightInvOp(Spinor &out, const Spinor &in) const override {
            _the_op.R_inv_matrix(out, *_u, in);
        }

        /**
         * out_e = M_ee^{-1} * in_e
         *
         * \param out: (out) return vector
         * \param in: input vector
         */

        void M_ee_inv(Spinor &out, const Spinor &in, IndexType type = LINOP_OP) const override {
            (void)type;
#pragma omp parallel
            {
                int tid = omp_get_thread_num();
                _the_op.M_diagInv(out, *_u, in, EVEN, LINOP_OP, tid);
            }
        }

        /**
         * out_o = M_oo^{-1} * in_o
         *
         * \param out: (out) return vector
         * \param in: input vector
         */

        void M_oo_inv(Spinor &out, const Spinor &in, IndexType type = LINOP_OP) const override {
            (void)type;
#pragma omp parallel
            {
                int tid = omp_get_thread_num();
                _the_op.M_diagInv(out, *_u, in, ODD, LINOP_OP, tid);
            }
        }

        void generateCoarse(const std::vector<Block> &blocklist,
                            const std::vector<std::shared_ptr<CoarseSpinor>> in_vecs,
                            CoarseGauge &u_coarse) const {
            // Generate the triple products directly into the u_coarse
            ZeroGauge(u_coarse);
            for (int mu = 0; mu < 8; ++mu) {
                MasterLog(INFO,
                          "CoarseEOCloverLinearOperator: Dslash Triple Product in direction:%d",
                          mu);
                dslashTripleProductDir(_the_op, blocklist, mu, (*_u), in_vecs, u_coarse);
            }

            MasterLog(INFO, "CoarseEOCloverLinearOperator: Clover Triple Product");
            clovTripleProduct(_the_op, blocklist, (*_u), in_vecs, u_coarse);

            MasterLog(INFO, "CoarseEOCloverLinearOperator: Inverting Diagonal (A) Links");
            invertCloverDiag(u_coarse);

            MasterLog(INFO, "CoarseEOCloverLinearOperator: Computing A^{-1} D Links");
            multInvClovOffDiagLeft(u_coarse);

            MasterLog(INFO, "CoarseEOCloverLinearOperator: Computing D A^{-1} Links");
            multInvClovOffDiagRight(u_coarse);
        }

        const LatticeInfo &GetInfo(void) const override { return _u->GetInfo(); }

    private:
        const std::shared_ptr<CoarseGauge> _u;
        const CoarseDiracOp _the_op;
    };

} // namespace MG

#endif /* TEST_QDPXX_COARSE_WILSON_CLOVER_LINEAR_OPERATOR_H_ */
