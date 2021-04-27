/*
 * coarse_op.h
 *
 *  Created on: Jan 21, 2016
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_COARSE_COARSE_OP_H_
#define INCLUDE_LATTICE_COARSE_COARSE_OP_H_

#include "MG_config.h"
#include "coarse_l1_blas.h"
#include "lattice/cmat_mult.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/thread_limits.h"
#include "lattice/constants.h"
#include "lattice/halo.h"
#include "utils/auxiliary.h"
#include <memory>

#include <omp.h>
namespace MG {

    /**
     * Operations with CoarseGauge and CoarseSpinor
     */

    class CoarseDiracOp : public AuxiliarySpinors<CoarseSpinor> {
    public:
        CoarseDiracOp(const LatticeInfo &l_info, IndexType n_smt = 1);

        ~CoarseDiracOp() {}

        // Applies M on target checkerboard
        // Full Op is:
        //
        // [ y_e ] = [ Mee  M_eo ] [ x_e ]
        // [ y_o ]   [ M_oe M_oo ] [ x_o ]
        //
        //   so  e.g.  y_cb = M_(cb,cb) x_(cb) + M_(cb,1-cb) x_(1-cb)

        /**
         * y[cb] = dagger(M)[cb,cb] * x[cb] + dagger(M)[cb,1-cb] * x[1-cb]
         *
         * \param y: (out) return vector
         * \param gauge: operator's (M) gauge field
         * \param x: input vector
         * \param target_cb: CB index of the output vector
         * \param dagger: apply M direct (LINOP_OP) or conjugate-transposed (LINOP_DAGGER)
         * \param tid: thread id
         */

        void unprecOp(CoarseSpinor &y, const CoarseGauge &gauge, const CoarseSpinor &x,
                      const IndexType target_cb, const IndexType dagger, const IndexType tid) const;

        /**
         * y[cb] = dagger(M)[cb,cb] * x[cb]
         *
         * \param y: (out) return vector
         * \param gauge: operator's (M) gauge field
         * \param x: input vector
         * \param target_cb: CB index of the output vector
         * \param dagger: apply M direct (LINOP_OP) or conjugate-transposed (LINOP_DAGGER)
         * \param tid: thread id
         */

        void M_diag(CoarseSpinor &y, const CoarseGauge &gauge, const CoarseSpinor &x,
                    const IndexType target_cb, const IndexType dagger, const IndexType tid) const;

        /**
         * y[cb] = inv(dagger(M)[cb,cb]) * x[cb]
         *
         * \param y: (out) return vector
         * \param gauge: operator's (M) gauge field
         * \param x: input vector
         * \param target_cb: CB index of the output vector
         * \param dagger: apply M direct (LINOP_OP) or conjugate-transposed (LINOP_DAGGER)
         * \param tid: thread id
         */

        void M_diagInv(CoarseSpinor &y, const CoarseGauge &gauge, const CoarseSpinor &x,
                       const IndexType target_cb, const IndexType dagger,
                       const IndexType tid) const;

        /**
         * y[cb] = alpha * dagger(M)[cb,1-cb] * x[1-cb] + y[cb]
         *
         * \param y: (in/out) return vector
         * \param alpha: scalar
         * \param gauge: operator's (M) gauge field
         * \param x: input vector
         * \param target_cb: CB index of the output vector
         * \param dagger: apply M direct (LINOP_OP) or conjugate-transposed (LINOP_DAGGER)
         * \param tid: thread id
         */

        void M_D_xpay(CoarseSpinor &y, const float alpha, const CoarseGauge &gauge,
                      const CoarseSpinor &x, const IndexType target_cb, const IndexType dagger,
                      const IndexType tid) const;

        /**
         * y[cb] = alpha * dagger(A * M)[cb,1-cb] * x[1-cb] + z[cb],
         * where A[cb,cb] = inv(M[cb,cb]), A[cb,1-cb] = 0.
         *
         * \param y: (in/out) return vector
         * \param alpha: scalar
         * \param gauge: operator's (M) gauge field
         * \param z: input vector
         * \param x: input vector
         * \param target_cb: CB index of the output vector
         * \param dagger: apply A*M direct (LINOP_OP) or conjugate-transposed (LINOP_DAGGER)
         * \param tid: thread id
         */

        void M_AD_xpayz(CoarseSpinor &y, const float alpha, const CoarseGauge &gauge,
                        const CoarseSpinor &z, const CoarseSpinor &x, const IndexType target_cb,
                        const IndexType dagger, const IndexType tid) const;

        /**
         * y[cb] = alpha * dagger(M)[cb,1-cb] * x[1-cb] + M[cb,cb]*z[cb].
         *
         * \param y: (in/out) return vector
         * \param alpha: scalar
         * \param gauge: operator's (M) gauge field
         * \param z: input vector
         * \param x: input vector
         * \param target_cb: CB index of the output vector
         * \param dagger: apply A*M direct (LINOP_OP) or conjugate-transposed (LINOP_DAGGER)
         * \param tid: thread id
         */

        void M_D_xpay_Mz(CoarseSpinor &y, const float alpha, const CoarseGauge &gauge,
                         const CoarseSpinor &z, const CoarseSpinor &x, const IndexType target_cb,
                         const IndexType dagger, const IndexType tid) const;

        /**
         * y[cb] = alpha * dagger(M * A)[cb,1-cb] * x[1-cb] + z[cb],
         * where A[cb,cb] = inv(M[cb,cb]), A[cb,1-cb] = 0.
         *
         * \param y: (in/out) return vector
         * \param alpha: scalar
         * \param gauge: operator's (M) gauge field
         * \param z: input vector
         * \param x: input vector
         * \param target_cb: CB index of the output vector
         * \param dagger: apply M*A direct (LINOP_OP) or conjugate-transposed (LINOP_DAGGER)
         * \param tid: thread id
         */

        void M_DA_xpayz(CoarseSpinor &y, const float alpha, const CoarseGauge &gauge,
                        const CoarseSpinor &z, const CoarseSpinor &x, const IndexType target_cb,
                        const IndexType dagger, const IndexType tid) const;

        /**
         * y[cb] = dagger(A * M)[cb,1-cb] * x[1-cb]
         * where A[cb,cb] = inv(M[cb,cb]), A[cb,1-cb] = 0.
         *
         * \param y: (in/out) return vector
         * \param gauge: operator's (M) gauge field
         * \param x: input vector
         * \param target_cb: CB index of the output vector
         * \param dagger: apply A*M direct (LINOP_OP) or conjugate-transposed (LINOP_DAGGER)
         * \param tid: thread id
         */

        void M_AD(CoarseSpinor &y, const CoarseGauge &gauge, const CoarseSpinor &x,
                  const IndexType target_cb, const IndexType dagger, const IndexType tid) const;

        /**
         * y[cb] = dagger(M * A)[cb,1-cb] * x[1-cb]
         * where A[cb,cb] = inv(M[cb,cb]), A[cb,1-cb] = 0.
         *
         * \param y: (in/out) return vector
         * \param gauge: operator's (M) gauge field
         * \param x: input vector
         * \param target_cb: CB index of the output vector
         * \param dagger: apply M*A direct (LINOP_OP) or conjugate-transposed (LINOP_DAGGER)
         * \param tid: thread id
         */

        void M_DA(CoarseSpinor &y, const CoarseGauge &gauge, const CoarseSpinor &x,
                  const IndexType target_cb, const IndexType dagger, const IndexType tid) const;

        /**
         * y[EVE] = x[EVE],
         * y[ODD] = (A * M)[ODD,EVE] * x[EVE] + x[ODD],
         * where A[cb,cb] = inv(M[cb,cb]), A[cb,1-cb] = 0.
         *
         * \param y: (out) return vector
         * \param gauge: operator's (M) gauge field
         * \param x: input vector
         *
         * In matrix form,
         *   y = [ 1           0 ] * x
         *       [ D_oe*A_ee   1 ]
         */

        void L_matrix(CoarseSpinor &y, const CoarseGauge &gauge, const CoarseSpinor &x) const {

            CopyVec(y, x, SUBSET_ALL);
#pragma omp parallel
            {
                int tid = omp_get_thread_num();
                M_DA_xpayz(y, 1.0, gauge, y, x, ODD, LINOP_OP, tid);
            }
        }

        /**
         * y[EVE] = x[EVE],
         * y[ODD] = -(A * M)[ODD,EVE] * x[EVE] + x[ODD],
         * where A[cb,cb] = inv(M[cb,cb]), A[cb,1-cb] = 0.
         *
         * \param y: (out) return vector
         * \param gauge: operator's (M) gauge field
         * \param x: input vector
         *
         * In matrix form,
         *   y = [  1           0 ] * x = [ 1           0 ]^{-1} * x
         *       [ -D_oe*A_ee   1 ]       [ D_oe*A_ee   1 ]
         */

        void L_inv_matrix(CoarseSpinor &y, const CoarseGauge &gauge, const CoarseSpinor &x) const {

            CopyVec(y, x, SUBSET_ALL);
#pragma omp parallel
            {
                int tid = omp_get_thread_num();
                M_DA_xpayz(y, -1.0, gauge, y, x, ODD, LINOP_OP, tid);
            }
        }

        /**
         * y[EVE] = (A * M)[EVE,ODD] * x[ODD] + x[EVE],
         * y[ODD] = x[ODD],
         * where A[cb,cb] = inv(M[cb,cb]), A[cb,1-cb] = 0.
         *
         * \param y: (out) return vector
         * \param gauge: operator's (M) gauge field
         * \param x: input vector
         *
         * In matrix form,
         *   y = [ 1   A_ee*D_oe ] * x
         *       [ 0           1 ]
         */

        void R_matrix(CoarseSpinor &y, const CoarseGauge &gauge, const CoarseSpinor &x) const {

            CopyVec(y, x, SUBSET_ALL);
#pragma omp parallel
            {
                int tid = omp_get_thread_num();
                M_AD_xpayz(y, 1, gauge, y, x, EVEN, LINOP_OP, tid);
            }
        }

        /**
         * y[EVE] = -(A * M)[EVE,ODD] * x[ODD] + x[EVE],
         * y[ODD] = x[ODD],
         * where A[cb,cb] = inv(M[cb,cb]), A[cb,1-cb] = 0.
         *
         * \param y: (out) return vector
         * \param gauge: operator's (M) gauge field
         * \param x: input vector
         *
         * In matrix form,
         *   y = [ 1  -A_ee*D_oe ] * x = [ 1   A_ee*D_oe ]^{-1} * x
         *       [ 0           1 ]       [ 0           1 ]
         */

        void R_inv_matrix(CoarseSpinor &y, const CoarseGauge &gauge, const CoarseSpinor &x) const {

            CopyVec(y, x, SUBSET_ALL);
#pragma omp parallel
            {
                int tid = omp_get_thread_num();
                M_AD_xpayz(y, -1, gauge, y, x, EVEN, LINOP_OP, tid);
            }
        }

        /**
         * y[cb] = dagger(M[cb,cb] - M[cb,1-cb] * M[1-cb,1-cb]^{-1} * M[1-cb,cb]) * x[cb]
         *
         * \param y: (in/out) return vector
         * \param gauge: operator's (M) gauge field
         * \param x: input vector
         * \param target_cb: CB index of the output vector
         * \param dagger: apply direct (LINOP_OP) or conjugate-transposed (LINOP_DAGGER)
         */

        void EOPrecOp(CoarseSpinor &spinor_out, const CoarseGauge &gauge_in,
                      const CoarseSpinor &spinor_in, const int target_cb,
                      const IndexType dagger) const {
            std::shared_ptr<CoarseSpinor> t = tmp(spinor_in);
#pragma omp parallel
            {
                int tid = omp_get_thread_num();

                M_AD(*t, gauge_in, spinor_in, 1 - target_cb, dagger, tid);
#pragma omp barrier

                M_D_xpay_Mz(spinor_out, -1.0, gauge_in, spinor_in, *t, target_cb, dagger, tid);
            }
        }

        void CloverApply(CoarseSpinor &spinor_out, const CoarseGauge &gauge_clov_in,
                         const CoarseSpinor &spinor_in, const IndexType target_cb,
                         const IndexType dagger, const IndexType tid) const {
            M_diag(spinor_out, gauge_clov_in, spinor_in, target_cb, dagger, tid);
        }

        void DslashDir(CoarseSpinor &spinor_out, const CoarseGauge &gauge_in,
                       const CoarseSpinor &spinor_in, const IndexType target_cb,
                       const IndexType dir, const IndexType tid) const;

        inline IndexType GetNumColorSpin() const { return _n_colorspin; }

        inline IndexType GetNumColor() const { return _n_color; }

        inline IndexType GetNumSpin() const { return _n_spin; }

        inline SpinorHaloCB &GetSpinorHalo() { return _halo; }

        inline const SpinorHaloCB &GetSpinorHalo() const { return _halo; }

        static void write(const CoarseGauge &gauge_clov_in, std::string &filename);

    private:
        const LatticeInfo _lattice_info;
        const IndexType _n_color;
        const IndexType _n_spin;
        const IndexType _n_colorspin;
        const IndexType _n_smt;
        const IndexType _n_vrows;

        int _n_threads;
        ThreadLimits *_thread_limits;

        // These are handy to have around
        // Scoped to the class. They can be computed on instantiation
        // as they are essentially in the lattice info.
        const IndexType _n_xh;
        const IndexType _n_x;
        const IndexType _n_y;
        const IndexType _n_z;
        const IndexType _n_t;

        mutable SpinorHaloCB _halo;
    };
}

#endif /* INCLUDE_LATTICE_COARSE_COARSE_OP_H_ */
