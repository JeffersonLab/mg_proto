/*
 * wilson_clover_linear_operator.h
 *
 *  Created on: Jan 9, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_FINE_QDPXX_WILSON_CLOVER_LINEAR_OPERATOR_H_
#define INCLUDE_LATTICE_FINE_QDPXX_WILSON_CLOVER_LINEAR_OPERATOR_H_

/*!
 * Wrap a QDP++ linear operator, so as to provide the interface
 * with new types
 */
#include "lattice/coarse/aggregate_block_coarse.h"
#include "lattice/coarse/block.h"
#include "lattice/coarse/coarse_l1_blas.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/fine_qdpxx/aggregate_block_qdpxx.h"
#include "lattice/fine_qdpxx/clover_fermact_params_w.h"
#include "lattice/fine_qdpxx/clover_term_qdp_w.h"
#include "lattice/fine_qdpxx/dslashm_w.h"
#include "lattice/fine_qdpxx/qdpxx_helpers.h"
#include "lattice/linear_operator.h"

using namespace QDP;

namespace MG {

    class QDPWilsonCloverLinearOperator : public LinearOperator<LatticeFermion> {
    public:
        using Gauge = multi1d<LatticeColorMatrix>;

        QDPWilsonCloverLinearOperator(double m_q, double c_sw, int t_bc, const Gauge &gauge_in)
            : _t_bc(t_bc) {
            _u.resize(Nd);
            for (int mu = 0; mu < Nd; ++mu) _u[mu] = gauge_in[mu];

            // Apply boundary
            _u[Nd - 1] *=
                where(Layout::latticeCoordinate(Nd - 1) == (Layout::lattSize()[Nd - 1] - 1),
                      Integer(_t_bc), Integer(1));
            _params.Mass = Real(m_q);
            _params.clovCoeffR = Real(c_sw);
            _params.clovCoeffT = Real(c_sw);
            _clov.create(_u, _params); // Make the clover term
            _invclov.create(_u, _params, _clov);

            _invclov.choles(EVEN);

            IndexArray latdims = {
                {QDP::Layout::subgridLattSize()[0], QDP::Layout::subgridLattSize()[1],
                 QDP::Layout::subgridLattSize()[2], QDP::Layout::subgridLattSize()[3]}};

            _node = new NodeInfo();
            _info = new LatticeInfo(latdims, 4, 3, *_node);
        }

        QDPWilsonCloverLinearOperator(double m_q, double u0, double xi0, double nu, double c_sw_r,
                                      double c_sw_t, int t_bc, const Gauge &gauge_in)
            : _t_bc(t_bc) {
            _u.resize(Nd);

            // Copy in the gauge field
            for (int mu = 0; mu < Nd; ++mu) { _u[mu] = gauge_in[mu]; }

            // Apply boundary
            _u[Nd - 1] *=
                where(Layout::latticeCoordinate(Nd - 1) == (Layout::lattSize()[Nd - 1] - 1),
                      Integer(_t_bc), Integer(1));

            // Set up and create the clover term
            // Use the unscaled links, because the scale factors will play
            // Into the clover calc.

            _params.Mass = Real(m_q);
            _params.clovCoeffR = Real(c_sw_r);
            _params.clovCoeffT = Real(c_sw_t);
            _params.u0 = Real(u0);
            _params.anisoParam.anisoP = true;
            _params.anisoParam.t_dir = 3;
            _params.anisoParam.xi_0 = Real(xi0);
            _params.anisoParam.nu = Real(nu);

            _clov.create(_u, _params); // Make the clover term
            _invclov.create(_u, _params, _clov);
            _invclov.choles(EVEN);

            // Now scale the links for use in the dslash
            // By the anisotropy.
            Real aniso_scale_fac = Real(nu / xi0);
            for (int mu = 0; mu < Nd; ++mu) {
                if (mu != 3) { _u[mu] *= aniso_scale_fac; }
            }
            IndexArray latdims = {
                {QDP::Layout::subgridLattSize()[0], QDP::Layout::subgridLattSize()[1],
                 QDP::Layout::subgridLattSize()[2], QDP::Layout::subgridLattSize()[3]}};

            _node = new NodeInfo();
            _info = new LatticeInfo(latdims, 4, 3, *_node);
        }

        ~QDPWilsonCloverLinearOperator() {
            delete _node;
            delete _info;
        }

        void operator()(Spinor &out, const Spinor &in, IndexType type = LINOP_OP) const override {

            // Cut this out of Chroma, modified for explicit checkerrboarding

            LatticeFermion tmp = zero;
            Real mhalf = -0.5;
            const int isign = (type == LINOP_OP) ? 1 : -1;

            for (int cb = 0; cb < n_checkerboard; ++cb) {
                _clov.apply(out, in, isign, cb);    // CB -> CB
                MG::dslash(tmp, _u, in, isign, cb); // (1-CB) -> C
                out[rb[cb]] += mhalf * tmp;
            }
        }

        void M_ee(Spinor &out, const Spinor &in, IndexType type = LINOP_OP) const {
            M_diag(out, in, type, EVEN);
        }

        void M_oo(Spinor &out, const Spinor &in, IndexType type = LINOP_OP) const {
            M_diag(out, in, type, ODD);
        }

        void M_eo(Spinor &out, const Spinor &in, IndexType type = LINOP_OP) const {
            M_offdiag(out, in, type, EVEN);
        }

        void M_oe(Spinor &out, const Spinor &in, IndexType type = LINOP_OP) const {
            M_offdiag(out, in, type, ODD);
        }

        void M_ee_inv(Spinor &out, const Spinor &in, IndexType type = LINOP_OP) const {
            const int isign = (type == LINOP_OP) ? 1 : -1;
            _invclov.apply(out, in, isign, EVEN);
        }

        const CBSubset &GetSubset() const override { return SUBSET_ALL; }

        const LatticeInfo &GetInfo(void) const override { return *_info; }

        void generateCoarse(const std::vector<Block> &blocklist,
                            const multi1d<LatticeFermion> &in_vecs, CoarseGauge &u_coarse) const {
            const LatticeInfo &info = u_coarse.GetInfo();
            int num_colorspin = info.GetNumColorSpins();

            // Generate the triple products directly into the u_coarse
            ZeroGauge(u_coarse);
            for (int mu = 0; mu < 8; ++mu) {
                MasterLog(INFO,
                          "QDPWilsonCloverLinearOperator: Dslash Triple Product in direction: %d ",
                          mu);
                dslashTripleProductDirQDPXX(blocklist, mu, _u, in_vecs, u_coarse);
            }

            for (int cb = 0; cb < n_checkerboard; ++cb) {
                for (int cbsite = 0; cbsite < u_coarse.GetInfo().GetNumCBSites(); ++cbsite) {
                    // Offdiag multiply
                    for (int mu = 0; mu < 8; ++mu) {
                        float *link = u_coarse.GetSiteDirDataPtr(cb, cbsite, mu);
                        for (int j = 0; j < n_complex * num_colorspin * num_colorspin; ++j) {
                            link[j] *= -0.5;
                        }
                    }
                    float *diag_link = u_coarse.GetSiteDiagDataPtr(cb, cbsite);
                    for (int j = 0; j < n_complex * num_colorspin * num_colorspin; ++j) {
                        diag_link[j] *= -0.5;
                    }
                }
            }

            MasterLog(INFO, "QDPWilsonCloverLinearOperator: Clover Triple Product");
            clovTripleProductQDPXX(blocklist, _clov, in_vecs, u_coarse);

            MasterLog(INFO, "QDPWilsonCloverLinearOperator: Inverting Diagonal (A) Links");
            invertCloverDiag(u_coarse);

            MasterLog(INFO, "QDPWilsonCloverLinearOperator: Computing A^{-1} D Links");
            multInvClovOffDiagLeft(u_coarse);

            MasterLog(INFO, "QDPWilsonCloverLinearOperator: Computing D A^{-1} Links");
            multInvClovOffDiagRight(u_coarse);
        }

        // Caller must zero appropriately
        void generateCoarseGauge(const std::vector<Block> &blocklist,
                                 const multi1d<LatticeFermion> &in_vecs,
                                 CoarseGauge &u_coarse) const {
            for (int mu = 0; mu < 8; ++mu) {
                QDPIO::cout << "QDPWilsonCloverLinearOperator: Dslash Triple Product in direction: "
                            << mu << std::endl;
                dslashTripleProductDirQDPXX(blocklist, mu, _u, in_vecs, u_coarse);
            }
        }

        // Caller must zero appropriately
        void generateCoarseClover(const std::vector<Block> &blocklist,
                                  const multi1d<LatticeFermion> &in_vecs,
                                  CoarseGauge &coarse_clov) const {
            QDPIO::cout << "QDPWilsonCloverLinearOperator: Clover Triple Product" << std::endl;
            clovTripleProductQDPXX(blocklist, _clov, in_vecs, coarse_clov);
        }

        const MG::QDPCloverTerm &getClov() const { return _clov; }
        const MG::QDPCloverTerm &getInvClov() const { return _invclov; }

    private:
        const int _t_bc;
        Gauge _u;
        MG::CloverFermActParams _params;
        MG::QDPCloverTerm _clov;
        MG::QDPCloverTerm _invclov;

        NodeInfo *_node;
        LatticeInfo *_info;

        inline void M_diag(Spinor &out, const Spinor &in, IndexType type, int cb) const {
            const int isign = (type == LINOP_OP) ? +1 : -1;
            _clov.apply(out, in, isign, cb);
        }

        inline void M_offdiag(Spinor &out, const Spinor &in, IndexType type, int target_cb) const {
            const int isign = (type == LINOP_OP) ? +1 : -1;
            MG::dslash(out, _u, in, isign, target_cb);
            out[rb[target_cb]] *= Real(-0.5);
        }
    };
}

#endif /* TEST_QDPXX_WILSON_CLOVER_LINEAR_OPERATOR_H_ */
