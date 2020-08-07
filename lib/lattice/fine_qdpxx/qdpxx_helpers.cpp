/*
 * qdpxx_helpers.cpp
 *
 *  Created on: Mar 17, 2016
 *      Author: bjoo
 */

#include "lattice/fine_qdpxx/qdpxx_helpers.h"
#include "lattice/lattice_info.h"
#include <qdp.h>
using namespace QDP;

namespace MG {

    void QDPSpinorToCoarseSpinor(const LatticeFermion &qdpxx_in, CoarseSpinor &coarse_out) {
        IndexType num_colorspin = coarse_out.GetNumColorSpin();
        IndexType num_cb_sites = coarse_out.GetInfo().GetNumCBSites();
        assert(coarse_out.GetNCol() == 1);

        // Assert site tables are equal
        if (num_cb_sites != rb[0].numSiteTable()) {
            QDPIO::cerr << "Num cb sites in coarse = " << num_cb_sites
                        << " But QDP++ checkerboard size is" << rb[0].numSiteTable() << std::endl;
            QDP_abort(1);
        }

        // Assert num colorspin is 12 otherwise these are not compatible
        if (num_colorspin != 12) {
            QDPIO::cerr << "Num colorspin ( " << num_colorspin << " != 12 " << std::endl;
            QDP_abort(1);
        }

#pragma omp parallel for collapse(2)
        for (int cb = 0; cb < 2; ++cb) {
            for (int cbsite = 0; cbsite < num_cb_sites; ++cbsite) {
                float *spinor_data = coarse_out.GetSiteDataPtr(0, cb, cbsite);

                for (int colorspin = 0; colorspin < num_colorspin; ++colorspin) {
                    int spin = colorspin / 3;
                    int color = colorspin % 3;

                    // real_part
                    int qdpxx_site = (rb[cb].siteTable())[cbsite];
                    spinor_data[RE + n_complex * colorspin] =
                        qdpxx_in.elem(qdpxx_site).elem(spin).elem(color).real();
                    spinor_data[IM + n_complex * colorspin] =
                        qdpxx_in.elem(qdpxx_site).elem(spin).elem(color).imag();
                }
            }
        }
    }

    void CoarseSpinorToQDPSpinor(const CoarseSpinor &coarse_in, LatticeFermion &qdpxx_out) {
        IndexType num_colorspin = coarse_in.GetNumColorSpin();
        IndexType num_cb_sites = coarse_in.GetInfo().GetNumCBSites();
        assert(coarse_in.GetNCol() == 1);

        // Assert site tables are equal
        if (num_cb_sites != rb[0].numSiteTable()) {
            QDPIO::cerr << "Num cb sites in coarse = " << num_cb_sites
                        << " But QDP++ checkerboard size is" << rb[0].numSiteTable() << std::endl;
            QDP_abort(1);
        }

        // Assert num colorspin is 12 otherwise these are not compatible
        if (num_colorspin != 12) {
            QDPIO::cerr << "Num colorspin ( " << num_colorspin << " != 12 " << std::endl;
            QDP_abort(1);
        }

#pragma omp parallel for collapse(2)
        for (int cb = 0; cb < 2; ++cb) {
            for (int cbsite = 0; cbsite < num_cb_sites; ++cbsite) {

                const float *spinor_data = coarse_in.GetSiteDataPtr(0, cb, cbsite);

                for (int colorspin = 0; colorspin < num_colorspin; ++colorspin) {
                    int spin = colorspin / 3;
                    int color = colorspin % 3;

                    // real_part
                    int qdpxx_site = (rb[cb].siteTable())[cbsite];
                    qdpxx_out.elem(qdpxx_site).elem(spin).elem(color).real() =
                        spinor_data[RE + n_complex * colorspin];
                    qdpxx_out.elem(qdpxx_site).elem(spin).elem(color).imag() =
                        spinor_data[IM + n_complex * colorspin];
                }
            }
        }
    }

    void QDPGaugeLinksToCoarseGaugeLinks(const multi1d<LatticeColorMatrix> &qdp_u_in,
                                         CoarseGauge &gauge_out) {
        const LatticeInfo &info = gauge_out.GetInfo();
        int num_cb_sites = info.GetNumCBSites();

        assert(info.GetNumColors() == 3);
        assert(info.GetNumSpins() == 4);
        assert(num_cb_sites == rb[0].numSiteTable());

        for (int dir = 0; dir < n_dim; ++dir) {
            LatticeColorMatrix u_back = adj(shift(qdp_u_in[dir], BACKWARD, dir));
            const LatticeColorMatrix &u_forw = qdp_u_in[dir];

            for (int cb = 0; cb < n_checkerboard; ++cb) {
#pragma omp parallel for
                for (int cbsite = 0; cbsite < num_cb_sites; ++cbsite) {

                    float *site_data_forw = gauge_out.GetSiteDirDataPtr(cb, cbsite, 2 * dir);
                    float *site_data_back = gauge_out.GetSiteDirDataPtr(cb, cbsite, 2 * dir + 1);
                    for (int row = 0; row < 3; ++row) {
                        for (int col = 0; col < 3; ++col) {
                            site_data_forw[RE + n_complex * (col + 3 * row)] =
                                u_forw.elem(rb[cb].siteTable()[cbsite])
                                    .elem()
                                    .elem(row, col)
                                    .real();
                            site_data_forw[IM + n_complex * (col + 3 * row)] =
                                u_forw.elem(rb[cb].siteTable()[cbsite])
                                    .elem()
                                    .elem(row, col)
                                    .imag();
                        } // col
                    }     // row
                    for (int row = 0; row < 3; ++row) {
                        for (int col = 0; col < 3; ++col) {
                            site_data_back[RE + n_complex * (col + 3 * row)] =
                                u_back.elem(rb[cb].siteTable()[cbsite])
                                    .elem()
                                    .elem(row, col)
                                    .real();
                            site_data_back[IM + n_complex * (col + 3 * row)] =
                                u_back.elem(rb[cb].siteTable()[cbsite])
                                    .elem()
                                    .elem(row, col)
                                    .imag();

                        } // col
                    }     // row
                }         // cbsite
            }             // cb
        }                 // dir
    }

    void CoarseGaugeLinksToQDPGaugeLinks(const CoarseGauge &coarse_in,
                                         multi1d<LatticeColorMatrix> &qdp_u_out) {
        const LatticeInfo &info = coarse_in.GetInfo();
        int num_cb_sites = info.GetNumCBSites();

        assert(info.GetNumColors() == 3);
        assert(info.GetNumSpins() == 4);
        assert(num_cb_sites == rb[0].numSiteTable());

        for (int dir = 0; dir < n_dim; ++dir) {
            //LatticeColorMatrix& u = qdp_u_out[dir];

            for (int cb = 0; cb < n_checkerboard; ++cb) {
#pragma omp parallel for
                for (int cbsite = 0; cbsite < num_cb_sites; ++cbsite) {

                    const float *site_data_forw = coarse_in.GetSiteDirDataPtr(cb, cbsite, 2 * dir);
                    for (int row = 0; row < 3; ++row) {
                        for (int col = 0; col < 3; ++col) {
                            qdp_u_out[dir]
                                .elem(rb[cb].siteTable()[cbsite])
                                .elem()
                                .elem(row, col)
                                .real() = site_data_forw[RE + n_complex * (col + 3 * row)];
                            qdp_u_out[dir]
                                .elem(rb[cb].siteTable()[cbsite])
                                .elem()
                                .elem(row, col)
                                .imag() = site_data_forw[IM + n_complex * (col + 3 * row)];
                        } // col
                    }     // row
                }         // cbsite
            }             // cb
        }                 // dir
    }

    void QDPPropToCoarseGaugeLink(const LatticePropagator &qdpxx_in, CoarseGauge &coarse_out,
                                  int dir) {
        IndexType num_colorspin = coarse_out.GetNumColorSpin();
        IndexType num_cb_sites = coarse_out.GetInfo().GetNumCBSites();

        // Assert site tables are equal
        if (num_cb_sites != rb[0].numSiteTable()) {
            QDPIO::cerr << "Num cb sites in coarse = " << num_cb_sites
                        << " But QDP++ checkerboard size is" << rb[0].numSiteTable() << std::endl;
            QDP_abort(1);
        }

        // Assert num colorspin is 12 otherwise these are not compatible
        if (num_colorspin != 12) {
            QDPIO::cerr << "Num colorspin ( " << num_colorspin << " != 12 " << std::endl;
            QDP_abort(1);
        }

#pragma omp parallel for collapse(2)
        for (int cb = 0; cb < 2; ++cb) {
            for (int cbsite = 0; cbsite < num_cb_sites; ++cbsite) {
                float *prop_data = nullptr;

                if (dir != 8) {
                    prop_data = coarse_out.GetSiteDirDataPtr(cb, cbsite, dir);
                } else {
                    prop_data = coarse_out.GetSiteDiagDataPtr(cb, cbsite);
                }

                for (int colorspin_col = 0; colorspin_col < num_colorspin; ++colorspin_col) {
                    int spin_col = colorspin_col / 3;
                    int color_col = colorspin_col % 3;

                    for (int colorspin_row = 0; colorspin_row < num_colorspin; ++colorspin_row) {
                        int spin_row = colorspin_row / 3;
                        int color_row = colorspin_row % 3;

                        int qdpxx_site = (rb[cb].siteTable())[cbsite];
                        prop_data[RE +
                                  n_complex * (colorspin_row + num_colorspin * colorspin_col)] =
                            qdpxx_in.elem(qdpxx_site)
                                .elem(spin_row, spin_col)
                                .elem(color_row, color_col)
                                .real();
                        prop_data[IM +
                                  n_complex * (colorspin_row + num_colorspin * colorspin_col)] =
                            qdpxx_in.elem(qdpxx_site)
                                .elem(spin_row, spin_col)
                                .elem(color_row, color_col)
                                .imag();
                    }
                }
            }
        }
    }

    void CoarseGaugeLinkToQDPProp(const CoarseGauge &coarse_in, LatticePropagator &qdpxx_out,
                                  IndexType dir) {
        IndexType num_colorspin = coarse_in.GetNumColorSpin();
        IndexType num_cb_sites = coarse_in.GetInfo().GetNumCBSites();

        // Assert site tables are equal
        if (num_cb_sites != rb[0].numSiteTable()) {
            QDPIO::cerr << "Num cb sites in coarse = " << num_cb_sites
                        << " But QDP++ checkerboard size is" << rb[0].numSiteTable() << std::endl;
            QDP_abort(1);
        }

        // Assert num colorspin is 12 otherwise these are not compatible
        if (num_colorspin != 12) {
            QDPIO::cerr << "Num colorspin ( " << num_colorspin << " != 12 " << std::endl;
            QDP_abort(1);
        }

#pragma omp parallel for collapse(2)
        for (int cb = 0; cb < 2; ++cb) {
            for (int cbsite = 0; cbsite < num_cb_sites; ++cbsite) {

                const float *prop_data = nullptr;

                if (dir != 8) {
                    prop_data = coarse_in.GetSiteDirDataPtr(cb, cbsite, dir);
                } else {
                    prop_data = coarse_in.GetSiteDiagDataPtr(cb, cbsite);
                }

                for (int colorspin_col = 0; colorspin_col < num_colorspin; ++colorspin_col) {
                    int spin_col = colorspin_col / 3;
                    int color_col = colorspin_col % 3;

                    for (int colorspin_row = 0; colorspin_row < num_colorspin; ++colorspin_row) {
                        int spin_row = colorspin_row / 3;
                        int color_row = colorspin_row % 3;

                        int qdpxx_site = (rb[cb].siteTable())[cbsite];
                        qdpxx_out.elem(qdpxx_site)
                            .elem(spin_row, spin_col)
                            .elem(color_row, color_col)
                            .real() = prop_data[RE + n_complex * (colorspin_row +
                                                                  num_colorspin * colorspin_col)];
                        qdpxx_out.elem(qdpxx_site)
                            .elem(spin_row, spin_col)
                            .elem(color_row, color_col)
                            .imag() = prop_data[IM + n_complex * (colorspin_row +
                                                                  num_colorspin * colorspin_col)];
                    }
                }
            }
        }
    }
}
