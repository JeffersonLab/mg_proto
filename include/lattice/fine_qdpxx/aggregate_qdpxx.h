/*
 * aggregate_qdpxx.h
 *
 *  Created on: Dec 9, 2016
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_FINE_QDPXX_AGGREGATE_QDPXX_H_
#define INCLUDE_LATTICE_FINE_QDPXX_AGGREGATE_QDPXX_H_

#include "lattice/coarse/coarse_types.h"
#include "lattice/fine_qdpxx/clover_term_qdp_w.h"
#include "qdp.h"

namespace MG {

    //! Apply a single direction of Dslash
    void DslashDirQDPXX(QDP::LatticeFermion &out, const QDP::multi1d<QDP::LatticeColorMatrix> &u,
                        const QDP::LatticeFermion &in, int dir);

    //! v *= alpha (alpha is real) over aggregate in a site, v is a QDP++ Lattice Fermion
    void axAggrQDPXX(const double alpha, QDP::LatticeFermion &v, int site, int aggr);

    //! y += alpha * x (alpha is complex) over aggregate in a site, x, y are QDP++ LatticeFermions;
    void caxpyAggrQDPXX(const std::complex<double> &alpha, const QDP::LatticeFermion &x,
                        QDP::LatticeFermion &y, int site, int aggr);

    //! return || v ||^2 over a spin aggregate in a site, v is a QDP++ LatticeFermion
    double norm2AggrQDPXX(const QDP::LatticeFermion &v, int site, int aggr);

    //! return < left | right > = sum left^\dagger_i * right_i for a spin aggregate, with a site
    std::complex<double> innerProductAggrQDPXX(const QDP::LatticeFermion &left,
                                               const QDP::LatticeFermion &right, int site,
                                               int aggr);

    //! Orthonormalize vecs over the spin aggregates within the sites
    void orthonormalizeAggregatesQDPXX(QDP::multi1d<QDP::LatticeFermion> &vecs);

    //! Extract the spins belonging to a given aggregate from QDP++ source vector src, into QDP++ target vector target
    void extractAggregateQDPXX(QDP::LatticeFermion &target, const QDP::LatticeFermion &src,
                               int aggr);

    //! 'Restrict' a QDP++ spinor to a CoarseSpinor with the same geometry (a 'block' is a 'site')
    void restrictSpinorQDPXXFineToCoarse(const QDP::multi1d<QDP::LatticeFermion> &v,
                                         const QDP::LatticeFermion &ferm_in, CoarseSpinor &out);

    //! 'Prolongate' a CoarseSpinor to a QDP++ Fine Spinor  ( a 'block' is a 'site'
    void prolongateSpinorCoarseToQDPXXFine(const QDP::multi1d<QDP::LatticeFermion> &v,
                                           const CoarseSpinor &coarse_in,
                                           QDP::LatticeFermion &fine_out);

    //! Coarsen one direction of a 'dslash' link ( 1 block = 1 site )
    void dslashTripleProductSiteDirQDPXX(int dir, const QDP::multi1d<QDP::LatticeColorMatrix> &u,
                                         const QDP::multi1d<QDP::LatticeFermion> &in_vecs,
                                         CoarseGauge &u_coarse);

    //! Coarsen the clover term (1 block = 1 site )
    void clovTripleProductSiteQDPXX(const QDPCloverTerm &clov,
                                    const QDP::multi1d<QDP::LatticeFermion> &in_vecs,
                                    CoarseGauge &cl_coarse);

    //! Coarsen one direction of a 'dslash' link, but the vector space is strictly 12x12 and is held in a propagator -- for testing
    void dslashTripleProduct12x12SiteDirQDPXX(int dir,
                                              const QDP::multi1d<QDP::LatticeColorMatrix> &u,
                                              const QDP::LatticePropagator &in_prop,
                                              QDP::LatticePropagator &out_prop);

    //! Coarsen one direction of the clover term, but the vector space is strictly 12x12 per site and is held in a propagator -- for testing
    void clovTripleProduct12cx12SiteQDPXX(const QDPCloverTerm &clov,
                                          const QDP::LatticePropagator &in_prop,
                                          QDP::LatticePropagator &out_prop);
}

#endif /* TEST_QDPXX_AGGREGATE_QDPXX_H_ */
