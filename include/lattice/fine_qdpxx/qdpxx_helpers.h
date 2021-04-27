/*
 * qdpxx_helpers.h
 *
 *  Created on: Mar 17, 2016
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_FINE_QDPXX_QDPXX_HELPERS_H_
#define INCLUDE_LATTICE_FINE_QDPXX_QDPXX_HELPERS_H_

#include "lattice/coarse/coarse_types.h"
#include "lattice/constants.h"
#include <qdp.h>

namespace MG {

    // Convert a QDP++ Spinor into a Coarse Spinor Datatype.
    // Geometries, Spins, Colors, must match
    void QDPSpinorToCoarseSpinor(const QDP::LatticeFermion &qdpxx_in, CoarseSpinor &coarse_out);

    // Convert a CoarseSpinor Datatype to a QDP++ Spinor Datatype
    // Geometries, Spins, Colors must match
    void CoarseSpinorToQDPSpinor(const CoarseSpinor &coarse_in, QDP::LatticeFermion &qdpxx_out);

    void QDPGaugeLinksToCoarseGaugeLinks(const QDP::multi1d<QDP::LatticeColorMatrix> &qdp_u_in,
                                         CoarseGauge &gauge_out);
    void CoarseGaugeLinksToQDPGaugeLinks(const CoarseGauge &gauge_in,
                                         QDP::multi1d<QDP::LatticeColorMatrix> &qdp_u_out);

    void QDPPropToCoarseGaugeLink(const QDP::LatticePropagator &qdpxx_in, CoarseGauge &coarse_out,
                                  IndexType dir);

    void CoarseGaugeLinkToQDPProp(const CoarseGauge &coarse_in, QDP::LatticePropagator &qdpxx_out,
                                  IndexType dir);
}

#endif /* TEST_QDPXX_QDPXX_HELPERS_H_ */
