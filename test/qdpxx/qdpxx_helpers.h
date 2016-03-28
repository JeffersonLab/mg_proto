/*
 * qdpxx_helpers.h
 *
 *  Created on: Mar 17, 2016
 *      Author: bjoo
 */

#ifndef TEST_QDPXX_QDPXX_HELPERS_H_
#define TEST_QDPXX_QDPXX_HELPERS_H_

#include <qdp.h>
#include "lattice/constants.h"
#include "lattice/coarse/coarse_types.h"


namespace MGTesting {
	void initQDPXXLattice(const MG::IndexArray& latdims);
	void QDPSpinorToCoarseSpinor(const QDP::LatticeFermion& qdpxx_in,
	    			CoarseSpinor& coarse_out);
	void CoarseSpinorToQDPSpinor(const CoarseSpinor& coarse_in,
	     			QDP::LatticeFermion& qdpxx_out);

	void QDPPropToCoarseGaugeLink(const QDP::LatticePropagator& qdpxx_in,
	    			CoarseGauge& coarse_out, IndexType dir);

	void CoarseGaugeLinkToQDPProp(const CoarseGauge& coarse_in,
	     			QDP::LatticePropagator& qdpxx_out, IndexType dir);

	void QDPPropToCoarseClover(const QDP::LatticePropagator& qdpxx_in,
							   CoarseClover& coarse_out);

};




#endif /* TEST_QDPXX_QDPXX_HELPERS_H_ */
