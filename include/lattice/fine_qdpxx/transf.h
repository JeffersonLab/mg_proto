/*
 * transf.h
 *
 *  Created on: Mar 17, 2016
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_FINE_QDPXX_TRANSF_H_
#define INCLUDE_LATTICE_FINE_QDPXX_TRANSF_H_

#include <qdp.h>
using namespace QDP;

namespace MG {
    void FermToProp(const LatticeFermion &a, LatticePropagator &b, int color_index, int spin_index);
    void PropToFerm(const LatticePropagator &b, LatticeFermion &a, int color_index, int spin_index);
}

#endif /* TEST_QDPXX_TRANSF_H_ */
