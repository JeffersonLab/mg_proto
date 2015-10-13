/*
 * coarsen.h
 *
 *  Created on: Oct 12, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_COARSEN_H_
#define INCLUDE_LATTICE_COARSEN_H_

#include "lattice/lattice_info.h"
#include "lattice/aggregation.h"

namespace MGGeometry {
	LatticeInfo CoarsenLattice(const LatticeInfo& fine_geom, const Aggregation& blocking, unsigned int num_vec);
}


#endif /* INCLUDE_LATTICE_COARSEN_H_ */
