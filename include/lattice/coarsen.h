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


	/*!  Coarsen a lattice info given a number of vectors and an aggregation */
	template<typename Aggregation>
	LatticeInfo CoarsenLattice(const LatticeInfo& fine_geom,
							   const Aggregation& blocking,
							   unsigned int num_vec)
	{
		IndexArray coarse_dims(fine_geom.GetLatticeDimensions());
		IndexArray blocking_dims(blocking.GetBlockDimensions());


		// Check Divisibility
		for (unsigned int mu = 0; mu < n_dim; ++mu) {
			if (coarse_dims[mu] % blocking_dims[mu] != 0) {
				MasterLog(ERROR, "blocking does not divide lattice in dimension %d",
						mu);
			} else {
				// If divisible, than divide
				coarse_dims[mu] /= blocking_dims[mu];
			}
		}

		int n_colors = num_vec; // The number of vectors is the number of coarse colors
		int n_spins = blocking.GetNumAggregates(); // The number of aggregates is the new number of spins
		LatticeInfo ret_val(coarse_dims, n_spins, n_colors, NodeInfo()); // This is the value to return. Initialize

		return ret_val;
	} // End of function
}


#endif /* INCLUDE_LATTICE_COARSEN_H_ */
