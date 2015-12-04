/*
 * playpen2.cpp
 *
 *  Created on: Oct 19, 2015
 *      Author: bjoo
 */

/* Things you may want to do during a multi-grid implementation */

/** You need to be able to orthonormalize Bock Vectors
 *
 * @param v      -- An array of fine vectors in some external (e.g. QD
 */
#include "lattice/lattice_info.h"
#include "lattice/nodeinfo.h"
#include "lattice/aggregation.h"
#include "lattice/lattice_spinor.h"
#include "lattice/geometry_utils.h"
#include <complex>
#include <vector>


/* In order to have an array of Spinors, Spinors must support a copy constructor.
 * However, probably you don't want to deep copy your vector spaces
 *
 */
using namespace MGGeometry;
using namespace MGUtils;

// Option 1:
template<typename T, typename BlockedLayout>
void BlockOrthonormalize(std::vector<GenericLayoutContainer<T,BlockedLayout>>& vectors)
{
	auto num_vectors = vectors.size();

	// Dumb?
	if( num_vectors == 0 ) return;

	// Vectors zero now is guaranteed to exist.
	const Aggregation& aggr = vectors[0].GetAggregation();

	IndexType num_blocks = aggr.GetNumBlocks();
	IndexType num_outerspins = aggr.GetNumAggregates();

	// There is some amount of nested parallelism needed. I am not going to bother with it
	// I will loop this level without threading, and I'll thread over the actual spinors.

	for(IndexType block =0; block < num_blocks; ++block) {
		for(IndexType outer_spin=0; outer_spin < num_outerspins; ++outer_spin) {

			// This is the sub-spinor type
			using subview_type = GenericLayoutContainer<T,LayoutTraits<BlockedLayout>::subview_layout_type>;

			// A vector to hold the sub-spinors
			std::vector<GenericLayoutContainer<T,subview_type>> block_spinors(num_vectors);
			for(auto v=0; v < num_vectors; ++v) {

				block_spinors[v] = vectors[v].GetSubview(block, outer_spin);

			}

			GramSchmidt(block_spinors);

		}
	}
}



