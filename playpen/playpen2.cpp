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

// Spinor Needs GetLatticeInfo function
// BlockSpinor needs Index(dst_spin, block, s_spin, color, reim, blocksite) -- accessor
// Spinor Needs Index(source_spins[s_spin] color, reim, fullsite) -- accessor

template<typename SpinorType, typename BlockSpinorType >
void ConvertToBlockOrder(std::vector< BlockSpinorType >& out,
						const std::vector< SpinorType >& v_in,
						const Aggregation& aggr)
{
   if( out.size() !=v_in.size()) {
	   MasterLog(ERROR, "Blah");
   }

   if( out.size() == 0 ) {
	   MasterLog(ERROR, "Blah");

   }

   // Assume no elegance:
   const LatticeInfo& lat_info = v_in[0].GetLatticeInfo();

   IndexType num_blocks = aggr.GetNumBlocks();
   IndexType num_dest_spins = aggr.GetNumAggregates();
   IndexType num_dest_colors = lat_info.GetNumColors();
   IndexType num_block_sites = aggr.GetBlockVolume();

   IndexArray& blocks_per_dim = aggr.GetNumBlocksPerDim();
   IndexArray& block_dims = aggr.GetBlockDimensions();
   IndexArray& lat_dims = lat_info.GetLatticeDimensions(); // Technically this could come from a lattice info


#pragma omp parallel for collapse(3)
	for (int vec = 0; vec < v_in.size(); ++vec) {
		for (int block = 0; block < num_blocks; ++block) {
			for (int dst_spin = 0; dst_spin < num_dest_spins; dst_spin++) {


					// Convert Block ID to coords of the block.
					IndexArray block_coords(n_dim);
					IndexToCoords(block, blocks_per_dim, block_coords);

					// Compute Block Origin
					IndexArray block_origin(block_coords);
					for (int mu = 0; mu < n_dim; ++mu)
						block_origin[mu] *= block_dims[mu];

					// Get the spin components making up this aggregate
					const std::vector<IndexType>& source_spins =
							aggr.GetSourceSpins(dst_spin);

					IndexType n_spin = source_spins.size();


					for (IndexType s_spin = 0; s_spin < n_spin; ++s_spin) {

					   // If color*reim*blocksite were together it would allow a longer
					   // inner loop for vectorization. Can I do collapse() with omp pragma simd?
					   // However, the site access on the unblocked spinor would not be vectorizable.

						for (int color = 0; color < num_dest_colors; color++) {
							for (int reim = 0; reim < n_complex; reim++) {

								// Vectorizable Inner Loop
								for (IndexType blocksite = 0; blocksite < aggr.GetBlockVolume();
										++blocksite) {

									// Turn block local site index, into node_local full index
									IndexArray coords(n_dim);

									IndexToCoords(blocksite, block_dims, coords);
									for (int mu = 0; mu < n_dim; ++mu)
										coords[mu] += block_origin[mu];


									out.Index(block, dst_spin, vec, blocksite, s_spin, color, reim) =
										v_in[vec].Index(coords,source_spins[s_spin],
												color, reim);
								} //blocksite
							} //reim
						}//color
					}// s_spin

			} //dst_spin
		} // block
	} // vec and OMP loop
} // Function


