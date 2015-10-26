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

   unsigned int num_blocks = aggr.GetNumBlocks();
   unsigned int num_dest_spins = aggr.GetNumAggregates();
   unsigned int num_dest_colors = lat_info.GetNumColors();
   unsigned int num_block_sites = aggr.GetBlockVolume();

   std::vector<unsigned int>& blocks_per_dim = aggr.GetNumBlocksPerDim();
   std::vector<unsigned int>& block_dims = aggr.GetBlockDimensions();
   std::vector<unsigned int>& lat_dims = lat_info.GetLatticeDimensions(); // Technically this could come from a lattice info


#pragma omp parallel for collapse(3)
	for (int vec = 0; vec < v_in.size(); ++vec) {
		for (int block = 0; block < num_blocks; ++block) {
			for (int dst_spin = 0; dst_spin < num_dest_spins; dst_spin++) {


					// Convert Block ID to coords of the block.
					std::vector<unsigned int> block_coords(n_dim);
					IndexToCoords(block, blocks_per_dim, block_coords);

					// Compute Block Origin
					std::vector<unsigned int> block_origin(block_coords);
					for (int mu = 0; mu < n_dim; ++mu)
						block_origin[mu] *= block_dims[mu];

					// Get the spin components making up this aggregate
					const std::vector<unsigned int>& source_spins =
							aggr.GetSourceSpins(dst_spin);

					unsigned int n_spin = source_spins.size();


					for (unsigned int s_spin = 0; s_spin < n_spin; ++s_spin) {

					   // If color*reim*blocksite were together it would allow a longer
					   // inner loop for vectorization. Can I do collapse() with omp pragma simd?
					   // However, the site access on the unblocked spinor would not be vectorizable.

						for (int color = 0; color < 3; color++) {
							for (int reim = 0; reim < 2; reim++) {

								// Vectorizable Inner Loop
								for (unsigned int blocksite = 0; blocksite < aggr.GetBlockVolume();
										++blocksite) {

									// Turn block local site index, into node_local full index
									std::vector<unsigned int> coords_in_block(n_dim);

									IndexToCoords(blocksite, block_dims, coords_in_block);
									for (int mu = 0; mu < n_dim; ++mu)
										coords_in_block[mu] += block_origin[mu];

									unsigned int fullsite = CoordsToIndex(coords_in_block, lat_dims);


									out.Index(block, dst_spin, vec, s_spin,
										color, reim, blocksite) =
										v_in[vec].Index(fullsite,source_spins[s_spin],
												color, reim);
								} //blocksite
							} //reim
						}//color
					}// s_spin

			} //dst_spin
		} // block
	} // vec and OMP loop
} // Function


