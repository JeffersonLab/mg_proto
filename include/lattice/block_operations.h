/*
 * block_operations.h
 *
 *  Created on: Oct 28, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_BLOCK_OPERATIONS_H_
#define INCLUDE_LATTICE_BLOCK_OPERATIONS_H_

#include "lattice/aggregation.h"
#include "lattice/lattice_spinor.h"

namespace MGGeometry {

	template<typename BlockedSpinorOut, typename SpinorIn>
	void zip(BlockedSpinorOut& v_out, const SpinorIn& spinor_in)
	{
		auto& layout = v_out.GetLayout();
		auto& info = v_out.GetLatticeInfo();
		auto& aggr = layout.GetAggregation();
		IndexType num_blocks = aggr.GetNumBlocks();
		IndexType num_dest_spins = aggr.GetNumAggregates();


		auto& blocks_per_dim = aggr.GetNumBlocksPerDim();
		auto& block_dims = aggr.GetBlockDimensions();
		auto& latdims = info.GetLatticeDimensions();

#pragma omp parallel for shared(blocks_per_dim) collapse(2)
		for (IndexType block = 0; block < num_blocks; ++block) {
			for (IndexType dst_spin = 0; dst_spin < num_dest_spins; dst_spin++) {

				// Convert Block ID to coords of the block.
				IndexArray block_coords;
				IndexToCoords(block, blocks_per_dim, block_coords);

				// Compute Block Origin
				IndexArray block_origin(block_coords);
				for (IndexType mu = 0; mu < n_dim; ++mu)
					block_origin[mu] *= block_dims[mu];

				// Get the spin components making up this aggregate
				const std::vector<IndexType>& source_spins =
						aggr.GetSourceSpins(dst_spin);
				const std::vector<IndexType>& source_colors =
						aggr.GetSourceColors(dst_spin);

				IndexType n_spin = source_spins.size();
				IndexType n_color = source_colors.size();

				for (IndexType s_spin = 0; s_spin < n_spin; ++s_spin) {

					// If color*reim*blocksite were together it would allow a longer
					// inner loop for vectorization. Can I do collapse() with omp pragma simd?
					// However, the site access on the unblocked spinor would not be vectorizable.

					for (IndexType s_color = 0; s_color < n_color; s_color++) {
						for (IndexType reim = 0; reim < n_complex; reim++) {

							// Vectorizable Inner Loop
							for (IndexType blocksite = 0;
									blocksite < aggr.GetBlockVolume();
									++blocksite) {

								// Turn block local site index, into node_local full index
								IndexArray coords;
								IndexToCoords(blocksite, block_dims, coords);
								for (IndexType mu = 0; mu < n_dim; ++mu)
									coords[mu] += block_origin[mu];
								IndexType fullsite = CoordsToIndex(coords,latdims);

								v_out.Index(block, dst_spin, blocksite,
										s_spin, s_color, reim) = spinor_in.Index(
										fullsite, source_spins[s_spin], source_colors[s_color],
										reim);
							} //blocksite
						} //reim
					} //color
				} // s_spin
			} //dst_spin
		} // block and OMP LOOP
	} // Function

	template<typename BlockedSpinorArrayOut, typename SpinorIn>
	void zip(BlockedSpinorArrayOut& v_out, const SpinorIn& spinor_in, IndexType vec)
	{
		auto& layout = v_out.GetLayout();
		auto& info = v_out.GetLatticeInfo();
		auto& aggr = layout.GetAggregation();
		IndexType num_blocks = aggr.GetNumBlocks();
		IndexType num_dest_spins = aggr.GetNumAggregates();


		auto& blocks_per_dim = aggr.GetNumBlocksPerDim();
		auto& block_dims = aggr.GetBlockDimensions();
		auto& latdims = info.GetLatticeDimensions();

#pragma omp parallel for shared(blocks_per_dim) collapse(2)
		for (IndexType block = 0; block < num_blocks; ++block) {
			for (IndexType dst_spin = 0; dst_spin < num_dest_spins; dst_spin++) {

				// Convert Block ID to coords of the block.
				IndexArray block_coords;
				IndexToCoords(block, blocks_per_dim, block_coords);

				// Compute Block Origin
				IndexArray block_origin(block_coords);
				for (IndexType mu = 0; mu < n_dim; ++mu)
					block_origin[mu] *= block_dims[mu];

				// Get the spin components making up this aggregate
				const std::vector<IndexType>& source_spins =
						aggr.GetSourceSpins(dst_spin);
				const std::vector<IndexType>& source_colors =
						aggr.GetSourceColors(dst_spin);

				IndexType n_spin = source_spins.size();
				IndexType n_color = source_colors.size();

				for (IndexType s_spin = 0; s_spin < n_spin; ++s_spin) {

					// If color*reim*blocksite were together it would allow a longer
					// inner loop for vectorization. Can I do collapse() with omp pragma simd?
					// However, the site access on the unblocked spinor would not be vectorizable.

					for (IndexType s_color = 0; s_color < n_color; s_color++) {
						for (IndexType reim = 0; reim < n_complex; reim++) {

							// Vectorizable Inner Loop
							for (IndexType blocksite = 0;
									blocksite < aggr.GetBlockVolume();
									++blocksite) {

								// Turn block local site index, into node_local full index
								IndexArray coords;
								IndexToCoords(blocksite, block_dims, coords);
								for (IndexType mu = 0; mu < n_dim; ++mu)
									coords[mu] += block_origin[mu];
								IndexType fullsite = CoordsToIndex(coords,latdims);

								v_out.Index(block, dst_spin, vec, blocksite,
										s_spin, s_color, reim) = spinor_in.Index(
										fullsite, source_spins[s_spin], source_colors[s_color],
										reim);
							} //blocksite
						} //reim
					} //color
				} // s_spin
			} //dst_spin
		} // block and OMP LOOP
	} // Function



	template<typename SpinorOut, typename BlockedSpinorIn>
	void unzip(SpinorOut& spinor_out, const BlockedSpinorIn& v_in)
	{
		auto& layout = v_in.GetLayout();
		auto& info = v_in.GetLatticeInfo();
		const Aggregation& aggr = layout.GetAggregation();
		IndexType num_chiral_blocks = aggr.GetNumAggregates();

		auto& blocks_per_dim = aggr.GetNumBlocksPerDim();
		auto& block_dims = aggr.GetBlockDimensions();
		auto& latdims = info.GetLatticeDimensions();

		for(IndexType chiral_block=0; chiral_block < num_chiral_blocks; ++chiral_block) {
			auto spin_list = aggr.GetSourceSpins(chiral_block);
			auto color_list = aggr.GetSourceColors(chiral_block);

			for(IndexType dst_spin=0; dst_spin <  spin_list.size(); ++dst_spin) {
				for(IndexType dst_color=0; dst_color < color_list.size(); ++dst_color) {
					auto spin = spin_list[dst_spin];
					auto color = color_list[dst_color];

					for(IndexType site=0; site < info.GetNumSites(); ++site) {

						IndexArray coords=info.GetLatticeOrigin(); // This is an nitial result that it discarded
						IndexToCoords(site,latdims,coords);

						IndexArray block_coords=coords;
						IndexArray coords_in_block=coords;

						for(IndexType mu=0; mu < n_dim; ++mu) {
							block_coords[mu]=coords[mu]/block_dims[mu];
							coords_in_block[mu]=coords[mu]%block_dims[mu];
						}
						IndexType blocksite_index = CoordsToIndex(coords_in_block, block_dims);
						IndexType block_index = CoordsToIndex(block_coords, blocks_per_dim);


						for(IndexType reim=0; reim < n_complex;++reim) {
							spinor_out.Index(site,spin,color,reim) = v_in.Index(block_index,chiral_block, blocksite_index,dst_spin,dst_color,reim);
						}
					}
				}
			}
		}
	} // Function


	template<typename SpinorOut, typename BlockSpinorArrayIn>
	void unzip(SpinorOut& spinor_out, const BlockSpinorArrayIn& v_in, IndexType vec)
	{
		auto& layout = v_in.GetLayout();
		auto& info = v_in.GetLatticeInfo();
		const Aggregation& aggr = layout.GetAggregation();
		IndexType num_chiral_blocks = aggr.GetNumAggregates();

		auto& blocks_per_dim = aggr.GetNumBlocksPerDim();
		auto& block_dims = aggr.GetBlockDimensions();
		auto& latdims = info.GetLatticeDimensions();

		for(IndexType chiral_block=0; chiral_block < num_chiral_blocks; ++chiral_block) {
			auto spin_list = aggr.GetSourceSpins(chiral_block);
			auto color_list = aggr.GetSourceColors(chiral_block);

			for(IndexType dst_spin=0; dst_spin <  spin_list.size(); ++dst_spin) {
				for(IndexType dst_color=0; dst_color < color_list.size(); ++dst_color) {
					auto spin = spin_list[dst_spin];
					auto color = color_list[dst_color];

					for(IndexType site=0; site < info.GetNumSites(); ++site) {

						IndexArray coords=info.GetLatticeOrigin(); // This is an nitial result that it discarded
						IndexToCoords(site,latdims,coords);

						IndexArray block_coords=coords;
						IndexArray coords_in_block=coords;

						for(IndexType mu=0; mu < n_dim; ++mu) {
							block_coords[mu]=coords[mu]/block_dims[mu];
							coords_in_block[mu]=coords[mu]%block_dims[mu];
						}
						IndexType blocksite_index = CoordsToIndex(coords_in_block, block_dims);
						IndexType block_index = CoordsToIndex(block_coords, blocks_per_dim);


						for(IndexType reim=0; reim < n_complex;++reim) {
							spinor_out.Index(site,spin,color,reim) = v_in.Index(block_index,chiral_block, vec, blocksite_index,dst_spin,dst_color,reim);
						}
					}
				}
			}
		}
	} // Function
}




#endif /* INCLUDE_LATTICE_BLOCK_OPERATIONS_H_ */
