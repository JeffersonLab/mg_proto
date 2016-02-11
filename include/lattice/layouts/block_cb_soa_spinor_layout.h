/*
 * spinor.h
 *
 *  Created on: Oct 20, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_BLOCK_SOA_SPINOR_LAYOUT_H_
#define INCLUDE_LATTICE_BLOCK_SOA_SPINOR_LAYOUT_H_

#include "MG_config.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/aggregation.h"
#include "lattice/geometry_utils.h"
#include "cb_soa_spinor_layout.h"
#include "utils/memory.h"
#include "utils/print_utils.h"
#include "lattice/layout_traits.h"
#include <memory>
namespace MG {


template<typename T, typename Aggregation=StandardAggregation>
class BlockAggregateVectorLayout {
private:
	const LatticeInfo _info;
	const Aggregation& _aggr;
	const IndexType _n_blocks;
	const IndexType _n_chiral_blocks;

	const IndexType _n_block_sites;
	const IndexType _n_colors;
	const IndexType _n_spins;


	IndexType _n_elem_per_chiral_block;
	IndexType _n_elem;

	// Inner layout type will be:
	using BlockInnerLayout = CBSOASpinorLayout<T>;
	BlockInnerLayout _block_layout;

public:
	BlockAggregateVectorLayout(const BlockAggregateVectorLayout& in) = default;

	BlockAggregateVectorLayout(const LatticeInfo& info, const Aggregation& aggr) :
			_info(info), _aggr(aggr), _n_blocks(aggr.GetNumBlocks()), _n_chiral_blocks(
					aggr.GetNumAggregates()), _n_block_sites(aggr.GetBlockVolume()),
					_n_colors(aggr.GetSourceColors(0).size()),
					_n_spins(aggr.GetSourceSpins(0).size()),
					_block_layout(LatticeInfo(info.GetLatticeOrigin(),
																   aggr.GetBlockDimensions(),
																   aggr.GetNumSourceSpins(),
																   aggr.GetNumSourceColors(),
																   info.GetNodeInfo()))
	{

		/* Work out the Layout for a single block, at the origin of our sublattice.
		 *
		 */

		_n_elem_per_chiral_block = _block_layout.GetNumData();
	    _n_elem = _n_elem_per_chiral_block * _n_chiral_blocks * _n_blocks;

	}

	~BlockAggregateVectorLayout() {
	}

	inline
	IndexType
	GetNumBlocks(void) const { return _n_blocks; }


	// Info is valid, since it refers to the whole lattice not just the block
	// So origin is OK.
	// Also all the spin-color info is there just relaid out.
	const LatticeInfo& GetLatticeInfo(void) const {
		return _info;
	}
	const Aggregation& GetAggregation(void) const {
		return _aggr;
	}

	inline IndexType ContainerIndex(IndexType block_index,
									IndexType chiral_block_index,
									IndexType block_site_index,
									IndexType spin_index,
									IndexType color_index,
									IndexType reim) const {

		IndexType n_th_chiral_block = chiral_block_index + _n_chiral_blocks * block_index;

		IndexType block_offset = _block_layout.ContainerIndex(block_site_index,spin_index,color_index,reim);

		return block_offset + _n_elem_per_chiral_block*n_th_chiral_block;

	}


	inline IndexType GetSubviewOffset(IndexType block_index, IndexType chiral_block_index) const
	{
		return (chiral_block_index + _n_chiral_blocks*block_index)*_n_elem_per_chiral_block;

	}

	inline
	BlockInnerLayout GetSubviewLayout(IndexType block_index, IndexType chiral_block_index) const
	{
		auto binfo = _block_layout.GetLatticeInfo();
		/* Work out coords of the block in the node */
		IndexArray bcoords={0,0,0,0};
		IndexToCoords(block_index, _aggr.GetNumBlocksPerDim(), bcoords);

		/* Now work out the coords of origin of the block.
		 * This is just the LatticeOrigin of the node, offset by the block coords and dimensions
		 */
		IndexArray block_origin = _info.GetLatticeOrigin();
		for(IndexType mu=0; mu < n_dim;++mu) {
			block_origin[mu] += bcoords[mu]*_aggr.GetBlockDimensions()[mu];
		}

		/* This returns a layout with the origin amended. */
		/* This will construct and copy */
		return BlockInnerLayout(LatticeInfo(block_origin, _aggr.GetBlockDimensions(), binfo.GetNumSpins(), binfo.GetNumColors(), binfo.GetNodeInfo()));

	}

	inline size_t GetNumData() const {
		return _n_elem;
	}

	inline size_t GetDataInBytes() const {
		return _n_elem * sizeof(T);
	}

};



template<>
struct LayoutTraits<BlockAggregateVectorLayout<float>>
{
	typedef float value_type;
	const bool has_subviews = true;
	typedef CBSOASpinorLayout<float> subview_layout_type;
};

template<>
struct LayoutTraits<BlockAggregateVectorLayout<double>>
{
	typedef double value_type;
	const bool has_subviews = true;
	typedef CBSOASpinorLayout<double> subview_layout_type;
};

template<>
struct LayoutTraits<BlockAggregateVectorLayout<IndexType>>
{
	typedef IndexType value_type;
	const bool has_subviews = true;
	typedef CBSOASpinorLayout<IndexType> subview_layout_type;
};

} // Namespace




#endif /* INCLUDE_LATTICE_LATTICE_SPINOR_H_ */
