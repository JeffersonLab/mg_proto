/*
 * aggregation.h
 *
 *  Created on: Oct 6, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_AGGREGATION_H_
#define INCLUDE_LATTICE_AGGREGATION_H_

#include <vector>
#include "lattice/constants.h"
#include "lattice/geometry_utils.h"
namespace MG {


class StandardAggregation  {
public:
	StandardAggregation(const IndexArray& lat_dims,
						const IndexArray& block_dims) :
			_block_dims(block_dims), _num_aggregates(2),
			_source_spins{ { 0, 1 }, { 2, 3 } },
			_source_colors{ { 0, 1, 2 }, { 0, 1, 2 } } {

		AssertVectorsDivideEachOther(lat_dims, block_dims);

		_blocks_per_dim[0] = lat_dims[0] / block_dims[0];
		_num_blocks = _blocks_per_dim[0];

		for (IndexType mu = 1; mu < n_dim; ++mu) {
			_blocks_per_dim[mu] = lat_dims[mu] / block_dims[mu];
			_num_blocks *= _blocks_per_dim[mu];
		}

		_block_volume = _block_dims[0];
		for(IndexType mu=1; mu < n_dim; ++mu) {
			_block_volume *= _block_dims[mu];
		}
	}

	~StandardAggregation() {}

	inline
	IndexType GetBlockVolume(void) const {
		return _block_volume;
	}


	inline
	const IndexArray& GetBlockDimensions(void) const {
		return _block_dims;
	}

	inline
	IndexType GetNumAggregates(void) const {
		return _num_aggregates;
	}

	inline
	const std::vector<IndexType>& GetSourceSpins(IndexType aggregate) const {
		return _source_spins[aggregate];
	}

	inline IndexType GetNumSourceSpins() const {
		return static_cast<IndexType>(_source_spins[0].size());
	}

	inline IndexType GetNumSourceColors() const {
		return static_cast<IndexType>(_source_colors[0].size());
	}

	inline
	const std::vector<IndexType>& GetSourceColors(IndexType aggregate) const {
		return _source_colors[aggregate];
	}

	inline
	IndexType GetNumBlocks(void) const {
		return _num_blocks;
	}

	inline
	const IndexArray& GetNumBlocksPerDim(void) const {
		return _blocks_per_dim;
	}
private:
	IndexArray _block_dims;
	const IndexType _num_aggregates;
	const std::vector<IndexType> _source_spins[2];
	const std::vector<IndexType> _source_colors[2];
	IndexType _num_blocks;
	IndexArray _blocks_per_dim;
	IndexType _block_volume;
};


}



#endif /* INCLUDE_LATTICE_AGGREGATION_H_ */
