/*
 * aggregation.h
 *
 *  Created on: Oct 6, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_AGGREGATION_H_
#define INCLUDE_LATTICE_AGGREGATION_H_

#include <vector>

namespace MGGeometry {

class Aggregation {
public:
	virtual ~Aggregation() {}
	virtual const std::vector<unsigned int>& GetBlockDimensions(void) const = 0;
	virtual unsigned int GetNumAggregates(void) const = 0;
	virtual const std::vector<unsigned int>& GetSourceSpins(unsigned int aggregate) const = 0;
	virtual const std::vector<unsigned int>& GetSourceColors(unsigned int aggregate) const = 0;
};

class FullSpinAggregation : public Aggregation {
public:
	FullSpinAggregation( const std::vector<unsigned int>& block_dims) : _block_dims(block_dims), _num_aggregates(1),
				_source_spins{0,1,2,3}, _source_colors{0,1,2} {}

	inline
	const std::vector<unsigned int>& GetBlockDimensions(void) const {
		return _block_dims;
	}

	inline
	unsigned int GetNumAggregates(void) const {
		return _num_aggregates;
	}

	inline
	const std::vector<unsigned int>& GetSourceSpins(unsigned int aggregate) const {
		return _source_spins;
	}

	inline
	const std::vector<unsigned int>& GetSourceColors(unsigned int aggregate) const {
		return _source_colors;
	}

private:
	std::vector<unsigned int> _block_dims;
	const unsigned int _num_aggregates;
	const std::vector<unsigned int> _source_spins;
	const std::vector<unsigned int> _source_colors;
};

class StandardAggregation : public Aggregation {
public:
	StandardAggregation( const std::vector<unsigned int>& block_dims) : _block_dims(block_dims), _num_aggregates(2),
				_source_spins{ {0,1},{2,3} }, _source_colors{{0,1,2},{0,1,2}} {}

	inline
	const std::vector<unsigned int>& GetBlockDimensions(void) const {
		return _block_dims;
	}

	inline
	unsigned int GetNumAggregates(void) const {
		return _num_aggregates;
	}

	inline
	const std::vector<unsigned int>& GetSourceSpins(unsigned int aggregate) const {
		return _source_spins[aggregate];
	}

	inline
	const std::vector<unsigned int>& GetSourceColors(unsigned int aggregate) const {
		return _source_colors[aggregate];
	}

private:
	std::vector<unsigned int> _block_dims;
	const unsigned int _num_aggregates;
	const std::vector<unsigned int> _source_spins[2];
	const std::vector<unsigned int> _source_colors[2];
};


}



#endif /* INCLUDE_LATTICE_AGGREGATION_H_ */
