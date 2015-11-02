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

namespace MGGeometry {
#if 1
template<typename T>
class BlockAggregateVectorLayout {
private:
	const LatticeInfo& _info;
	const Aggregation& _aggr;
	const IndexType _n_blocks;
	const IndexType _n_chiral_blocks;

	const IndexType _n_block_sites;
	const IndexType _n_colors;
	const IndexType _n_spins;

	IndexType _n_block_sites_stride;
	IndexType _n_elem_per_chiral_block;
	IndexType _n_elem;
public:
	BlockAggregateVectorLayout(const BlockAggregateVectorLayout& in) =  default;

	BlockAggregateVectorLayout(const LatticeInfo& info, const Aggregation& aggr) :
			_info(info), _aggr(aggr), _n_blocks(aggr.GetNumBlocks()), _n_chiral_blocks(
					aggr.GetNumAggregates()), _n_block_sites(aggr.GetBlockVolume()),
					_n_colors(aggr.GetSourceColors(0).size()),
					_n_spins(aggr.GetSourceSpins(0).size())
	{
		IndexType sites_per_line = MG_DEFAULT_ALIGNMENT/(n_complex*sizeof(T));
		IndexType sites_rem = _n_block_sites % sites_per_line;
	    if ( sites_rem > 0 ) {
	  		  _n_block_sites_stride = _n_block_sites+(sites_per_line-sites_rem);
	    }
	    else {
	  		  _n_block_sites_stride = _n_block_sites; // No padding needed
	    }
	    _n_elem_per_chiral_block = _n_colors*_n_spins*n_complex*_n_block_sites_stride;
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

		IndexType block_offset = reim + n_complex*(block_site_index +
							_n_block_sites_stride*(color_index + _n_colors * spin_index));

		return block_offset + _n_elem_per_chiral_block*n_th_chiral_block;

	}

	inline size_t DataNumElem() const {
		return _n_elem;
	}

	inline size_t DataInBytes() const {
		return _n_elem * sizeof(T);
	}

};
#endif

/** Blocked SOA Spinor Layout
 *  Takes a lattice info and some Block sizes
 *
 *
 *
 *
 */

template<typename T>
class BlockAggregateVectorArrayLayout {
private:
	const LatticeInfo& _info;
	const Aggregation& _aggr;
	const IndexType _n_vec;
	const IndexType _n_blocks;
	const IndexType _n_chiral_blocks;

	const IndexType _n_block_sites;
	const IndexType _n_colors;
	const IndexType _n_spins;

	IndexType _n_block_sites_stride;
	IndexType _n_elem_per_chiral_block;
	IndexType _n_elem;
public:
	BlockAggregateVectorArrayLayout(const BlockAggregateVectorArrayLayout& in) = default;

	BlockAggregateVectorArrayLayout(const LatticeInfo& info,
			const Aggregation& aggr, IndexType n_vec) :
			_info(info), _aggr(aggr), _n_vec(n_vec), _n_blocks(
					aggr.GetNumBlocks()), _n_chiral_blocks(
					aggr.GetNumAggregates()), _n_block_sites(
					aggr.GetBlockVolume()), _n_colors(
					aggr.GetSourceColors(0).size()), _n_spins(
					aggr.GetSourceSpins(0).size()) {


		IndexType sites_per_line = MG_DEFAULT_ALIGNMENT / (n_complex * sizeof(T));
		IndexType sites_rem = _n_block_sites % sites_per_line;
		if (sites_rem > 0) {
			_n_block_sites_stride = _n_block_sites
					+ (sites_per_line - sites_rem);
		} else {
			_n_block_sites_stride = _n_block_sites; // No padding needed
		}
		_n_elem_per_chiral_block = _n_colors * _n_spins * n_complex
				* _n_block_sites_stride;
		_n_elem = _n_elem_per_chiral_block * _n_chiral_blocks * _n_vec* _n_blocks;
	}


	~BlockAggregateVectorArrayLayout() {
	}

	const LatticeInfo& GetLatticeInfo(void) const {
		return _info;
	}
	const Aggregation& GetAggregation(void) const {
		return _aggr;
	}

	inline IndexType ContainerIndex(IndexType block_index,
			IndexType chiral_block_index, IndexType vec_index,
			IndexType block_site_index, IndexType spin_index,
			IndexType color_index, IndexType reim) const {

		IndexType n_th_chiral_block = vec_index + _n_vec*(chiral_block_index + _n_chiral_blocks * block_index);

		IndexType block_offset = reim + n_complex*(block_site_index +
								_n_block_sites_stride*(color_index + _n_colors * spin_index));

			return block_offset + _n_elem_per_chiral_block*n_th_chiral_block;

	}

	inline size_t DataNumElem() const {
		return _n_elem;
	}

	inline size_t DataInBytes() const {
		return _n_elem * sizeof(T);
	}

	inline size_t GetNumVecs() const {
		return _n_vec;
	}


};




} // Namespace




#endif /* INCLUDE_LATTICE_LATTICE_SPINOR_H_ */
