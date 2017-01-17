/*
 * block.h
 *
 *  Created on: Jan 16, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_COARSE_BLOCK_H_
#define INCLUDE_LATTICE_COARSE_BLOCK_H_

#include "lattice/constants.h"
#include <vector>



namespace MG {

struct CBSite
{
	IndexType cb;
	IndexType site;
};

class Block {
public:
	// I am doing it this way so we can make a std::vector of these...
	// That  means it needs an argument free constructor

	Block(void) {
		_created = false;
		_num_sites = 0;
	}

	void create(const IndexArray local_lattice_dimensions,
				const IndexArray block_origin,
				const IndexArray block_dimensions,
				const IndexArray local_lattice_origin);

	inline
	const std::vector<IndexType>& getSiteList(void) const {
		return _site_list;
	}

	inline
	const std::vector< CBSite >& getCBSiteList(void) const {
		return _cbsite_list;
	}

	inline
	const bool isCreated(void) {
		return _created;
	}

	inline
	unsigned int getNumSites() const {
		return _num_sites;
	}

	// Destructor is automatic
	~Block() {}
private:
	IndexArray _local_latt_origin;
	IndexArray _origin;
	IndexArray _dimensions;
	unsigned int _num_sites;
	bool _created = false;
	std::vector<IndexType> _site_list;
	std::vector< CBSite > _cbsite_list;
};

void CreateBlockList(std::vector<Block>& blocklist,
					IndexArray& blocked_lattice_dimensions,
					const IndexArray& local_lattice_dimensions,
					const IndexArray& block_dimensions,
					const IndexArray& local_lattice_origin);

}


#endif /* TEST_QDPXX_BLOCK_H_ */
