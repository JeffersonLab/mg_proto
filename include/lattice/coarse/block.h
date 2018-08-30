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
	IndexArray coords;
};

class Block {
public:
	// I am doing it this way so we can make a std::vector of these...
	// That  means it needs an argument free constructor

	Block(void) {
		_created = false;
		_num_sites = 0;
		_num_cbsites = 0;
	}

	void create(const IndexArray local_lattice_dimensions,
				const IndexArray block_origin,
				const IndexArray block_dimensions,
				const IndexArray local_lattice_origin);

	//inline
	//const std::vector<IndexType>& getSiteList(void) const {
	//	return _site_list;
	//}

	inline
	const std::vector< CBSite >& getCBSiteList(void) const {
		return _cbsite_list;
	}



	inline
	const std::vector< CBSite >& getInnerBodySiteList(void) const {
		return _inner_body;
	}

	inline
	const std::vector< CBSite >& getFaceList(const int dir) const {
		return _face[dir];
	}

	inline
	const std::vector< CBSite>& getNotFaceList(const int dir) const {
		return _not_face[dir];
	}

	inline
	bool isCreated(void) {
		return _created;
	}

	inline
	unsigned int getNumSites() const {
		return _num_sites;
	}

	inline
	unsigned int getNumCBSites() const {
		return _num_cbsites;
	}
	// Destructor is automatic
	~Block() {}
private:
	IndexArray _local_latt_origin;
	IndexArray _origin;
	IndexArray _dimensions;
	unsigned int _num_sites;
	unsigned int _num_cbsites;

	bool _created = false;
	// All the sites in the block
	std::vector< CBSite > _cbsite_list;

	// Inner Body Sites
	std::vector< CBSite > _inner_body;
	std::vector< CBSite > _face[8];
	std::vector< CBSite > _not_face[8];

};

void CreateBlockList(std::vector<Block>& blocklist,
					IndexArray& coarse_lattice_dimensions,
					IndexArray& coarse_lattice_origin,
					const IndexArray& local_lattice_dimensions,
					const IndexArray& block_dimensions,
					const IndexArray& local_lattice_origin);

}


#endif /* TEST_QDPXX_BLOCK_H_ */
