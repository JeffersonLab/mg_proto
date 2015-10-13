#ifndef LATTICE_H
#define LATTICE_H

#include <vector>

#include "lattice/constants.h"  // n_dim and friends
#include "lattice/nodeinfo.h"

#include "utils/print_utils.h"

using namespace MGUtils;

namespace MGGeometry {



class LatticeInfo {
public:

	/** Most General Constructor
	 *  \param origin   is a vector containing the coordinates of the origin of the Lattice Block
	 *  \param lat_dims is a vector containing the dimensions of the lattice block
	 *  \param n_spin   is the number of spin components of the lattice block
	 *  \param n_color  is the number of color components of the lattice block
	 *  \param node     is the NodeInfo() object for the current node.
	 */
	LatticeInfo(const std::vector<unsigned int>& lat_origin,
				const std::vector<unsigned int>& lat_dims,
				const unsigned int n_spin,
				const unsigned int n_color,
				const NodeInfo& node);

	/** DelegatingConstructor -- for when there is only one lattice block per node. Local origin assumed
	 *   to be ( lat_dims[0]*node_coord[0], lat_dims[1]*node_coord[1], lat_dims[2]*node_coord[2], lat_dims[3]*node_coord[3] )
	 *
	 * \param lat_dims is a vector containing the dimensions of the lattice  in sites
	 * \param n_spin is the number of spin components
	 * \param n_colo is the number of color components
	 * \param node_info has details about the node (to help work out origins
	 *                              checkerboards etc.
	 */
	LatticeInfo(const std::vector<unsigned int>& lat_dims,
				const unsigned int n_spin,
				const unsigned int n_color,
				const NodeInfo& node);


	/** Delegating Constructor
	 *  Local origin assumed to be ( lat_dims[0]*node_coord[0], lat_dims[1]*node_coord[1], lat_dims[2]*node_coord[2], lat_dims[3]*node_coord[3] )
	 *  n_spin = 4, n_color = 3, NodeInfo instantiated on demand
	 */
	LatticeInfo(const std::vector<unsigned int>& lat_dims);

	~LatticeInfo();

	inline
	const std::vector<unsigned int>& GetLatticeDimensions() const {
		return _lat_dims;
	}

	inline
	unsigned int GetNumColors() const {
		return _n_color;
	}

	inline
	unsigned int GetNumSpins() const {
		return _n_spin;
	}

	inline
	unsigned int GetNumCBSites(unsigned int cb) const {
		return _n_cb_sites[cb];
	}

	inline
	unsigned int GetNumSites() const {
		return _n_sites;
	}

	inline const std::vector<unsigned int>& GetCBSiteTable(
			unsigned int cb) const {
		return _cb_sites[cb];
	}

	inline const std::vector<unsigned int>& GetCBSurfaceSiteTable(
			unsigned int dim, unsigned int fb, unsigned int cb) const {

		return _surface_sites[dim][static_cast<unsigned int>(fb)][cb];

	}

	inline
	unsigned int GetNumCBSurfaceSites(unsigned int dim, unsigned int fb,
			unsigned int cb) const {
		return _surface_sites[dim][static_cast<unsigned int>(fb)][cb].size();
	}
private:
	std::vector<unsigned int> _lat_origin;
	std::vector<unsigned int> _lat_dims;         // The lattice dimensions (COPIED In)
	unsigned int _n_color;
	unsigned int _n_spin;
	const NodeInfo& _node_info;			   	   // The Node Info -- copied in


	unsigned int _n_sites;                          // The total number of sites
	unsigned int _sum_orig_coords;

	unsigned int _n_cb_sites[2];     // The number of sites of each checkerboard
	std::vector<unsigned int> _cb_sites[2]; // The site tables for each checkerboard

	std::vector<unsigned int> _surface_sites[n_dim][2][2];

	/* Compute Origin from NodeInfo and NodeCoords */
	std::vector<unsigned int>
	ComputeOriginCoords(const std::vector<unsigned int>& lat_dims, const NodeInfo& node_info) {
		std::vector<unsigned int> origin_coords;
		const std::vector<unsigned int>& node_coords = node_info.NodeCoords();
		for(unsigned int mu=0; mu < n_dim; ++mu) {
		   origin_coords.push_back(lat_dims[mu]*node_coords[mu]);
		}

		return origin_coords;
	}
};

#if 0
class Lattice {
public:
	Lattice(const std::vector<unsigned int>& lat_dims,
			unsigned int n_spin,
			unsigned int n_color,
			NodeInfo& node);

	inline
	unsigned int GetNumBlocks() const {
		return _num_blocks;
	}

	inline const std::vector<LatticeInfo>& GetCBBlockTable(unsigned int cb) const {
		return _cb_blocks[cb];
	}

	inline
	unsigned int GetNumSpin() const {
		return _n_spin;
	}

	inline
	unsigned int GetNumColor() const {
		return _n_color;
	}


private:
	unsigned int _n_spin;
	unsigned int _n_color;
	unsigned int _num_blocks;
	unsigned int _cb_num_blocks[2];
	std::vector<LatticeInfo> _cb_blocks[2];

};
#endif

}

#endif
