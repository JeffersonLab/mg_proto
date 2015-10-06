#ifndef LATTICE_H
#define LATTICE_H

#include <vector>

#include "lattice/constants.h"  // n_dim and friends

namespace MGGeometry {

class NodeInfo;
// Forward declare this

class LatticeInfo {
public:

	/** Constructor
	 * \param lat_dims is a vector containing the dimensions of the lattice  in sites
	 * \param n_spin is the number of spin components
	 * \param n_colo is the number of color components
	 * \param node_info has details about the node (to help work out origins
	 *                              checkerboards etc.
	 */
	LatticeInfo(const std::vector<unsigned int>& lat_dims, unsigned int n_spin,
			unsigned int n_color, const NodeInfo& node);

	~LatticeInfo();

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
	const unsigned int _n_color;                         // The number of colors
	const unsigned int _n_spin;                           // The number of spins

	const NodeInfo& _node_info;			   	   // The Node Info
	unsigned int _lat_dims[n_dim];         // The lattice dimensions (COPIED In)
	unsigned int _n_sites;                          // The total number of sites
	unsigned int _lat_origin[n_dim];
	unsigned int _sum_orig_coords;

	unsigned int _n_cb_sites[2];     // The number of sites of each checkerboard
	std::vector<unsigned int> _cb_sites[2]; // The site tables for each checkerboard

	std::vector<unsigned int> _surface_sites[n_dim][2][2];
};

}

#endif
