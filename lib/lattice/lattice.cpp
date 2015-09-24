

#include "lattice/lattice.h"
#include "lattice/nodeinfo.h"

#include "utils/print_utils.h"

using namespace MGUtils;

namespace MGGeometry { 

Lattice::Lattice(const std::vector<int>& lat_dims,
		 const int n_spin,
		 const int n_color,
		 const NodeInfo& node) : _n_color(n_color), _n_spin(n_spin)
{
	/* Copy in the Lattice Dimensions */
	if( lat_dims.size() == n_dim ) {
		for(int mu=0; mu < n_dim; ++mu) {
			_lat_dims[mu] = lat_dims[mu];
		}
	}
	else {
		MasterLog(ERROR, "lat_dims size is different from n_dim");
		/* This should abort */
	}

	/* Count the number of sites */
	_n_sites = _lat_dims[0];
	for(int mu=1; mu < n_dim; +	+mu) {
		_n_sites *= _lat_dims[mu];
	}

	_n_cb_sites[0] = 0;
	_n_cb_sites[1] = 0;

	/* Now I need info from NodeInfo, primarily my node coordinates
	 * so that I can use the local lattice dims and the node coords
	 * to find out the checkerboard of my origin, and/or to find my
	 * global coordinates
	 */

}

/** Destructor for the lattice class
 */
Lattice::~Lattice() {}


};
