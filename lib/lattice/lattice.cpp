

#include "lattice/lattice.h"
#include "lattice/nodeinfo.h"

namespace MGGeometry { 

Lattice::Lattice(const std::vector<int>& lat_dims,
		 const int n_spin,
		 const int n_color,
		 const NodeInfo& node) : _n_color(n_color), _n_spin(n_spin)
{
  /* Init code goes here */
}

/** Destructor for the lattice class
 */
Lattice::~Lattice() {}


};
