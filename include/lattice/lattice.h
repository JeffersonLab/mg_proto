#ifndef LATTICE_H
#define LATTICE_H

#include <vector>

#include "lattice/constants.h"  // n_dim and friends

namespace MGGeometry { 

  class NodeInfo; // Forward declare this

  class Lattice {
  public:
   
    /** Constructor
     * \param lat_dims is a vector containing the dimensions of the lattice  in sites
     * \param n_spin is the number of spin components
     * \param n_colo is the number of color components
     * \param node_info has details about the node (to help work out origins
     *                              checkerboards etc.
     */
    Lattice(const std::vector<int>& lat_dims,
	       const int n_spin,
	    const int n_color, const NodeInfo& node);

    ~Lattice();
 
    
    
  private:
    int _lat_dims[n_dim];       // The lattice dimensions (COPIED In)
    int _n_sites;                                // The total number of sites
    int _n_color;                              // The number of colors
    int _n_spin;                               // The number of spins
    int _n_cb_sites[2];                      // The number of sites of each checkerboard
    std::vector<int> _cb_sites[2];   // The site tables for each checkerboard

    std::vector<int> _n_surface_sites[2];  // The number of surface sites for each CB.
    std::vector<int> _sufrace_sites[2][n_dim]; 
  };

}



#endif
