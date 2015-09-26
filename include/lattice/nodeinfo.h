#ifndef NODEINFO_H
#define NODEINFO_H

#include "lattice/constants.h"
#include <vector>

namespace MGGeometry {

  // Place holder
  class NodeInfo {
  public: 
	/* Basic Constructor... Should fill out all the info
	 * Either from Single Node or QMP or something
	 * Locate this in the C-file for implementation
	 */
    NodeInfo();     // Construct
    ~NodeInfo() {};    // Destruct

    /* Copy */
    NodeInfo(const NodeInfo& i); // Copy
    NodeInfo& operator=(const NodeInfo& i); // Copy Assignment


    /* Public methods */
    int NumNodes() const;
    int NodeID() const;

    const std::vector<int>& NodeDims() const;
    const std::vector<int>& NodeCoords() const;
    int NeighborNode(int dim, Direction dir) const;

  private:
    int _num_nodes; 	/*!< The number of nodes */
    int _node_id;       /*!< My own unique ID */
    std::vector<int> _node_dims; /*!< The dimensions of the Node Grid */
    std::vector<int> _node_coords; /*!< My own coordinates */
    int _neighbor_ids[n_dim][2]; /*!< The ID's of my neighbor nodes */
    
  };

};




#endif
