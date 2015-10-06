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
    inline
    unsigned int NumNodes() const
    {
    	return _num_nodes;
    }


    inline
    unsigned int NodeID() const
    {
    	return _node_id;
    }

    inline
    const std::vector<unsigned int>& NodeDims() const
	{
    	return _node_dims;
	}

    inline
    const std::vector<unsigned int>& NodeCoords() const
    {
    	return _node_coords;
    }

    unsigned int NeighborNode(unsigned int dim, unsigned int dir) const
    {
    	return _neighbor_ids[dim][dir];
    }

  /* These are protected, so mock object can inherit */
  /* I don't like inheriting data, and this is for testing */
  /* Only the Mock Object should inherit.Nodeinfo may need
   * to be accessed a lot, so I don't want to pay Virtual Func overehad
   */
  protected:
    unsigned int _num_nodes; 	/*!< The number of nodes */
    unsigned int _node_id;       /*!< My own unique ID */
    std::vector<unsigned int> _node_dims; /*!< The dimensions of the Node Grid */
    std::vector<unsigned int> _node_coords; /*!< My own coordinates */
    unsigned int _neighbor_ids[n_dim][2]; /*!< The ID's of my neighbor nodes */
    
  };

};




#endif
