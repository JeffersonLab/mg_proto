/*
 * nodeinfo.cpp
 *
 *  Created on: Sep 25, 2015
 *      Author: bjoo
 */

#include "lattice/nodeinfo.h"

namespace MGGeometry {

	/*! Copy Constructor */
	inline
	NodeInfo::NodeInfo(const NodeInfo& i) {
		_num_nodes = i._num_nodes;
		_node_id = i._node_id;
		for(unsigned int mu=0; mu < n_dim; ++mu) {
			_node_dims[mu] = i._node_dims[mu];
			_node_coords[mu] = i._node_coords[mu];
			_neighbor_ids[mu][BACKWARD] = i._neighbor_ids[mu][BACKWARD];
			_neighbor_ids[mu][FORWARD] = i._neighbor_ids[mu][FORWARD];
		}
	}

	/*! Copy Assignment */
	inline
	NodeInfo& NodeInfo::operator=(const NodeInfo& i) {
		_num_nodes = i._num_nodes;
		_node_id = i._node_id;
		for(unsigned int mu=0; mu < n_dim; ++mu) {
			_node_dims[mu] = i._node_dims[mu];
			_node_coords[mu] = i._node_coords[mu];
			_neighbor_ids[mu][BACKWARD] = i._neighbor_ids[mu][BACKWARD];
			_neighbor_ids[mu][FORWARD] = i._neighbor_ids[mu][FORWARD];
		}

		return (*this);
	}

#if 0
	/*! Get The Number of Nodes */
	inline
	unsigned int NodeInfo::NumNodes(void) const {
		return _num_nodes;
	}

	/*! Get the ID of this Node */
	inline
    unsigned int NodeInfo::NodeID(void) const {
		return _node_id;
	}

	/*! Get the Dimensions of the processor grid */
	inline
	const std::vector<unsigned int>& NodeInfo::NodeDims() const {
		return _node_dims;
	}

	/*! Get the Coordinates of this node */
	const std::vector<unsigned int>& NodeInfo::NodeCoords() const {
		return _node_coords;
	}

	/*! Get the id of my neighbor
	 * \param dim   - the dimension in which the neighbor is sought
	 * \param dir   - the direction in which the neigbour is needed: BACKWARD or FORWARD
	 */
	unsigned int NodeInfo::NeighborNode(unsigned int dim, Direction dir) const {
		return _neighbor_ids[dim][dir];
	}
#endif

}



