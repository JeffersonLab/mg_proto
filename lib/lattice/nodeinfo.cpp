/*
 * nodeinfo.cpp
 *
 *  Created on: Sep 25, 2015
 *      Author: bjoo
 */

#include "lattice/nodeinfo.h"

namespace MGGeometry {

	/*! Copy Constructor */
	NodeInfo::NodeInfo(const NodeInfo& i) : _num_nodes{i._num_nodes},
			_node_id{i._node_id}, _node_dims{i._node_dims}, _node_coords{i._node_coords} {

		for(IndexType mu=0; mu < n_dim; ++mu) {
			_neighbor_ids[mu][BACKWARD] = i._neighbor_ids[mu][BACKWARD];
			_neighbor_ids[mu][FORWARD] = i._neighbor_ids[mu][FORWARD];
		}
	}

	/*! Copy Assignment */
	NodeInfo& NodeInfo::operator=(const NodeInfo& i)  {
		_num_nodes = i._num_nodes;
		_node_id = i._node_id;
		_node_dims= i._node_dims;
		_node_coords= i._node_coords;

		for(IndexType mu=0; mu < n_dim; ++mu) {
			_neighbor_ids[mu][BACKWARD] = i._neighbor_ids[mu][BACKWARD];
			_neighbor_ids[mu][FORWARD] = i._neighbor_ids[mu][FORWARD];
		}

		return (*this);
	}



#

}



