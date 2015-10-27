/*
 * mock_nodeinfo.h
 *
 *  Created on: Oct 2, 2015
 *      Author: bjoo
 */

#ifndef TEST_MOCK_NODEINFO_H_
#define TEST_MOCK_NODEINFO_H_

#include "lattice/constants.h"
#include "lattice/nodeinfo.h"
#include "utils/print_utils.h"


namespace MGGeometry {

	class MockNodeInfo : public NodeInfo {
	public:
		MockNodeInfo(const IndexArray& pe_dims,
				     const IndexArray& pe_coords) {

			for(IndexType mu=0; mu < n_dim; ++mu) {
				_node_dims[mu]=pe_dims[mu];
				_node_coords[mu]=pe_coords[mu];
				if( _node_coords[mu] >= _node_dims[mu] ) {
					MGUtils::MasterLog(MGUtils::ERROR,
								"Node Coordinate has to be less than or equal to node dimension.");
				}
			}

			_num_nodes=_node_dims[0];
			for(IndexType mu=1; mu < n_dim; ++mu ) {
				_num_nodes *= _node_dims[mu];
			}

			_node_id = _node_coords[0] +
						_node_dims[0]*(_node_coords[1] +
						  _node_dims[1]*(_node_coords[2] +
		                     _node_dims[2]*_node_coords[3]
							            )
						   			);

			IndexArray fwd_neighbor(_node_coords);
			IndexArray bwd_neighbor(_node_coords);

			for(IndexType mu=0; mu < n_dim; ++mu) {
				// Get fwd and backward neighbours with wraparound
				fwd_neighbor[mu] =
						(_node_coords[mu] == _node_dims[mu]-1) ? 0 : _node_coords[mu]+1;
				bwd_neighbor[mu] =
						(_node_coords[mu] == 0 ) ? _node_dims[mu]-1 : _node_coords[mu]-1;

				IndexType fwd_nodeid = fwd_neighbor[0] +
						_node_dims[0]*(fwd_neighbor[1] +
						  _node_dims[1]*(fwd_neighbor[2] +
		                     _node_dims[2]*fwd_neighbor[3]
							            )
						   			);

				IndexType bwd_nodeid = bwd_neighbor[0] +
						_node_dims[0]*(bwd_neighbor[1] +
						  _node_dims[1]*(bwd_neighbor[2] +
		                     _node_dims[2]*bwd_neighbor[3]
							            )
						   			);

				_neighbor_ids[mu][BACKWARD] = bwd_nodeid;
				_neighbor_ids[mu][FORWARD]  = fwd_nodeid;

			}	// for int mu
		}  // Mock Node Info constructor

	};

}



#endif /* TEST_MOCK_NODEINFO_H_ */
