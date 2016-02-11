/*
 * test_nodeinfo.cpp
 *
 *  Created on: Sep 25, 2015
 *      Author: bjoo
 */


#include "gtest/gtest.h"
#include "utils/print_utils.h"
#include <vector>
#include <iostream>
#include "MG_config.h"
#include "test_env.h"

#include "lattice/nodeinfo.h"
#include "lattice/constants.h"
#include "mock_nodeinfo.h"

#include <vector>

using namespace MG;
using namespace MG;


TEST(TestMockNodeinfo, MockNodeInfoCreate)
{
	IndexArray pe_dims={{1,2,4,4}};  // Pretend 32 nodes
	IndexArray pe_coords={{0,0,0,0}};
	MockNodeInfo mock_node(pe_dims, pe_coords);

	ASSERT_EQ( mock_node.NumNodes(), (IndexType)32);
	ASSERT_EQ( mock_node.NodeID(),(IndexType)0);

	const IndexArray& node_dims = mock_node.NodeDims();
	const IndexArray& node_coords = mock_node.NodeCoords();

	for(IndexType mu=0; mu < n_dim; ++mu ) {
		ASSERT_EQ( pe_dims[mu], node_dims[mu]);
		ASSERT_EQ( pe_coords[mu], node_coords[mu]);
	}
}

int main(int argc, char *argv[])
{
	return MGTesting::TestMain(&argc, argv);
}
