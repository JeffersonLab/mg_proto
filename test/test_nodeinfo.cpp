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
#include "qmp_test_env.h"

#include "lattice/nodeinfo.h"

using namespace MGUtils;
using namespace MGGeometry;


int main(int argc, char *argv[])
{
	return MGTesting::TestMain(&argc, argv);
}
