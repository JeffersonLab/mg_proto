/*
 * qdpxx_helpers.cpp
 *
 *  Created on: Mar 17, 2016
 *      Author: bjoo
 */

#include "qdpxx_helpers.h"

#include <qdp.h>
using namespace QDP;

#include "lattice/constants.h"
using namespace MG;

namespace MGTesting {
	void initQDPXXLattice(const IndexArray& latdims )
	{
		multi1d<int> nrow(n_dim);
		for(int i=0; i < n_dim; ++i) nrow[i] = latdims[i];

		 Layout::setLattSize(nrow);
		 Layout::create();
	}

};
