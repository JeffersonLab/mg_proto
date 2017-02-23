#ifndef TEST_QDPXX_DSLASH_M_W_H_
#define TEST_QDPXX_DSLASH_M_W_H_

#ifndef QDP_INCLUDE
#include "qdp.h"
#endif

using namespace QDP;

namespace MGTesting {

void dslash(LatticeFermionF& chi, 
	    const multi1d<LatticeColorMatrixF>& u, 
	    const LatticeFermionF& psi,
	    int isign, int cb);


void dslash(LatticeFermionD& chi, 
	    const multi1d<LatticeColorMatrixD>& u, 
	    const LatticeFermionD& psi,
	    int isign, int cb);
};

#endif /* TEST_QDPXX_DSLASH_M_W_H_ */
