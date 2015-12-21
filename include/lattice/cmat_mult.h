/*
 * cmat_mult.h
 *
 *  Created on: Dec 16, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_CMAT_MULT_H_
#define INCLUDE_LATTICE_CMAT_MULT_H_

#include "constants.h"
#include <complex>
namespace MGGeometry {

/* Complex Matrix Multiply, hopefully vectorized
 * y is the output vector: of length 2N floats (N complexes)
 * x is the input  vector: of length 2N floats (N complexes)
 * A is the Small Dense Matrix: of size (2N*2N) floats (N*N) complexes
 * NB: precondition, N is minimally 8
 */


void CMatMult(float *y, const float *A,  const float *x, IndexType N);

void CMatMultNaive(std::complex<float>* y, const std::complex<float>* A, const std::complex<float>* x, IndexType N);


}



#endif /* INCLUDE_LATTICE_CMAT_MULT_H_ */
