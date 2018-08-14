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

#undef SSE
#undef AVX2
#undef AVX

#ifdef  SSE
#define VECLEN 4  // SSE
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#endif

#ifdef  AVX2
#define VECLEN 8 	// AVX2
#define AVX2_FMA
#undef AVX2_FMA_ADDSUB
#include <immintrin.h>
#endif

#ifdef AVX
#define VECLEN	8
#include <immintrin.h>
#endif

#define VECLEN 4
#define VECLEN2 (VECLEN/2)



namespace MG {

inline int MinInt(const int& a, const int& b)
{
	return (a < b) ? a:b;
}
/* Complex Matrix Multiply, hopefully vectorized
 * y is the output vector: of length 2N floats (N complexes)
 * x is the input  vector: of length 2N floats (N complexes)
 * A is the Small Dense Matrix: of size (2N*2N) floats (N*N) complexes
 * NB: precondition, N is minimally 8
 */


/* Same as CMatMult, but passing in the min and max vrows
 * -- caller computes, and possibly stores in a ThreadInfo structure
 */

/* y = A x */
void CMatMultNaive(float* y, const float* A, const float* x, IndexType N);

/* y = A^\dagger x */
void CMatAdjMultNaive(float *y, const float *A, const float* x, IndexType N);


/* y += A x */
void CMatMultNaiveAdd(float* y, const float* A, const float* x, IndexType N);

/* y += A^\dagger x */
void CMatAdjMultNaiveAdd(float *y, const float *A, const float *x, IndexType N);

/* y += alpha A x,  alpha is real */
void CMatMultNaiveCoeffAdd(float* y, const float alpha, const float* A, const float* x, IndexType N);

/* y += alpha A^\dagger x, alpha is real */
void CMatAdjMultNaiveCoeffAdd(float *y, const float alpha, const float *A, const float *x, IndexType N);

}



#endif /* INCLUDE_LATTICE_CMAT_MULT_H_ */
