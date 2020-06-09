/*
 * cmat_mult.h
 *
 *  Created on: Dec 16, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_CMAT_MULT_H_
#define INCLUDE_LATTICE_CMAT_MULT_H_
#include "MG_config.h"
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

	typedef int LAPACK_BLASINT;
	void XGEMM(const char *transa, const char *transb, LAPACK_BLASINT m,
			LAPACK_BLASINT n, LAPACK_BLASINT k, std::complex<float> alpha,
			const std::complex<float> *a, LAPACK_BLASINT lda, const std::complex<float> *b,
			LAPACK_BLASINT ldb, std::complex<float> beta, std::complex<float> *c,
			LAPACK_BLASINT ldc);

/* Same as CMatMult, but passing in the min and max vrows
 * -- caller computes, and possibly stores in a ThreadInfo structure
 */

/* y = A x */
void CMatMultNaive(float* y, const float* A, const float* x, IndexType N, IndexType ncol=1);



/* y = A^\dagger x */
void CMatAdjMultNaive(float *y, const float *A, const float* x, IndexType N, IndexType ncol=1);


/* y += A x */
void CMatMultAddNaive(float* y, const float* A, const float* x, IndexType N, IndexType ncol=1);



/* y = alpha A x + beta*y,  alpha and beta are real */
void CMatMultCoeffAddNaive(float beta, float* y,  float alpha, const float* A, const float* x, IndexType N, IndexType ncol=1);

void CMatMultCoeffAddNaive(float beta,
		std::complex<float>* y,
		IndexType ldy,
		float alpha,
		const std::complex<float>* A,
		IndexType ldA,
		const std::complex<float>* x,
		IndexType ldx,
		IndexType Arows,
		IndexType Acols,
		IndexType xcols);

void CMatAdjMultCoeffAddNaive(float beta,
		std::complex<float>* y,
		IndexType ldy,
		float alpha,
		const std::complex<float>* A,
		IndexType ldA,
		const std::complex<float>* x,
		IndexType ldx,
		IndexType Arows,
		IndexType Acols,
		IndexType xcols);

void GcCMatMultGcNaive(float* y,
				   const float* A,
				   const float* x,
				   IndexType N,
					IndexType ncol=1);


void GcCMatMultGcCoeffAddNaive(float beta, float* y, float alpha,
				   const float* A,
				   const float* x,
				   IndexType N,
					IndexType ncol=1);
}



#endif /* INCLUDE_LATTICE_CMAT_MULT_H_ */
