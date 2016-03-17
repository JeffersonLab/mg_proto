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

#define SSE
#undef AVX2
#undef AVX

#ifdef  SSE
#define VECLEN 4  // SSE
#include <xmmintrin.h>
#include <pmmintrin.h>
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


void CMatMult(float *y, const float *A,  const float *x, const IndexType N, const int tid, const int nthreads);

/* Same as CMatMult, but passing in the min and max vrows
 * -- caller computes, and possibly stores in a ThreadInfo structure
 */

void CMatMultNaive(float* y, const float* A, const float* x, IndexType N);
void CMatMultNaiveAdd(float* y, const float* A, const float* x, IndexType N);




inline
void CMatMultVrow(float *y,
			  const float* A,
			  const float* x,
			  const IndexType N,
			  const int min_vrow,
			  const int max_vrow)
{

	const IndexType TwoN=2*N;

	for(int vrow=min_vrow*VECLEN; vrow < max_vrow*VECLEN; vrow++) {
			y[ vrow ] = 0;
	}

	for(IndexType col = 0; col < N; col++) {
		for(int vrow=min_vrow; vrow < max_vrow; vrow++) {
			int row = vrow*VECLEN2;


			float A_orig[VECLEN];
			float A_perm[VECLEN];
			for(int i=0; i < VECLEN; ++i) A_orig[i] = A[ TwoN*col + 2*row + i];
			for(int i=0; i < VECLEN/2; ++i) {
				A_perm[2*i] = -A_orig[2*i+1];
				A_perm[2*i+1] = A_orig[2*i];
			}

 			for(int i=0; i < VECLEN; ++i) {
 					y[2*row+i] += A_orig[i]*x[2*col] + A_perm[i]*x[2*col+1];
 			}

		}
	}
}

inline
void CMatMultVrowSMT(float *y,
			  	  	 const float* A,
					 const float* x,
					 const IndexType N,
					 const IndexType smt_id,
					 const IndexType n_smt,
					 const int N_vrows
			  )
{

	const IndexType TwoN=2*N;

	/* Initialize y */
	for(IndexType vrow=smt_id; vrow < N_vrows; vrow+=n_smt) {
		for(IndexType i=0; i < VECLEN; ++i) {
			y[vrow*VECLEN+i] = 0;
		}
	}


	for(IndexType col = 0; col < N; col++) {

		// thread 0 does = 0,2,4,... Max
		// thread 1 does = 1,3,....  Max
		for(int vrow=smt_id; vrow < N_vrows; vrow+=n_smt) {
			int row = vrow*VECLEN2;
			float A_orig[VECLEN];
			float A_perm[VECLEN];
			for(int i=0; i < VECLEN; ++i) A_orig[i] = A[ TwoN*col + 2*row + i];
			for(int i=0; i < VECLEN/2; ++i) {
				A_perm[2*i] = -A_orig[2*i+1];
				A_perm[2*i+1] = A_orig[2*i];
			}

			for(int i=0; i < VECLEN; ++i) {
				y[2*row+i] += A_orig[i]*x[2*col] + A_perm[i]*x[2*col+1];
	 		}

		}
	}

}

void AllocSpace(const int N);
void DestroySpace();


inline
void CMatMultVrowAdd(float *y,
			  const float* A,
			  const float* x,
			  const IndexType N,
			  const int min_vrow,
			  const int max_vrow)
{

	const IndexType TwoN=2*N;


	for(IndexType col = 0; col < N; col++) {

		// thread 0 does = 0..9
		// thread 1 does = 10-19
		for(int vrow=min_vrow; vrow < max_vrow; vrow++) {
			int row = vrow*VECLEN2;


			float A_orig[VECLEN];
			float A_perm[VECLEN];
			for(int i=0; i < VECLEN; ++i) A_orig[i] = A[ TwoN*col + 2*row + i];
			for(int i=0; i < VECLEN/2; ++i) {
				A_perm[2*i] = -A_orig[2*i+1];
				A_perm[2*i+1] = A_orig[2*i];
			}

 			for(int i=0; i < VECLEN; ++i) {
 					y[2*row+i] += A_orig[i]*x[2*col] + A_perm[i]*x[2*col+1];
 			}

		}
	}

}

inline
void CMatMultVrowAddMulti(float *y_vec[],
			  const float* A,
			  const float* x_vec[],
			  const IndexType N,
			  const IndexType smt_id,
			  const IndexType n_smt,
			  const IndexType n_src,
			  const IndexType min_vrow,
			  const IndexType max_vrow)
{

	const IndexType TwoN=2*N;
	for(IndexType col = 0; col < N; col++) {
		for(int vrow=min_vrow; vrow < max_vrow; vrow++) {
			int row = vrow*VECLEN2;


			float A_orig[VECLEN];
			float A_perm[VECLEN];
			for(int i=0; i < VECLEN; ++i) A_orig[i] = A[ TwoN*col + 2*row + i];
			for(int i=0; i < VECLEN/2; ++i) {
				A_perm[2*i] = -A_orig[2*i+1];
				A_perm[2*i+1] = A_orig[2*i];
			}

			// Each SMT thread does some number of sources.
			// Each source itself ought to be cacheline aligned, so we don't stomp on each other
			// With the SMTs
			for(int src=smt_id; src < n_src; src+=n_smt) {
				for(int i=0; i < VECLEN; ++i) {
 					y_vec[src][2*row+i] += A_orig[i]*x_vec[src][2*col] + A_perm[i]*x_vec[src][2*col+1];
				}
			}

		}
	}
}

inline
void CMatMultVrowAddSMT(float *y,
			  const float* A,
			  const float* x,
			  const IndexType N,
			  const IndexType smt_id,
			  const IndexType n_smt,
			  const IndexType N_vrows)
{

	const IndexType TwoN=2*N;


	for(IndexType col = 0; col < N; col++) {

		// thread 0 does = 0..9
		// thread 1 does = 10-19
		for(IndexType vrow=smt_id; vrow < N_vrows; vrow+=n_smt) {
			IndexType row = vrow*VECLEN2;

			float A_orig[VECLEN];
			float A_perm[VECLEN];
			for(int i=0; i < VECLEN; ++i) A_orig[i] = A[ TwoN*col + 2*row + i];
			for(int i=0; i < VECLEN/2; ++i) {
				A_perm[2*i] = -A_orig[2*i+1];
				A_perm[2*i+1] = A_orig[2*i];
			}

			// Each SMT thread does some number of sources.
			// Each source itself ought to be cacheline aligned, so we don't stomp on each other
			// With the SMTs

				for(int i=0; i < VECLEN; ++i) {
 					y[2*row+i] += A_orig[i]*x[2*col] + A_perm[i]*x[2*col+1];
				}



		}
	}

}



}



#endif /* INCLUDE_LATTICE_CMAT_MULT_H_ */
