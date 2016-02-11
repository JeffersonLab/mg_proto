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

#if 0
void CMatMultVrow(float *y,
			  const float* A,
			  const float* x,
			  const IndexType N,
			  const int min_vrow,
			  const int max_vrow);
#endif

void CMatMultNaive(std::complex<float>* y, const std::complex<float>* A, const std::complex<float>* x, IndexType N);


inline
void CMatMultVrow(float *y,
			  const float* A,
			  const float* x,
			  const IndexType N,
			  const int min_vrow,
			  const int max_vrow)
{

	const IndexType TwoN=2*N;

#if defined(AVX2_FMA) || defined(AVX)
	const __m256 signs=_mm256_set_ps(1,-1,1,-1,1,-1,1,-1);
#endif

	/* Initialize y */
	for(IndexType vrow=min_vrow; vrow < max_vrow; ++vrow) {

#pragma omp simd aligned(y:16)
		for(IndexType i=0; i < VECLEN; ++i) {
			y[vrow*VECLEN+i] = 0;
		}
	}

	for(IndexType col = 0; col < N; col++) {

		// thread 0 does = 0..9
		// thread 1 does = 10-19
		for(int vrow=min_vrow; vrow < max_vrow; vrow++) {
			int row = vrow*VECLEN2;
#ifdef SSE
			__m128 xr,xi, A_orig, A_perm;
			__m128 y1, y2;

			// NB: Single instruction broadcast...
			xr = _mm_load1_ps(&x[2*col]); // (unaligned load Broadcast...)
			xi = _mm_load1_ps(&x[2*col+1]); // Broadcast
			// Load VECLEN2 rows of A (A is row major so this is a simple thing)
			A_orig = _mm_load_ps(&A[ TwoN*col + 2*row] );
			A_perm = _mm_shuffle_ps( A_orig, A_orig, _MM_SHUFFLE(2,3,0,1));

			// Do the maths.. Load in rows of result
			__m128 yv = _mm_load_ps(&y[2*row]);

			// 2 FMAs one with addsub
			y1 = _mm_mul_ps(A_orig,xr);
			yv = _mm_add_ps(yv,y1);
			y2 = _mm_mul_ps(A_perm,xi);
			yv = _mm_addsub_ps(yv, y2);

			// Store
			_mm_store_ps(&y[2*row],yv);
#endif

#ifdef AVX
			// Use sign array
			__m256 xr,xi, A_orig, A_perm;
			__m256 y1,y2;

			__m256 yv = _mm256_load_ps(&y[2*row]);
			xr = _mm256_broadcast_ss(&x[2*col]);
			xi = _mm256_broadcast_ss(&x[2*col+1]);
			A_orig = _mm256_load_ps( &A[ TwoN*col + 2*row] );

			// In lane shuffle. Never cross 128bit lanes, only shuffle
			// Real Imag parts of a lane. This is like two separate SSE
			// Shuffles, hence the use of a single _MM_SHUFFLE() Macro
			A_perm = _mm256_shuffle_ps(A_orig,A_orig, _MM_SHUFFLE(2,3,0,1));
			y1 = _mm256_mul_ps(A_orig,xr);
			yv = _mm256_add_ps(yv,y1);
			y2 = _mm256_mul_ps(A_perm, xi);
			yv = _mm256_addsub_ps(yv,y2);

			_mm256_store_ps(&y[2*row],yv);
#endif


#ifdef AVX2

#ifdef AVX2_FMA_ADDSUB
			// Use addsub
			__m256 xr,xi, A_orig, A_perm;
			__m256 y1,y2;

			xr = _mm256_broadcast_ss(&x[2*col]);
			xi = _mm256_broadcast_ss(&x[2*col+1]);
			A_orig = _mm256_load_ps( &A[ TwoN*col + 2*row] );
			// In lane shuffle. Never cross 128bit lanes, only shuffle
			// Real Imag parts of a lane. This is like two separate SSE
			// Shuffles, hence the use of a single _MM_SHUFFLE() Macro
			A_perm = _mm256_shuffle_ps(A_orig,A_orig, _MM_SHUFFLE(2,3,0,1));
			__m256 yv = _mm256_load_ps(&y[2*row]);
			__m256 tmp = _mm256_mul_ps(A_perm,xi);
			yv = _mm256_fmadd_ps(A_orig,xr,yv);
			yv = _mm256_addsub_ps(yv,tmp);
			_mm256_store_ps(&y[2*row],yv);
#endif

#ifdef AVX2_FMA
			// Use sign array
			__m256 xr,xi, A_orig, A_perm;
			__m256 y1,y2;

			__m256 yv = _mm256_load_ps(&y[2*row]);
			xr = _mm256_broadcast_ss(&x[2*col]);
			xi = _mm256_broadcast_ss(&x[2*col+1]);
			A_orig = _mm256_load_ps( &A[ TwoN*col + 2*row] );

			// In lane shuffle. Never cross 128bit lanes, only shuffle
			// Real Imag parts of a lane. This is like two separate SSE
			// Shuffles, hence the use of a single _MM_SHUFFLE() Macro
			A_perm = _mm256_shuffle_ps(A_orig,A_orig, _MM_SHUFFLE(2,3,0,1));
			__m256 tmp = _mm256_mul_ps(A_perm, xi);
			yv = _mm256_fmadd_ps(A_orig,xr,yv);

			// Instead of addsub, I am multiplying
			// signs into tmp, to use 2 FMAs. This appears to
			// be faster: 19.1GF vs 16.8GF at N=40
			yv = _mm256_fmadd_ps(signs,tmp,yv);

			_mm256_store_ps(&y[2*row],yv);
#endif // AVX2 FMA
#endif // AVX2

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

#if defined(AVX2_FMA) || defined(AVX)
	const __m256 signs=_mm256_set_ps(1,-1,1,-1,1,-1,1,-1);
#endif

	/* Initialize y */
	for(IndexType vrow=smt_id; vrow < N_vrows; vrow+=n_smt) {

#pragma omp simd aligned(y:16)
		for(IndexType i=0; i < VECLEN; ++i) {
			y[vrow*VECLEN+i] = 0;
		}
	}

	for(IndexType col = 0; col < N; col++) {

		// thread 0 does = 0,2,4,... Max
		// thread 1 does = 1,3,....  Max
		for(int vrow=smt_id; vrow < N_vrows; vrow+=n_smt) {
			int row = vrow*VECLEN2;
#ifdef SSE
			__m128 xr,xi, A_orig, A_perm;
			__m128 y1, y2;

			// NB: Single instruction broadcast...
			xr = _mm_load1_ps(&x[2*col]); // (unaligned load Broadcast...)
			xi = _mm_load1_ps(&x[2*col+1]); // Broadcast
			// Load VECLEN2 rows of A (A is row major so this is a simple thing)
			A_orig = _mm_load_ps(&A[ TwoN*col + 2*row] );
			A_perm = _mm_shuffle_ps( A_orig, A_orig, _MM_SHUFFLE(2,3,0,1));

			// Do the maths.. Load in rows of result
			__m128 yv = _mm_load_ps(&y[2*row]);

			// 2 FMAs one with addsub
			y1 = _mm_mul_ps(A_orig,xr);
			yv = _mm_add_ps(yv,y1);
			y2 = _mm_mul_ps(A_perm,xi);
			yv = _mm_addsub_ps(yv, y2);

			// Store
			_mm_store_ps(&y[2*row],yv);
#endif

#ifdef AVX
			// Use sign array
			__m256 xr,xi, A_orig, A_perm;
			__m256 y1,y2,y3;

			__m256 yv = _mm256_load_ps(&y[2*row]);
			xr = _mm256_broadcast_ss(&x[2*col]);
			xi = _mm256_broadcast_ss(&x[2*col+1]);
			A_orig = _mm256_load_ps( &A[ TwoN*col + 2*row] );

			// In lane shuffle. Never cross 128bit lanes, only shuffle
			// Real Imag parts of a lane. This is like two separate SSE
			// Shuffles, hence the use of a single _MM_SHUFFLE() Macro
			A_perm = _mm256_shuffle_ps(A_orig,A_orig, _MM_SHUFFLE(2,3,0,1));
			y1 = _mm256_mul_ps(A_orig,xr);
			yv = _mm256_add_ps(yv,y1);
			y2 = _mm256_mul_ps(A_perm, xi);
			y3 = _mm256_mul_ps(signs,y2);
			yv = _mm256_add_ps(yv,y3);

			_mm256_store_ps(&y[2*row],yv);
#endif


#ifdef AVX2

#ifdef AVX2_FMA_ADDSUB
			// Use addsub
			__m256 xr,xi, A_orig, A_perm;
			__m256 y1,y2;

			xr = _mm256_broadcast_ss(&x[2*col]);
			xi = _mm256_broadcast_ss(&x[2*col+1]);
			A_orig = _mm256_load_ps( &A[ TwoN*col + 2*row] );
			// In lane shuffle. Never cross 128bit lanes, only shuffle
			// Real Imag parts of a lane. This is like two separate SSE
			// Shuffles, hence the use of a single _MM_SHUFFLE() Macro
			A_perm = _mm256_shuffle_ps(A_orig,A_orig, _MM_SHUFFLE(2,3,0,1));
			__m256 yv = _mm256_load_ps(&y[2*row]);
			__m256 tmp = _mm256_mul_ps(A_perm,xi);
			yv = _mm256_fmadd_ps(A_orig,xr,yv);
			yv = _mm256_addsub_ps(yv,tmp);
			_mm256_store_ps(&y[2*row],yv);
#endif

#ifdef AVX2_FMA
			// Use sign array
			__m256 xr,xi, A_orig, A_perm;
			__m256 y1,y2;

			__m256 yv = _mm256_load_ps(&y[2*row]);
			xr = _mm256_broadcast_ss(&x[2*col]);
			xi = _mm256_broadcast_ss(&x[2*col+1]);
			A_orig = _mm256_load_ps( &A[ TwoN*col + 2*row] );

			// In lane shuffle. Never cross 128bit lanes, only shuffle
			// Real Imag parts of a lane. This is like two separate SSE
			// Shuffles, hence the use of a single _MM_SHUFFLE() Macro
			A_perm = _mm256_shuffle_ps(A_orig,A_orig, _MM_SHUFFLE(2,3,0,1));
			__m256 tmp = _mm256_mul_ps(A_perm, xi);
			yv = _mm256_fmadd_ps(A_orig,xr,yv);

			// Instead of addsub, I am multiplying
			// signs into tmp, to use 2 FMAs. This appears to
			// be faster: 19.1GF vs 16.8GF at N=40
			yv = _mm256_fmadd_ps(signs,tmp,yv);

			_mm256_store_ps(&y[2*row],yv);
#endif // AVX2 FMA
#endif // AVX2

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

#if defined(AVX2_FMA)
	const __m256 signs=_mm256_set_ps(1,-1,1,-1,1,-1,1,-1);
#endif

#pragma unroll_and_jam
	for(IndexType col = 0; col < N; col++) {

		// thread 0 does = 0..9
		// thread 1 does = 10-19
#pragma unroll(4)
		for(int vrow=min_vrow; vrow < max_vrow; vrow++) {
			int row = vrow*VECLEN2;
#ifdef SSE
			__m128 xr,xi, A_orig, A_perm;
			__m128 y1, y2;

			// NB: Single instruction broadcast...
			xr = _mm_load1_ps(&x[2*col]); // (unaligned load Broadcast...)
			xi = _mm_load1_ps(&x[2*col+1]); // Broadcast
			// Load VECLEN2 rows of A (A is row major so this is a simple thing)
			A_orig = _mm_load_ps(&A[ TwoN*col + 2*row] );
			A_perm = _mm_shuffle_ps( A_orig, A_orig, _MM_SHUFFLE(2,3,0,1));

			// Do the maths.. Load in rows of result
			__m128 yv = _mm_load_ps(&y[2*row]);

			// 2 FMAs one with addsub
			y1 = _mm_mul_ps(A_orig,xr);
			yv = _mm_add_ps(yv,y1);
			y2 = _mm_mul_ps(A_perm,xi);
			yv = _mm_addsub_ps(yv, y2);

			// Store
			_mm_store_ps(&y[2*row],yv);
#endif

#ifdef AVX
			// Use sign array
			__m256 xr,xi, A_orig, A_perm;
			__m256 y1,y2,y3;
			A_orig = _mm256_load_ps( &A[ TwoN*col + 2*row] );
			__m256 yv = _mm256_load_ps(&y[2*row]);
			xr = _mm256_broadcast_ss(&x[2*col]);
			xi = _mm256_broadcast_ss(&x[2*col+1]);


			// In lane shuffle. Never cross 128bit lanes, only shuffle
			// Real Imag parts of a lane. This is like two separate SSE
			// Shuffles, hence the use of a single _MM_SHUFFLE() Macro
			A_perm = _mm256_shuffle_ps(A_orig,A_orig, _MM_SHUFFLE(2,3,0,1));
			y1 = _mm256_mul_ps(A_orig,xr);
			yv = _mm256_add_ps(yv,y1);
			y2 = _mm256_mul_ps(A_perm, xi);
			yv = _mm256_addsub_ps(yv,y2);

			_mm256_store_ps(&y[2*row],yv);
#endif


#ifdef AVX2

#ifdef AVX2_FMA_ADDSUB
			// Use addsub
			__m256 xr,xi, A_orig, A_perm;
			__m256 y1,y2;

			xr = _mm256_broadcast_ss(&x[2*col]);
			xi = _mm256_broadcast_ss(&x[2*col+1]);
			A_orig = _mm256_load_ps( &A[ TwoN*col + 2*row] );
			// In lane shuffle. Never cross 128bit lanes, only shuffle
			// Real Imag parts of a lane. This is like two separate SSE
			// Shuffles, hence the use of a single _MM_SHUFFLE() Macro
			A_perm = _mm256_shuffle_ps(A_orig,A_orig, _MM_SHUFFLE(2,3,0,1));
			__m256 yv = _mm256_load_ps(&y[2*row]);
			__m256 tmp = _mm256_mul_ps(A_perm,xi);
			yv = _mm256_fmadd_ps(A_orig,xr,yv);
			yv = _mm256_addsub_ps(yv,tmp);
			_mm256_store_ps(&y[2*row],yv);
#endif

#ifdef AVX2_FMA
			// Use sign array
			__m256 xr,xi, A_orig, A_perm;
			__m256 y1,y2;

			__m256 yv = _mm256_load_ps(&y[2*row]);
			xr = _mm256_broadcast_ss(&x[2*col]);
			xi = _mm256_broadcast_ss(&x[2*col+1]);
			A_orig = _mm256_load_ps( &A[ TwoN*col + 2*row] );

			// In lane shuffle. Never cross 128bit lanes, only shuffle
			// Real Imag parts of a lane. This is like two separate SSE
			// Shuffles, hence the use of a single _MM_SHUFFLE() Macro
			A_perm = _mm256_shuffle_ps(A_orig,A_orig, _MM_SHUFFLE(2,3,0,1));
			__m256 tmp = _mm256_mul_ps(A_perm, xi);
			yv = _mm256_fmadd_ps(A_orig,xr,yv);

			// Instead of addsub, I am multiplying
			// signs into tmp, to use 2 FMAs. This appears to
			// be faster: 19.1GF vs 16.8GF at N=40
			yv = _mm256_fmadd_ps(signs,tmp,yv);

			_mm256_store_ps(&y[2*row],yv);
#endif // AVX2 FMA
#endif // AVX2

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

#if defined(AVX2_FMA)
	const __m256 signs=_mm256_set_ps(1,-1,1,-1,1,-1,1,-1);
#endif

	for(int j=smt_id; j < n_src; j+=n_smt ) {
		float* y = y_vec[j];
		const float *x = x_vec[j];

		if( j+n_smt < n_src ) {
			_mm_prefetch((const char *)(y_vec[j+n_smt]),_MM_HINT_T0);
			_mm_prefetch((const char *)(x_vec[j+n_smt]),_MM_HINT_T0);
		}

#pragma unroll_and_jam
	for(IndexType col = 0; col < N; col++) {

			// thread 0 does = 0..9
			// thread 1 does = 10-19
#pragma unroll(4)
			for(int vrow=min_vrow; vrow < max_vrow; vrow++) {
				int row = vrow*VECLEN2;


				// Use sign array
				__m256 xr, xi, A_orig, A_perm;
				__m256 y1, y2;

				__m256 yv = _mm256_load_ps(&y[2 * row]);
				xr = _mm256_broadcast_ss(&x[2 * col]);
				xi = _mm256_broadcast_ss(&x[2 * col + 1]);
				A_orig = _mm256_load_ps(&A[TwoN * col + 2 * row]);

				// In lane shuffle. Never cross 128bit lanes, only shuffle
				// Real Imag parts of a lane. This is like two separate SSE
				// Shuffles, hence the use of a single _MM_SHUFFLE() Macro
				A_perm = _mm256_shuffle_ps(A_orig, A_orig,
						_MM_SHUFFLE(2, 3, 0, 1));
				__m256 tmp = _mm256_mul_ps(A_perm, xi);
				yv = _mm256_fmadd_ps(A_orig, xr, yv);

				// Instead of addsub, I am multiplying
				// signs into tmp, to use 2 FMAs. This appears to
				// be faster: 19.1GF vs 16.8GF at N=40
				yv = _mm256_fmadd_ps(signs, tmp, yv);

				_mm256_store_ps(&y[2 * row], yv);

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

#if defined(AVX2_FMA) || defined(AVX)
	const __m256 signs=_mm256_set_ps(1,-1,1,-1,1,-1,1,-1);
#endif


	for(IndexType col = 0; col < N; col++) {

		// thread 0 does = 0..9
		// thread 1 does = 10-19
		for(IndexType vrow=smt_id; vrow < N_vrows; vrow+=n_smt) {
			IndexType row = vrow*VECLEN2;
#ifdef SSE
			__m128 xr,xi, A_orig, A_perm;
			__m128 y1, y2;

			// NB: Single instruction broadcast...
			xr = _mm_load1_ps(&x[2*col]); // (unaligned load Broadcast...)
			xi = _mm_load1_ps(&x[2*col+1]); // Broadcast
			// Load VECLEN2 rows of A (A is row major so this is a simple thing)
			A_orig = _mm_load_ps(&A[ TwoN*col + 2*row] );
			A_perm = _mm_shuffle_ps( A_orig, A_orig, _MM_SHUFFLE(2,3,0,1));

			// Do the maths.. Load in rows of result
			__m128 yv = _mm_load_ps(&y[2*row]);

			// 2 FMAs one with addsub
			y1 = _mm_mul_ps(A_orig,xr);
			yv = _mm_add_ps(yv,y1);
			y2 = _mm_mul_ps(A_perm,xi);
			yv = _mm_addsub_ps(yv, y2);

			// Store
			_mm_store_ps(&y[2*row],yv);
#endif

#ifdef AVX
			// Use sign array
			__m256 xr,xi, A_orig, A_perm;
			__m256 y1,y2,y3;

			__m256 yv = _mm256_load_ps(&y[2*row]);
			xr = _mm256_broadcast_ss(&x[2*col]);
			xi = _mm256_broadcast_ss(&x[2*col+1]);
			A_orig = _mm256_load_ps( &A[ TwoN*col + 2*row] );

			// In lane shuffle. Never cross 128bit lanes, only shuffle
			// Real Imag parts of a lane. This is like two separate SSE
			// Shuffles, hence the use of a single _MM_SHUFFLE() Macro
			A_perm = _mm256_shuffle_ps(A_orig,A_orig, _MM_SHUFFLE(2,3,0,1));
			y1 = _mm256_mul_ps(A_orig,xr);
			yv = _mm256_add_ps(yv,y1);
			y2 = _mm256_mul_ps(A_perm, xi);
			y3 = _mm256_mul_ps(signs,y2);
			yv = _mm256_add_ps(yv,y3);

			_mm256_store_ps(&y[2*row],yv);
#endif


#ifdef AVX2

#ifdef AVX2_FMA_ADDSUB
			// Use addsub
			__m256 xr,xi, A_orig, A_perm;
			__m256 y1,y2;

			xr = _mm256_broadcast_ss(&x[2*col]);
			xi = _mm256_broadcast_ss(&x[2*col+1]);
			A_orig = _mm256_load_ps( &A[ TwoN*col + 2*row] );
			// In lane shuffle. Never cross 128bit lanes, only shuffle
			// Real Imag parts of a lane. This is like two separate SSE
			// Shuffles, hence the use of a single _MM_SHUFFLE() Macro
			A_perm = _mm256_shuffle_ps(A_orig,A_orig, _MM_SHUFFLE(2,3,0,1));
			__m256 yv = _mm256_load_ps(&y[2*row]);
			__m256 tmp = _mm256_mul_ps(A_perm,xi);
			yv = _mm256_fmadd_ps(A_orig,xr,yv);
			yv = _mm256_addsub_ps(yv,tmp);
			_mm256_store_ps(&y[2*row],yv);
#endif

#ifdef AVX2_FMA
			// Use sign array
			__m256 xr,xi, A_orig, A_perm;
			__m256 y1,y2;

			__m256 yv = _mm256_load_ps(&y[2*row]);
			xr = _mm256_broadcast_ss(&x[2*col]);
			xi = _mm256_broadcast_ss(&x[2*col+1]);
			A_orig = _mm256_load_ps( &A[ TwoN*col + 2*row] );

			// In lane shuffle. Never cross 128bit lanes, only shuffle
			// Real Imag parts of a lane. This is like two separate SSE
			// Shuffles, hence the use of a single _MM_SHUFFLE() Macro
			A_perm = _mm256_shuffle_ps(A_orig,A_orig, _MM_SHUFFLE(2,3,0,1));
			__m256 tmp = _mm256_mul_ps(A_perm, xi);
			yv = _mm256_fmadd_ps(A_orig,xr,yv);

			// Instead of addsub, I am multiplying
			// signs into tmp, to use 2 FMAs. This appears to
			// be faster: 19.1GF vs 16.8GF at N=40
			yv = _mm256_fmadd_ps(signs,tmp,yv);

			_mm256_store_ps(&y[2*row],yv);
#endif // AVX2 FMA
#endif // AVX2

		}
	}

}



}



#endif /* INCLUDE_LATTICE_CMAT_MULT_H_ */
