#include "lattice/cmat_mult.h"
#include <complex>
#include <xmmintrin.h>
#include <pmmintrin.h>
namespace MGGeometry {



void CMatMultNaive(std::complex<float>*y,
				   const std::complex<float>* A,
				   const std::complex<float>* x,
				   IndexType N)
{
    for(IndexType row=0; row < N; ++row) {
    	y[row] = std::complex<float>(0,0);
    }
    for(IndexType row=0; row < N; ++row) {
    	for(IndexType col=0; col < N; ++col) {

    		// NB: These are complex multiplies
    		y[row] += A[ N*row + col ] * x[ col ];
    	}
    }
}

#define VECLEN 4  // SSE
#define VECLEN2 (VECLEN/2)


void CMatMult(float *y,
			  const float* A,
			  const float* x,
			  const IndexType N)
{

	const IndexType TwoN=2*N;

	/* Initialize y */
	for(IndexType row=0; row < TwoN; ++row) {
	    y[row] = 0;
	}

	// 2 columns
	for(IndexType col = 0; col < N; col++) {
		for(int row=0; row < N; row += VECLEN2) {
#if 0
			float xr[VECLEN] __attribute((aligned(64)));
			float xi[VECLEN] __attribute((aligned(64)));
			float A_orig[VECLEN] __attribute((aligned(64)));
			float A_perm[VECLEN] __attribute((aligned(64)));

#else
			__m128 xr,xi, A_orig, A_perm;
			__m128 y1, y2;
#endif

			// NB: Single instruction broadcast...
#if 0
#pragma omp simd
			for(IndexType i=0; i < VECLEN; ++i) {
				xr[i] = x[2*col];
			}
#else
			xr = _mm_load1_ps(&x[2*col]); // (unaligned load Broadcast...)
#endif
			// NB: Single instruction broadcast
#if 0
#pragma omp simd
			for(IndexType i=0; i < VECLEN; ++i) {
				xi[i] = x[2*col + 1];
			}
#else
			xi = _mm_load1_ps(&x[2*col+1]); // Broadcast
#endif
			// This is kinda transposy: Get COLUMN of a into A_orig
#if 0
#pragma omp simd
			for(IndexType i=0; i < VECLEN2; ++i) {
				IndexType row = block*VECLEN2 + i;
				A_orig[2*i] = A[TwoN*row + 2*col];
				A_orig[2*i+1] = A[ TwoN*row + 2*col+1];
			}
#else
			{

				// Load VECLEN2 rows of A (A is row major so this is a simple thing)
				A_orig = _mm_load_ps(&A[ TwoN*col + VECLEN2*row] );


			}

#endif



			// This would work best by permuting the previous column
			// Ideally a single shuffle
#if 0
#pragma omp simd
			for(IndexType i=0; i < VECLEN2; ++i) {
				IndexType row = block*VECLEN2 + i;
				A_perm[2*i] = -A[TwoN*row + 2*col + 1];
				A_perm[2*i+1] = A[TwoN*row + 2*col];
			}
#else
			{
				// Permute them
				A_perm = _mm_shuffle_ps( A_orig, A_orig, _MM_SHUFFLE(2,3,0,1));
			}
#endif

#if 0
			// Two FMAs
#pragma omp simd
			for (IndexType i = 0; i < VECLEN; ++i) {
				y[ start+i ] +=  A_orig[i]* xr[i] + A_perm[i]*xi[i];
			}
#else
			{
				// Do the maths.. Load in rows of result
				__m128 yv = _mm_load_ps(&y[VECLEN2*row]);

				// 2 FMAs one with addsub
				y1 = _mm_mul_ps(A_orig,xr);
				yv = _mm_add_ps(yv,y1);
				y2 = _mm_mul_ps(A_perm,xi);
				yv = _mm_addsub_ps(yv, y2);

				// Store
				_mm_store_ps(&y[VECLEN2*row],yv);
			}
#endif
		}

#if 0
		for(IndexType row= n_block*VECLEN2 ; row < 2*N; ++row ) {
			y[2*row] +=  A[TwoN * row + 2 * col]* x[2*col];
				y[2*row + 1] +=  A[TwoN * row + 2	* col + 1] * x[2*col];
				y[2*row] -=  A[TwoN * row + 2 * col + 1] * x[2*col + 1];
				y[2*row + 1] += A[TwoN * row + 2 * col] * x[2*col+1];
		}
#endif
	}

}

}
