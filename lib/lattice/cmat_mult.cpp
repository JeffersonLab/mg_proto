#if 0

#define SSE
#define VECLEN 4  // SSE
#include <xmmintrin.h>
#include <pmmintrin.h>

#else

#define AVX2
#define VECLEN 8 	// AVX2
#define AVX2_FMA
#undef AVX2_FMA_ADDSUB
#include <immintrin.h>
#endif

#define VECLEN2 (VECLEN/2)



#include "lattice/cmat_mult.h"
#include <complex>

#include <immintrin.h>
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



void CMatMult(float *y,
			  const float* A,
			  const float* x,
			  const IndexType N)
{

	const IndexType TwoN=2*N;

#ifdef AVX2
	const __m256 signs=_mm256_set_ps(1,-1,1,-1,1,-1,1,-1);
#endif

	/* Initialize y */
	for(IndexType row=0; row < TwoN; ++row) {
	    y[row] = 0;
	}

	// 2 columns
	for(IndexType col = 0; col < N; col++) {
		for(int row=0; row < N; row += VECLEN2) {

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
#else

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
#endif // SSE

		}
	}

}

}
