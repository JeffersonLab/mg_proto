
#include <omp.h>
#include <cstdio>

#include "lattice/cmat_mult.h"
#include "utils/memory.h"
#include "utils/print_utils.h"
#include <complex>
#include <cmath>
#include <cfloat>
#include <iostream>

#if defined(AVX) || defined(AVX2) || defined(AVX2_FMA)
#include <immintrin.h>
#endif

//#include "../../include/lattice/thread_info.h.bak"
namespace MG {

template<const int N>
void CMatMultNaiveT(std::complex<float>*y,
				   const std::complex<float>* A,
				   const std::complex<float>* x)
{

#pragma omp simd safelen(4)
    for(IndexType row=0; row < N; ++row) {
    	y[row] = std::complex<float>(0,0);
    }

    for(IndexType col=0; col < N; ++col) {

#pragma omp simd safelen(4)
      for(IndexType row=0; row < N; ++row) {

        // NB: These are complex multiplies
        y[row] += A[ row +  N*col ] * x[ col ];
      }
    }
}

template<const int N, const int Br, const int Bc>
void CMatMultNaiveBlockedT(std::complex<float>*y,
           const std::complex<float>* A,
           const std::complex<float>* x)
{

#pragma omp simd safelen(Br)
    for(IndexType row=0; row < N; ++row) {
      y[row] = std::complex<float>(0,0);
    }

    for(IndexType ocol=0; ocol < N; ocol += Bc) {
      for(IndexType orow=0; orow < N; orow += Br) {

        for(IndexType icol=0; icol < Bc; ++icol) {
          int col = ocol+icol;

#pragma omp simd safelen(Br) aligned(y,A,x : 8*Br)
          for(IndexType irow=0; irow < Br; ++irow) {
            int row = orow + irow;

            // NB: These are complex multiplies
            y[row] += A[ row +  N*col ] * x[ col ];
          }
        }
      }
    }
}

void CMatMultNaive(float* y,
				   const float* A,
				   const float* x,
				   IndexType N)
{
	std::complex<float>* yc = reinterpret_cast<std::complex<float>*>(y);
	const std::complex<float>* Ac = reinterpret_cast<const std::complex<float>*>(A);
	const std::complex<float>* xc = reinterpret_cast<const std::complex<float>*>(x);

	if( N == 6 ) {

		CMatMultNaiveT<6>(yc, Ac, xc);
	}
	else if( N == 8 ) {
		CMatMultNaiveBlockedT<8,8,8>(yc, Ac, xc);
	}
	else if ( N == 12 ) {

		CMatMultNaiveT<12>(yc, Ac, xc);
	}
	else if ( N == 16 ) {
		CMatMultNaiveBlockedT<16,8,8>(yc, Ac,xc);
	}
	else if (N == 24 ) {
		CMatMultNaiveBlockedT<24,8,8>(yc, Ac,xc);
	}
	else if (N == 32 ) {
		CMatMultNaiveBlockedT<32,8,8>(yc, Ac,xc);
	}
	else if (N == 48 ) {
		CMatMultNaiveBlockedT<48,8,8>(yc, Ac, xc);
	}
	else if (N == 64 ) {
		CMatMultNaiveBlockedT<64,8,8>(yc, Ac, xc);
	}
	else {
		MasterLog(ERROR, "Matrix size %d not supported in CMatMultNaive", N );
	}
}

template<const int N>
void CMatMultNaiveAddT(std::complex<float>*y,
				   const std::complex<float>* A,
				   const std::complex<float>* x)
{

    	for(IndexType col=0; col < N; ++col) {

#pragma omp simd safelen(4)
        for(IndexType row=0; row < N; ++row) {

    		// NB: These are complex multiplies
    		y[row] += A[ row  +  N*col  ] * x[ col ];
    	}
    }
}

template<const int N, const int Br, const int Bc>
void CMatMultNaiveAddBlockedT(std::complex<float>*y,
           const std::complex<float>* A,
           const std::complex<float>* x)
{

  for(IndexType ocol=0; ocol < N; ocol += Bc) {
    for(IndexType orow=0; orow < N; orow += Br) {

      for(IndexType icol=0; icol < Bc; ++icol) {
        IndexType col=ocol + icol;

#pragma omp simd safelen(Br) aligned(y,A,x: 8*Br)
        for(IndexType irow=0; irow < Br; ++irow) {
          IndexType row = orow + irow;

          // NB: These are complex multiplies
          y[row] += A[ row  +  N*col  ] * x[ col ];
        }
      }
    }
  }
}

void CMatMultNaiveAdd(float* y,
				   const float* A,
				   const float* x,
				   IndexType N)
{
	// Pretend these are arrays of complex numbers
	std::complex<float>* yc = reinterpret_cast<std::complex<float>*>(y);
	const std::complex<float>* Ac = reinterpret_cast<const std::complex<float>*>(A);
	const std::complex<float>* xc = reinterpret_cast<const std::complex<float>*>(x);

	if (N == 6 ) {
		CMatMultNaiveAddT<6>(yc, Ac, xc);
	}
	else if( N == 8 ) {
		CMatMultNaiveAddBlockedT<8,8,8>(yc, Ac, xc);
	}
	else if( N == 12 ) {
		CMatMultNaiveAddT<12>(yc, Ac, xc);
	}
	else if ( N == 16 ) {
		CMatMultNaiveAddBlockedT<16,8,8>(yc,Ac,xc);
	}
	else if (N == 24 ) {
		CMatMultNaiveAddBlockedT<24,8,8>(yc,Ac,xc);
	}
	else if (N == 32 ) {
		CMatMultNaiveAddBlockedT<32,8,8>(yc,Ac,xc);
	}
	else if (N == 48 ) {
		CMatMultNaiveAddBlockedT<48,8,8>(yc, Ac, xc);
	}
	else if (N == 64 ) {
		CMatMultNaiveAddBlockedT<64,8,8>(yc, Ac, xc);
	}
	else {
		MasterLog(ERROR, "Matrix size %d not supported in CMatMultNaiveAdd" , N );
	}
}

template<const int N>
void CMatMultNaiveCoeffAddT(std::complex<float>*y,
				   const float alpha,
				   const std::complex<float>* A,
				   const std::complex<float>* x)
{

    for(IndexType row=0; row < N; ++row) {
    	// Zero out result of row x col product
    	std::complex<float> tmp=std::complex<float>(0,0);

    	// Accumulate row x col product
    	for(IndexType col=0; col < N; ++col) {

    		// NB: These are complex multiplies
    		tmp += A[ N*row + col ] * x[ col ];
    	}

    	// Multiply it by alpha
    	y[row] += alpha*tmp;
    }
}


void CMatMultNaiveCoeffAdd(float* y,
				   const float alpha,
				   const float* A,
				   const float* x,
				   IndexType N)
{
	// Pretend these are arrays of complex numbers
	std::complex<float>* yc = reinterpret_cast<std::complex<float>*>(y);
	const std::complex<float>* Ac = reinterpret_cast<const std::complex<float>*>(A);
	const std::complex<float>* xc = reinterpret_cast<const std::complex<float>*>(x);

	if (N == 6 ) {
		CMatMultNaiveCoeffAddT<6>(yc, alpha, Ac, xc);
	}
	else if( N == 8 ) {
		CMatMultNaiveCoeffAddT<8>(yc, alpha, Ac, xc);
	}
	else if( N == 12 ) {
		CMatMultNaiveCoeffAddT<12>(yc, alpha, Ac, xc);
	}
	else if ( N == 16 ) {
		CMatMultNaiveCoeffAddT<16>(yc, alpha, Ac,xc);
	}
	else if (N == 24 ) {
		CMatMultNaiveCoeffAddT<24>(yc, alpha, Ac,xc);
	}
	else if (N == 32 ) {
		CMatMultNaiveCoeffAddT<32>(yc, alpha, Ac,xc);
	}
	else if (N == 48 ) {
		CMatMultNaiveCoeffAddT<48>(yc, alpha, Ac, xc);
	}
	else if (N == 64 ) {
		CMatMultNaiveCoeffAddT<64>(yc, alpha, Ac, xc);
	}
	else {
		MasterLog(ERROR, "Matrix size %d not supported in CMatMultNaiveAdd" , N );
	}
}


#if 1
void CMatMult(float *y,
			  const float* A,
			  const float* x,
			  const IndexType N,
			  const int tid,
			  const int nthreads)
{

	const IndexType TwoN=2*N;

#ifdef AVX2_FMA
	const __m256 signs=_mm256_set_ps(1,-1,1,-1,1,-1,1,-1);
#endif

	int num_veclen2 = N/VECLEN2; // For N=40 this is 20

	int blocks_per_thread = num_veclen2/nthreads; // For 2 threads this is 10
	if( num_veclen2 % nthreads > 0) blocks_per_thread++;

	// For thread 0 => 0, for thread 1 => 10
	int minblock = tid*blocks_per_thread;
	// For thread 0 => 10, for thread 2 = 20
	int maxblock = (tid+1)*blocks_per_thread > num_veclen2 ? num_veclen2 : (tid+1)*blocks_per_thread;

	/* Initialize y */
	for(IndexType vrow=minblock; vrow < maxblock; ++vrow) {

#pragma omp simd aligned(y:16)
		for(IndexType i=0; i < VECLEN; ++i) {
			y[vrow*VECLEN+i] = 0;
		}
	}

	for(IndexType col = 0; col < N; col++) {

		// thread 0 does = 0..9
		// thread 1 does = 10-19
		for(int vrow=minblock; vrow < maxblock; vrow++) {
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
#if 1

#else
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
#endif



		}
	}

}
#endif

 float* tmp_space;

 void AllocSpace(const int N)
{
	tmp_space = static_cast<float*>(MG::MemoryAllocate(8*2*N*sizeof(float)));
}

void DestroySpace(void)
{
	if( tmp_space != nullptr ) MG::MemoryFree(tmp_space);
}


#if 1
void CMatMultVrow(float *y,
			  const float* A,
			  const float* x,
			  const IndexType N,
			  const int min_vrow,
			  const int max_vrow)
{

	const IndexType TwoN=2*N;

#ifdef AVX2_FMA
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

#endif

}
