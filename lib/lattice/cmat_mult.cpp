

#include <cstdio>

#include "MG_config.h"
#include "lattice/cmat_mult.h"
#include "utils/memory.h"
#include "utils/print_utils.h"
#include <complex>
#include <cmath>
#include <cfloat>
#include <iostream>

#include <Eigen/Dense>
using namespace Eigen;

namespace MG {



template<const int N>
void CMatMultNaiveT(std::complex<float>*y,
		const std::complex<float>* A,
		const std::complex<float>* x)
{


#pragma omp simd aligned(y:64)
	for(int row=0; row < N; ++row) {
		y[row] = std::complex<float>(0,0);
	}


	for(IndexType col=0; col < N; ++col) {


#pragma omp simd aligned(y,A,x:64)
		for(IndexType row=0; row < N; ++row) {
			y[row] += A[ row +  N*col ] * x[ col ];
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
		CMatMultNaiveT<8>(yc, Ac, xc);
	}
	else if ( N == 12 ) {
		CMatMultNaiveT<12>(yc, Ac, xc);
	}
	else if ( N == 16 ) {
		CMatMultNaiveT<16>(yc, Ac,xc);
	}
	else if (N == 24 ) {
		CMatMultNaiveT<24>(yc, Ac,xc);
	}
	else if (N == 32 ) {
		CMatMultNaiveT<32>(yc, Ac,xc);
	}
	else if (N == 48 ) {
		CMatMultNaiveT<48>(yc, Ac, xc);
	}
	else if (N == 56 ) {
		CMatMultNaiveT<56>(yc, Ac, xc);
	}
	else if (N == 64 ) {
		CMatMultNaiveT<64>(yc, Ac, xc);
	}
	else {
		MasterLog(ERROR, "Matrix size %d not supported in CMatMultNaive", N );
	}
}

template<const int N>
void CMatMultAddNaiveT(std::complex<float>*y,
           const std::complex<float>* A,
           const std::complex<float>* x)
{

	for(IndexType col=0; col <N; ++col) {

#pragma omp simd aligned(y,A,x: 64)
		for(IndexType row=0; row < N; ++row) {


			// NB: These are complex multiplies
			y[row] += A[ row  +  N*col  ] * x[ col ];
		}

	}
}

void CMatMultAddNaive(float* y,
				   const float* A,
				   const float* x,
				   IndexType N)
{
	// Pretend these are arrays of complex numbers
	std::complex<float>* yc = reinterpret_cast<std::complex<float>*>(y);
	const std::complex<float>* Ac = reinterpret_cast<const std::complex<float>*>(A);
	const std::complex<float>* xc = reinterpret_cast<const std::complex<float>*>(x);

	if (N == 6 ) {
		CMatMultAddNaiveT<6>(yc, Ac, xc);
	}
	else if( N == 8 ) {
		CMatMultAddNaiveT<8>(yc, Ac, xc);
	}
	else if( N == 12 ) {
		CMatMultAddNaiveT<12>(yc, Ac, xc);
	}
	else if ( N == 16 ) {
		CMatMultAddNaiveT<16>(yc,Ac,xc);
	}
	else if (N == 24 ) {
		CMatMultAddNaiveT<24>(yc,Ac,xc);
	}
	else if (N == 32 ) {
		CMatMultAddNaiveT<32>(yc,Ac,xc);
	}
	else if (N == 48 ) {
		CMatMultAddNaiveT<48>(yc, Ac, xc);
	}
	else if (N == 56 ) {
		CMatMultAddNaiveT<56>(yc, Ac, xc);
	}
	else if (N == 64 ) {
		CMatMultAddNaiveT<64>(yc, Ac, xc);
	}
	else {
		MasterLog(ERROR, "Matrix size %d not supported in CMatMultNaiveAdd" , N );
	}
}


template<const int N>
void CMatMultCoeffAddNaiveT(std::complex<float>*y,
			float alpha,
           const std::complex<float>* A,
           const std::complex<float>* x)
{


	  for(IndexType col=0; col < N; ++col) {

		  const std::complex<float> tmp=alpha*x[col];
#pragma omp simd aligned(y,A,x: 64)
		  for(IndexType row=0; row < N; ++row) {


			  // NB: These are complex multiplies
			  y[row] += A[ row  +  N*col  ] * tmp;
		  }
	  }



}

void CMatMultCoeffAddNaive(float* y,
		float alpha,
		const float* A,
		const float* x,
		IndexType N)
{
	// Pretend these are arrays of complex numbers
	std::complex<float>* yc = reinterpret_cast<std::complex<float>*>(y);
	const std::complex<float>* Ac = reinterpret_cast<const std::complex<float>*>(A);
	const std::complex<float>* xc = reinterpret_cast<const std::complex<float>*>(x);

	if (N == 6 ) {
		CMatMultCoeffAddNaiveT<6>(yc, alpha, Ac, xc);
	}
	else if( N == 8 ) {
		CMatMultCoeffAddNaiveT<8>(yc, alpha, Ac, xc);
	}
	else if( N == 12 ) {
		CMatMultCoeffAddNaiveT<12>(yc,alpha, Ac, xc);
	}
	else if ( N == 16 ) {
		CMatMultCoeffAddNaiveT<16>(yc,alpha, Ac,xc);
	}
	else if (N == 24 ) {
		CMatMultCoeffAddNaiveT<24>(yc,alpha,Ac,xc);
	}
	else if (N == 32 ) {
		CMatMultCoeffAddNaiveT<32>(yc,alpha,Ac,xc);
	}
	else if (N == 48 ) {
		CMatMultCoeffAddNaiveT<48>(yc,alpha,Ac, xc);
	}
	else if (N == 56 ) {
		CMatMultCoeffAddNaiveT<56>(yc,alpha,Ac, xc);
	}
	else if (N == 64 ) {
		CMatMultCoeffAddNaiveT<64>(yc, alpha, Ac, xc);
	}
	else {
		MasterLog(ERROR, "Matrix size %d not supported in CMatMultNaiveAdd" , N );
	}
}


template<const int N>
void CMatAdjMultNaiveT(std::complex<float>*y,
		const std::complex<float>* A,
		const std::complex<float>* x)
{

#pragma omp simd aligned(y:64)
	for(int row=0; row < N; ++row) {
		y[row] = std::complex<float>(0,0);
	}



				for(IndexType row=0; row < N; ++row) {

					for(IndexType col=0; col < N; ++col) {
							y[row] += std::conj(A[ col +  N*row ]) * x[ col ];
					}
				}

}

void CMatAdjMultNaive(float* y,
				   const float* A,
				   const float* x,
				   IndexType N)
{
	std::complex<float>* yc = reinterpret_cast<std::complex<float>*>(y);
	const std::complex<float>* Ac = reinterpret_cast<const std::complex<float>*>(A);
	const std::complex<float>* xc = reinterpret_cast<const std::complex<float>*>(x);

	if( N == 6 ) {
		CMatAdjMultNaiveT<6>(yc, Ac, xc);
	}
	else if( N == 8 ) {
		CMatAdjMultNaiveT<8>(yc, Ac, xc);

	}
	else if ( N == 12 ) {
		CMatAdjMultNaiveT<12>(yc, Ac, xc);
	}
	else if ( N == 16 ) {
		CMatAdjMultNaiveT<16>(yc, Ac,xc);
	}
	else if (N == 24 ) {
		CMatAdjMultNaiveT<24>(yc, Ac,xc);
	}
	else if (N == 32 ) {
		CMatAdjMultNaiveT<32>(yc, Ac,xc);
	}
	else if (N == 40 ) {
		CMatAdjMultNaiveT<40>(yc, Ac,xc);
	}
	else if (N == 48 ) {
		CMatAdjMultNaiveT<48>(yc, Ac, xc);
	}
	else if (N == 56 ) {
		CMatAdjMultNaiveT<56>(yc, Ac, xc);
	}
	else if (N == 64 ) {
		CMatAdjMultNaiveT<64>(yc, Ac, xc);
	}
	else {
		MasterLog(ERROR, "Matrix size %d not supported in CMatAdjMultNaive", N );
	}
}

template<const int N>
void GcCMatMultGcNaiveT(std::complex<float>*y,
           const std::complex<float>* A,
           const std::complex<float>* x)
{

	constexpr int NbyTwo = N/2;

#pragma omp simd aligned(y:64)
	for(int row=0; row < N; ++row) {
		y[row] = std::complex<float>(0,0);
	}


	for(IndexType col=0; col < NbyTwo; ++col) {

#pragma omp simd aligned(y,A,x:64)
		for(IndexType row=0; row < NbyTwo; ++row) {

			y[row] += A[ row +  N*col ] * x[ col ];
		}
#pragma omp simd aligned(y,A,x:64)
		for(IndexType row=NbyTwo; row < N; ++row) {

			y[row] -= A[ row +  N*col ] * x[ col ];
		}

	}

	for(IndexType col=NbyTwo; col < N; ++col) {

#pragma omp simd aligned(y,A,x:64)
		for(IndexType row=0; row < NbyTwo; ++row) {

			y[row] -= A[ row +  N*col ] * x[ col ];
		}
#pragma omp simd aligned(y,A,x:64)
		for(IndexType row=NbyTwo; row < N; ++row) {

			y[row] += A[ row +  N*col ] * x[ col ];
		}

	}
}
void GcCMatMultGcNaive(float* y,
				   const float* A,
				   const float* x,
				   IndexType N)
{
	std::complex<float>* yc = reinterpret_cast<std::complex<float>*>(y);
	const std::complex<float>* Ac = reinterpret_cast<const std::complex<float>*>(A);
	const std::complex<float>* xc = reinterpret_cast<const std::complex<float>*>(x);

	if( N == 6 ) {
		GcCMatMultGcNaiveT<6>(yc, Ac, xc);
	}
	else if( N == 8 ) {
		GcCMatMultGcNaiveT<8>(yc, Ac, xc);

	}
	else if ( N == 12 ) {
		GcCMatMultGcNaiveT<12>(yc, Ac, xc);
	}
	else if ( N == 16 ) {
		GcCMatMultGcNaiveT<16>(yc, Ac,xc);
	}
	else if (N == 24 ) {
		GcCMatMultGcNaiveT<24>(yc, Ac,xc);
	}
	else if (N == 32 ) {
		GcCMatMultGcNaiveT<32>(yc, Ac,xc);
	}
	else if (N == 40 ) {
		GcCMatMultGcNaiveT<40>(yc, Ac,xc);
	}
	else if (N == 48 ) {
		GcCMatMultGcNaiveT<48>(yc, Ac, xc);
	}
	else if (N == 56 ) {
		GcCMatMultGcNaiveT<56>(yc, Ac, xc);
	}
	else if (N == 64 ) {
		GcCMatMultGcNaiveT<64>(yc, Ac, xc);
	}
	else {
		MasterLog(ERROR, "Matrix size %d not supported in GcCMatMultGcNaive", N );
	}
}


template<const int N>
void GcCMatMultGcCoeffAddNaiveT(std::complex<float>*y, float alpha,
           const std::complex<float>* A,
           const std::complex<float>* x)
{

	constexpr int NbyTwo = N/2;

	for(IndexType col=0; col < NbyTwo; ++col) {
		const std::complex<float> ax = alpha*x[col];

#pragma omp simd aligned(y,A,x:64)
		for(IndexType row=0; row < NbyTwo; ++row) {

			y[row] += A[ row +  N*col ] * ax;
		}
#pragma omp simd aligned(y,A,x:64)
		for(IndexType row=NbyTwo; row < N; ++row) {

			y[row] -=  A[ row +  N*col ] * ax;
		}

	}

	for(IndexType col=NbyTwo; col < N; ++col) {
		const std::complex<float> ax = alpha*x[col];

#pragma omp simd aligned(y,A,x:64)
		for(IndexType row=0; row < NbyTwo; ++row) {

			y[row] -= A[ row +  N*col ] * ax;
		}
#pragma omp simd aligned(y,A,x:64)
		for(IndexType row=NbyTwo; row < N; ++row) {

			y[row] += A[ row +  N*col ] * ax;;
		}

	}
}

void GcCMatMultGcCoeffAddNaive(float* y, float alpha,
				   const float* A,
				   const float* x,
				   IndexType N)
{
	std::complex<float>* yc = reinterpret_cast<std::complex<float>*>(y);
	const std::complex<float>* Ac = reinterpret_cast<const std::complex<float>*>(A);
	const std::complex<float>* xc = reinterpret_cast<const std::complex<float>*>(x);

	if( N == 6 ) {
		GcCMatMultGcCoeffAddNaiveT<6>(yc, alpha, Ac, xc);
	}
	else if( N == 8 ) {
		GcCMatMultGcCoeffAddNaiveT<8>(yc, alpha, Ac, xc);

	}
	else if ( N == 12 ) {
		GcCMatMultGcCoeffAddNaiveT<12>(yc, alpha, Ac, xc);
	}
	else if ( N == 16 ) {
		GcCMatMultGcCoeffAddNaiveT<16>(yc, alpha, Ac,xc);
	}
	else if (N == 24 ) {
		GcCMatMultGcCoeffAddNaiveT<24>(yc, alpha, Ac,xc);
	}
	else if (N == 32 ) {
		GcCMatMultGcCoeffAddNaiveT<32>(yc, alpha, Ac,xc);
	}
	else if (N == 40 ) {
		GcCMatMultGcCoeffAddNaiveT<40>(yc, alpha, Ac,xc);
	}
	else if (N == 48 ) {
		GcCMatMultGcCoeffAddNaiveT<48>(yc, alpha, Ac, xc);
	}
	else if (N == 56 ) {
		GcCMatMultGcCoeffAddNaiveT<56>(yc, alpha, Ac, xc);
	}
	else if (N == 64 ) {
		GcCMatMultGcCoeffAddNaiveT<64>(yc, alpha, Ac, xc);
	}
	else {
		MasterLog(ERROR, "Matrix size %d not supported in CMatAdjMultNaive", N );
	}
}


#ifdef MG_USE_AVX512
template<const int N>
void CMatMultAVX512T(float *y,
          const float* A,
           const float* x)
{

	constexpr int TwoN = 2*N;

	for(int row=0; row < TwoN; row +=16) {
		__m512 z=_mm512_setzero_ps();
		_mm512_store_ps(y + row, z );
	}

	for(IndexType col=0; col < N; col++) {

		__m512 xcol_re = _mm512_set1_ps(x[2*col]);
		__m512 xcol_im = _mm512_set1_ps(x[2*col+1]);

#pragma unroll
		for(IndexType row=0; row < TwoN; row+=16) {

			__m512 A_col = _mm512_load_ps( A + row + TwoN*col );
			__m512 A_perm = _mm512_shuffle_ps( A_col, A_col, _MM_SHUFFLE(2,3,0,1));

			__m512 y_vec = _mm512_load_ps( y + row );

			y_vec = _mm512_fmaddsub_ps( A_col, xcol_re,
					_mm512_fmaddsub_ps( A_perm,xcol_im, y_vec));

			_mm512_store_ps( y + row, y_vec);
		}
	}
}

void CMatMultAVX512(float *y, const float *A, const float *x, IndexType N )
{
	std::complex<float>* yc = reinterpret_cast<std::complex<float>*>(y);
	const std::complex<float>* Ac = reinterpret_cast<const std::complex<float>*>(A);
	const std::complex<float>* xc = reinterpret_cast<const std::complex<float>*>(x);

	if( N == 6 ) {
		CMatMultNaiveT<6>(yc, Ac, xc);
	}
	else if( N == 12) {
		CMatMultNaiveT<12>(yc, Ac, xc);
	}
	else if( N == 8 ) {
		CMatMultAVX512T<8>(y, A, x);
	}
	else if (N == 16 ) {
		CMatMultAVX512T<16>(y, A, x);
	}
	else if (N == 24 ) {
		CMatMultAVX512T<24>(y, A, x);
	}
	else if (N == 32 ) {
		CMatMultAVX512T<32>(y, A, x);
	}
	else if (N == 40 ) {
		CMatMultAVX512T<40>(y, A, x);
	}
	else if (N == 48 ) {
		CMatMultAVX512T<48>(y, A, x);
	}
	else if (N == 56 ) {
		CMatMultAVX512T<56>(y, A, x);
	}
	else if (N == 64 ) {
		CMatMultAVX512T<64>(y, A, x);
	}
	else {
		MasterLog(ERROR, "Matrix size %d not supported in CMatMultAVX512", N );
	}
}



template<const int N>
void CMatMultAddAVX512T(float *y,
          const float* A,
           const float* x)
{

	constexpr int TwoN = 2*N;

	for(IndexType col=0; col < N; col++) {

		__m512 xcol_re = _mm512_set1_ps(x[2*col]);
		__m512 xcol_im = _mm512_set1_ps(x[2*col+1]);

#pragma unroll
		for(IndexType row=0; row < TwoN; row+=16) {

			__m512 A_col = _mm512_load_ps( A + row + TwoN*col );
			__m512 A_perm = _mm512_shuffle_ps( A_col, A_col, _MM_SHUFFLE(2,3,0,1));

			__m512 y_vec = _mm512_load_ps( y + row );

			y_vec = _mm512_fmaddsub_ps( A_col, xcol_re,
					_mm512_fmaddsub_ps( A_perm,xcol_im, y_vec));

			_mm512_store_ps( y + row, y_vec);
		}
	}
}

void CMatMultAddAVX512(float *y, const float *A, const float *x, IndexType N )
{
	std::complex<float>* yc = reinterpret_cast<std::complex<float>*>(y);
	const std::complex<float>* Ac = reinterpret_cast<const std::complex<float>*>(A);
	const std::complex<float>* xc = reinterpret_cast<const std::complex<float>*>(x);

	if( N == 6 ) {
		CMatMultAddNaiveT<6>(yc, Ac, xc);
	}
	else if( N == 12) {
		CMatMultAddNaiveT<12>(yc, Ac, xc);
	}
	else if( N == 8 ) {
		CMatMultAddAVX512T<8>(y, A, x);
	}
	else if (N == 16 ) {
		CMatMultAddAVX512T<16>(y, A, x);
	}
	else if (N == 24 ) {
		CMatMultAddAVX512T<24>(y, A, x);
	}
	else if (N == 32 ) {
		CMatMultAddAVX512T<32>(y, A, x);
	}
	else if (N == 40 ) {
		CMatMultAddAVX512T<40>(y, A, x);
	}
	else if (N == 48 ) {
		CMatMultAddAVX512T<48>(y, A, x);
	}
	else if (N == 56 ) {
		CMatMultAddAVX512T<56>(y, A, x);
	}
	else if (N == 64 ) {
		CMatMultAddAVX512T<64>(y, A, x);
	}
	else {
		MasterLog(ERROR, "Matrix size %d not supported in CMatMultAVX512", N );
	}
}

template<const int N>
void CMatMultCoeffAddAVX512T(float *y,
		 float alpha,
          const float* A,
           const float* x)
{

	constexpr int TwoN = 2*N;
	__m512 alphav = _mm512_set1_ps(alpha);

	for(IndexType col=0; col < N; col++) {

		__m512 xcol_re =_mm512_mul_ps(alphav, _mm512_set1_ps(x[2*col]));
		__m512 xcol_im =_mm512_mul_ps(alphav, _mm512_set1_ps(x[2*col+1]));

#pragma unroll
		for(IndexType row=0; row < TwoN; row+=16) {

			__m512 A_col = _mm512_load_ps( A + row + TwoN*col );
			__m512 A_perm = _mm512_shuffle_ps( A_col, A_col, _MM_SHUFFLE(2,3,0,1));

			__m512 y_vec = _mm512_load_ps( y + row );

			y_vec = _mm512_fmaddsub_ps( A_col, xcol_re,
					_mm512_fmaddsub_ps( A_perm,xcol_im, y_vec));

			_mm512_store_ps( y + row, y_vec);
		}
	}
}

void CMatMultCoeffAddAVX512(float *y, float alpha, const float *A, const float *x, IndexType N )
{
	std::complex<float>* yc = reinterpret_cast<std::complex<float>*>(y);
	const std::complex<float>* Ac = reinterpret_cast<const std::complex<float>*>(A);
	const std::complex<float>* xc = reinterpret_cast<const std::complex<float>*>(x);

	if( N == 6 ) {
		CMatMultCoeffAddNaiveT<6>(yc, alpha, Ac, xc);
	}
	else if( N == 12) {
		CMatMultCoeffAddNaiveT<12>(yc, alpha, Ac, xc);
	}
	else if( N == 8 ) {
		CMatMultCoeffAddAVX512T<8>(y, alpha, A, x);
	}
	else if (N == 16 ) {
		CMatMultCoeffAddAVX512T<16>(y, alpha, A, x);
	}
	else if (N == 24 ) {
		CMatMultCoeffAddAVX512T<24>(y, alpha, A, x);
	}
	else if (N == 32 ) {
		CMatMultCoeffAddAVX512T<32>(y, alpha, A, x);
	}
	else if (N == 40 ) {
		CMatMultCoeffAddAVX512T<40>(y, alpha, A, x);
	}
	else if (N == 48 ) {
		CMatMultCoeffAddAVX512T<48>(y, alpha, A, x);
	}
	else if (N == 56 ) {
		CMatMultCoeffAddAVX512T<56>(y, alpha, A, x);
	}
	else if (N == 64 ) {
		CMatMultCoeffAddAVX512T<64>(y, alpha, A, x);
	}
	else {
		MasterLog(ERROR, "Matrix size %d not supported in CMatMultAVX512", N );
	}
}

void CMatAdjMultAVX512(float* y,
				   const float* A,
				   const float* x,
				   IndexType N)
{
	std::complex<float>* yc = reinterpret_cast<std::complex<float>*>(y);
	const std::complex<float>* Ac = reinterpret_cast<const std::complex<float>*>(A);
	const std::complex<float>* xc = reinterpret_cast<const std::complex<float>*>(x);

	if( N == 6 ) {
		CMatAdjMultNaiveT<6>(yc, Ac, xc);
	}
	else if( N == 8 ) {
		CMatAdjMultNaiveT<8>(yc, Ac, xc);

	}
	else if ( N == 12 ) {
		CMatAdjMultNaiveT<12>(yc, Ac, xc);
	}
	else if ( N == 16 ) {
		CMatAdjMultNaiveT<16>(yc, Ac,xc);
	}
	else if (N == 24 ) {
		CMatAdjMultNaiveT<24>(yc, Ac,xc);
	}
	else if (N == 32 ) {
		CMatAdjMultNaiveT<32>(yc, Ac,xc);
	}
	else if (N == 40 ) {
		CMatAdjMultNaiveT<40>(yc, Ac,xc);
	}
	else if (N == 48 ) {
			CMatAdjMultNaiveT<48>(yc, Ac, xc);
	}
	else if (N == 56 ) {
			CMatAdjMultNaiveT<56>(yc, Ac, xc);
	}
	else if (N == 64 ) {
		CMatAdjMultNaiveT<64>(yc, Ac, xc);
	}
	else {
		MasterLog(ERROR, "Matrix size %d not supported in CMatMultNaive", N );
	}
}

template<const int N>
void GcCMatMultGcAVX512T(float *y,
           const float* A,
           const float* x)
{

	constexpr int NbyTwo = N/2;
	constexpr int TwoN = 2*N;

	// Zero result
	for(IndexType row=0; row < TwoN; row+=16) {
		__m512 y_vec = _mm512_setzero_ps();
		_mm512_store_ps(y + row, y_vec);
	}

	for(IndexType col=0; col < NbyTwo; ++col) {

		__m512 x_r =  _mm512_set1_ps(x[2*col] );
		__m512 x_i =  _mm512_set1_ps(x[2*col+1]);
		__m512 mx_r =_mm512_set1_ps(-x[2*col]);

		for(IndexType row=0; row < N; row+=16) {
			__m512 A_col = _mm512_load_ps(A + row + TwoN*col );
			__m512 A_perm = _mm512_shuffle_ps( A_col, A_col, _MM_SHUFFLE(2,3,0,1));

			__m512 y_vec = _mm512_load_ps( y + row );

			y_vec = _mm512_fmaddsub_ps( A_col, x_r,
					_mm512_fmaddsub_ps( A_perm,x_i, y_vec));

			_mm512_store_ps( y + row, y_vec);
		}

		for(IndexType row=N; row < TwoN; row+=16) {
			__m512 A_col = _mm512_load_ps(A + row + TwoN*col );
			__m512 A_perm = _mm512_shuffle_ps( A_col, A_col, _MM_SHUFFLE(2,3,0,1));

			__m512 y_vec = _mm512_load_ps( y + row );
			y_vec=_mm512_fmsubadd_ps( A_col, mx_r, _mm512_fmsubadd_ps( A_perm, x_i, y_vec));
			_mm512_store_ps( y + row, y_vec);
		}
	}

	for(IndexType col=NbyTwo; col < N; ++col) {
		__m512 x_r =  _mm512_set1_ps(x[2*col] );
		__m512 x_i =  _mm512_set1_ps(x[2*col+1]);
		__m512 mx_r =_mm512_set1_ps(-x[2*col]);

		for(IndexType row=0; row < N; row+=16) {
			__m512 A_col = _mm512_load_ps(A + row + TwoN*col );
			__m512 A_perm = _mm512_shuffle_ps( A_col, A_col, _MM_SHUFFLE(2,3,0,1));

			__m512 y_vec = _mm512_load_ps( y + row );

			y_vec=_mm512_fmsubadd_ps( A_col, mx_r, _mm512_fmsubadd_ps( A_perm, x_i, y_vec));

			_mm512_store_ps( y + row, y_vec);
		}


		for(IndexType row=N; row < TwoN; row+=16) {
			__m512 A_col = _mm512_load_ps(A + row + TwoN*col );
			__m512 A_perm = _mm512_shuffle_ps( A_col, A_col, _MM_SHUFFLE(2,3,0,1));

			__m512 y_vec = _mm512_load_ps( y + row );

			y_vec = _mm512_fmaddsub_ps( A_col, x_r,
					_mm512_fmaddsub_ps( A_perm,x_i, y_vec));

			_mm512_store_ps( y + row, y_vec);
		}

	}
}


void GcCMatMultGcAVX512(float* y,
				   const float* A,
				   const float* x,
				   IndexType N)
{
	std::complex<float>* yc = reinterpret_cast<std::complex<float>*>(y);
	const std::complex<float>* Ac = reinterpret_cast<const std::complex<float>*>(A);
	const std::complex<float>* xc = reinterpret_cast<const std::complex<float>*>(x);

	if( N == 6 ) {
		GcCMatMultGcNaiveT<6>(yc, Ac, xc);
	}
	else if( N == 8 ) {
		GcCMatMultGcNaiveT<8>(yc, Ac, xc);

	}
	else if ( N == 12 ) {
		GcCMatMultGcNaiveT<12>(yc, Ac, xc);
	}
	else if ( N == 16 ) {
		GcCMatMultGcAVX512T<16>(y, A,x);
	}
	else if (N == 24 ) {
		GcCMatMultGcNaiveT<24>(yc, Ac,xc);
	}
	else if (N == 32 ) {
		GcCMatMultGcAVX512T<32>(y, A,x);
	}
	else if (N == 40 ) {
		GcCMatMultGcNaiveT<40>(yc, Ac,xc);
	}
	else if (N == 48 ) {
		GcCMatMultGcAVX512T<48>(y, A, x);
	}
	else if (N == 56 ) {
		GcCMatMultGcNaiveT<56>(yc, Ac, xc);
	}
	else if (N == 64 ) {
		GcCMatMultGcAVX512T<64>(y, A, x);
	}
	else {
		MasterLog(ERROR, "Matrix size %d not supported in GcCMatMultGcAVX512", N );
	}
}


template<const int N>
void GcCMatMultGcCoeffAddAVX512T(float *y, float alpha,
           const float* A,
           const float* x)
{

	constexpr int NbyTwo = N/2;
	constexpr int TwoN = 2*N;

	for(IndexType col=0; col < NbyTwo; ++col) {

		__m512 alphax_r =  _mm512_set1_ps(alpha*x[2*col] );
		__m512 alphax_i =  _mm512_set1_ps(alpha*x[2*col+1]);
		__m512 malphax_r =_mm512_set1_ps(-alpha*x[2*col]);

		for(IndexType row=0; row < N; row+=16) {
			__m512 A_col = _mm512_load_ps(A + row + TwoN*col );
			__m512 A_perm = _mm512_shuffle_ps( A_col, A_col, _MM_SHUFFLE(2,3,0,1));

			__m512 y_vec = _mm512_load_ps( y + row );

			y_vec = _mm512_fmaddsub_ps( A_col, alphax_r,
					_mm512_fmaddsub_ps( A_perm,alphax_i, y_vec));

			_mm512_store_ps( y + row, y_vec);
		}

		for(IndexType row=N; row < TwoN; row+=16) {
			__m512 A_col = _mm512_load_ps(A + row + TwoN*col );
			__m512 A_perm = _mm512_shuffle_ps( A_col, A_col, _MM_SHUFFLE(2,3,0,1));

			__m512 y_vec = _mm512_load_ps( y + row );
			y_vec=_mm512_fmsubadd_ps( A_col, malphax_r, _mm512_fmsubadd_ps( A_perm, alphax_i, y_vec));
			_mm512_store_ps( y + row, y_vec);
		}
	}

	for(IndexType col=NbyTwo; col < N; ++col) {
		__m512 alphax_r =  _mm512_set1_ps(alpha*x[2*col] );
		__m512 alphax_i =  _mm512_set1_ps(alpha*x[2*col+1]);
		__m512 malphax_r =_mm512_set1_ps(-alpha*x[2*col]);

		for(IndexType row=0; row < N; row+=16) {
			__m512 A_col = _mm512_load_ps(A + row + TwoN*col );
			__m512 A_perm = _mm512_shuffle_ps( A_col, A_col, _MM_SHUFFLE(2,3,0,1));

			__m512 y_vec = _mm512_load_ps( y + row );

			y_vec=_mm512_fmsubadd_ps( A_col, malphax_r, _mm512_fmsubadd_ps( A_perm, alphax_i, y_vec));

			_mm512_store_ps( y + row, y_vec);
		}


		for(IndexType row=N; row < TwoN; row+=16) {
			__m512 A_col = _mm512_load_ps(A + row + TwoN*col );
			__m512 A_perm = _mm512_shuffle_ps( A_col, A_col, _MM_SHUFFLE(2,3,0,1));

			__m512 y_vec = _mm512_load_ps( y + row );

			y_vec = _mm512_fmaddsub_ps( A_col, alphax_r,
					_mm512_fmaddsub_ps( A_perm,alphax_i, y_vec));

			_mm512_store_ps( y + row, y_vec);
		}

	}
}

void GcCMatMultGcCoeffAddAVX512(float* y, float alpha,
				   const float* A,
				   const float* x,
				   IndexType N)
{
	std::complex<float>* yc = reinterpret_cast<std::complex<float>*>(y);
	const std::complex<float>* Ac = reinterpret_cast<const std::complex<float>*>(A);
	const std::complex<float>* xc = reinterpret_cast<const std::complex<float>*>(x);

	if( N == 6 ) {
		GcCMatMultGcCoeffAddNaiveT<6>(yc, alpha, Ac, xc);
	}
	else if( N == 8 ) {
		GcCMatMultGcCoeffAddNaiveT<8>(yc, alpha, Ac, xc);

	}
	else if ( N == 12 ) {
		GcCMatMultGcCoeffAddNaiveT<12>(yc, alpha, Ac, xc);
	}
	else if ( N == 16 ) {
		GcCMatMultGcCoeffAddAVX512T<16>(y, alpha, A,x);
	}
	else if (N == 24 ) {
		GcCMatMultGcCoeffAddNaiveT<24>(yc, alpha, Ac,xc);
	}
	else if (N == 32 ) {
		GcCMatMultGcCoeffAddAVX512T<32>(y, alpha, A,x);
	}
	else if (N == 40 ) {
		GcCMatMultGcCoeffAddNaiveT<40>(yc, alpha, Ac,xc);
	}
	else if (N == 48 ) {
		GcCMatMultGcCoeffAddAVX512T<48>(y, alpha, A, x);
	}
	else if (N == 56 ) {
		GcCMatMultGcCoeffAddNaiveT<56>(yc, alpha, Ac, xc);
	}
	else if (N == 64 ) {
		GcCMatMultGcCoeffAddAVX512T<64>(y, alpha, A, x);
	}
	else {
		MasterLog(ERROR, "Matrix size %d not supported in GcCMatMultGcCoeffAddAVX512", N );
	}
}

#endif



}
