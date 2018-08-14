

#include <cstdio>

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



template<const int N, const int Br, const int Bc>
void CMatMultNaiveBlockedT(std::complex<float>*y,
           const std::complex<float>* A,
           const std::complex<float>* x)
{

	for(int row=0; row < N; ++row) {
		y[row] = std::complex<float>(0,0);
	}

    for(IndexType ocol=0; ocol < N; ocol += Bc) {
      for(IndexType orow=0; orow < N; orow += Br) {

    	  for(IndexType icol=0; icol < Bc; ++icol) {
          int col = ocol+icol;

#pragma omp simd aligned(y,A,x:64)
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
		CMatMultNaiveBlockedT<6,6,6>(yc, Ac, xc);
	}
	else if( N == 8 ) {
		CMatMultNaiveBlockedT<8,8,8>(yc, Ac, xc);

	}
	else if ( N == 12 ) {
		CMatMultNaiveBlockedT<12,6,6>(yc, Ac, xc);
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

template<const int N, const int Br, const int Bc>
void CMatMultNaiveAddBlockedT(std::complex<float>*y,
           const std::complex<float>* A,
           const std::complex<float>* x)
{

  for(IndexType ocol=0; ocol < N; ocol += Bc) {
    for(IndexType orow=0; orow < N; orow += Br) {

      for(IndexType icol=0; icol < Bc; ++icol) {
        IndexType col=ocol + icol;

#pragma omp simd aligned(y,A,x: 64)
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
		CMatMultNaiveAddBlockedT<6,6,6>(yc, Ac, xc);
	}
	else if( N == 8 ) {
		CMatMultNaiveAddBlockedT<8,8,8>(yc, Ac, xc);
	}
	else if( N == 12 ) {
		CMatMultNaiveAddBlockedT<12,6,6>(yc, Ac, xc);
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


template<const int N, const int Br, const int Bc>
void CMatMultNaiveCoeffAddBlockedT(std::complex<float>*y,
		   const float alpha,
           const std::complex<float>* A,
           const std::complex<float>* x)
{

  for(IndexType ocol=0; ocol < N; ocol += Bc) {
    for(IndexType orow=0; orow < N; orow += Br) {

      for(IndexType icol=0; icol < Bc; ++icol) {
        IndexType col=ocol + icol;

        std::complex<float> tmp[Br] __attribute__((aligned(64)));
#pragma omp simd aligned(tmp: 64)
        for(IndexType irow=0; irow < Br; ++irow) {
        	tmp[irow] = std::complex<float>(0,0);
        }

#pragma omp simd aligned(A,x,tmp: 64)
        for(IndexType irow=0; irow < Br; ++irow) {
          IndexType row = orow + irow;

          // NB: These are complex multiplies
          tmp[irow] += A[ row  +  N*col  ] * x[ col ];
        }

#pragma omp simd aligned(y,tmp: 64)
        for(IndexType irow=0; irow < Br; ++irow) {
          IndexType row = orow + irow;

          // NB: These are complex multiplies
          y[row] += alpha*tmp[irow];
        }

      }
    }
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
		CMatMultNaiveCoeffAddBlockedT<6,6,6>(yc, alpha, Ac, xc);
	}
	else if( N == 8 ) {
		CMatMultNaiveCoeffAddBlockedT<8,8,8>(yc, alpha, Ac, xc);
	}
	else if( N == 12 ) {
		CMatMultNaiveCoeffAddBlockedT<12,6,6>(yc,alpha, Ac, xc);
	}
	else if ( N == 16 ) {
		CMatMultNaiveCoeffAddBlockedT<16,8,8>(yc,alpha, Ac,xc);
	}
	else if (N == 24 ) {
		CMatMultNaiveCoeffAddBlockedT<24,8,8>(yc,alpha,Ac,xc);
	}
	else if (N == 32 ) {
		CMatMultNaiveCoeffAddBlockedT<32,8,8>(yc,alpha,Ac,xc);
	}
	else if (N == 48 ) {
		CMatMultNaiveCoeffAddBlockedT<48,8,8>(yc,alpha,Ac, xc);
	}
	else if (N == 64 ) {
		CMatMultNaiveCoeffAddBlockedT<64,8,8>(yc, alpha, Ac, xc);
	}
	else {
		MasterLog(ERROR, "Matrix size %d not supported in CMatMultNaiveAdd" , N );
	}
}


template<const int N, const int Br, const int Bc>
void CMatAdjMultNaiveBlockedT(std::complex<float>*y,
		const std::complex<float>* A,
		const std::complex<float>* x)
{

#pragma omp simd aligned(y:64)
	for(int row=0; row < N; ++row) {
		y[row] = std::complex<float>(0,0);
	}

	for(IndexType orow=0; orow < N; orow += Br) {
		for(IndexType ocol=0; ocol < N; ocol += Bc) {

			for(IndexType icol=0; icol < Bc; ++icol) {
				int col = ocol+icol;

#pragma omp simd aligned(y,A,x:64)
				for(IndexType irow=0; irow < Br; ++irow) {
					int row = orow + irow;
					// NB: These are complex multiplies
					y[row] += std::conj(A[ col +  N*row ]) * x[ col ];
				}
			}
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
		CMatAdjMultNaiveBlockedT<6,6,6>(yc, Ac, xc);
	}
	else if( N == 8 ) {
		CMatAdjMultNaiveBlockedT<8,8,8>(yc, Ac, xc);

	}
	else if ( N == 12 ) {
		CMatAdjMultNaiveBlockedT<12,6,6>(yc, Ac, xc);
	}
	else if ( N == 16 ) {
		CMatAdjMultNaiveBlockedT<16,8,8>(yc, Ac,xc);
	}
	else if (N == 24 ) {
		CMatAdjMultNaiveBlockedT<24,8,8>(yc, Ac,xc);
	}
	else if (N == 32 ) {
		CMatAdjMultNaiveBlockedT<32,8,8>(yc, Ac,xc);
	}
	else if (N == 48 ) {
		CMatAdjMultNaiveBlockedT<48,8,8>(yc, Ac, xc);
	}
	else if (N == 64 ) {
		CMatAdjMultNaiveBlockedT<64,8,8>(yc, Ac, xc);
	}
	else {
		MasterLog(ERROR, "Matrix size %d not supported in CMatMultNaive", N );
	}
}

template<const int N, const int Br, const int Bc>
void CMatAdjMultNaiveAddBlockedT(std::complex<float>*y,
		const std::complex<float>* A,
		const std::complex<float>* x)
{

	for(IndexType orow=0; orow < N; orow += Br) {
		for(IndexType ocol=0; ocol < N; ocol += Bc) {

			for(IndexType icol=0; icol < Bc; ++icol) {
				int col = ocol+icol;

#pragma omp simd aligned(y,A,x:64)
				for(IndexType irow=0; irow < Br; ++irow) {
					int row = orow + irow;
					// NB: These are complex multiplies
					y[row] += std::conj(A[ col +  N*row ]) * x[ col ];
				}
			}
		}
	}

}

void CMatAdjMultNaiveAdd(float* y,
				   const float* A,
				   const float* x,
				   IndexType N)
{
	std::complex<float>* yc = reinterpret_cast<std::complex<float>*>(y);
	const std::complex<float>* Ac = reinterpret_cast<const std::complex<float>*>(A);
	const std::complex<float>* xc = reinterpret_cast<const std::complex<float>*>(x);

	if( N == 6 ) {
		CMatAdjMultNaiveAddBlockedT<6,6,6>(yc, Ac, xc);
	}
	else if( N == 8 ) {
		CMatAdjMultNaiveAddBlockedT<8,8,8>(yc, Ac, xc);

	}
	else if ( N == 12 ) {
		CMatAdjMultNaiveAddBlockedT<12,6,6>(yc, Ac, xc);
	}
	else if ( N == 16 ) {
		CMatAdjMultNaiveAddBlockedT<16,8,8>(yc, Ac,xc);
	}
	else if (N == 24 ) {
		CMatAdjMultNaiveAddBlockedT<24,8,8>(yc, Ac,xc);
	}
	else if (N == 32 ) {
		CMatAdjMultNaiveAddBlockedT<32,8,8>(yc, Ac,xc);
	}
	else if (N == 48 ) {
		CMatAdjMultNaiveAddBlockedT<48,8,8>(yc, Ac, xc);
	}
	else if (N == 64 ) {
		CMatAdjMultNaiveAddBlockedT<64,8,8>(yc, Ac, xc);
	}
	else {
		MasterLog(ERROR, "Matrix size %d not supported in CMatMultNaive", N );
	}
}

template<const int N, const int Br, const int Bc>
void CMatAdjMultNaiveCoeffAddBlockedT(std::complex<float>*y,
		const float alpha,
		const std::complex<float>* A,
		const std::complex<float>* x)
{

	for(IndexType orow=0; orow < N; orow += Br) {
		for(IndexType ocol=0; ocol < N; ocol += Bc) {

			for(IndexType icol=0; icol < Bc; ++icol) {
				int col = ocol+icol;

#pragma omp simd aligned(y,A,x:64)
				for(IndexType irow=0; irow < Br; ++irow) {
					int row = orow + irow;
					// NB: These are complex multiplies
					y[row] += alpha*std::conj(A[ col +  N*row ]) * x[ col ];
				}
			}
		}
	}

}

void CMatAdjMultNaiveCoeffAdd(float* y,
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
		CMatAdjMultNaiveCoeffAddBlockedT<6,6,6>(yc, alpha, Ac, xc);
	}
	else if( N == 8 ) {
		CMatAdjMultNaiveCoeffAddBlockedT<8,8,8>(yc, alpha, Ac, xc);
	}
	else if( N == 12 ) {
		CMatAdjMultNaiveCoeffAddBlockedT<12,6,6>(yc,alpha, Ac, xc);
	}
	else if ( N == 16 ) {
		CMatAdjMultNaiveCoeffAddBlockedT<16,8,8>(yc,alpha, Ac,xc);
	}
	else if (N == 24 ) {
		CMatAdjMultNaiveCoeffAddBlockedT<24,8,8>(yc,alpha,Ac,xc);
	}
	else if (N == 32 ) {
		CMatAdjMultNaiveCoeffAddBlockedT<32,8,8>(yc,alpha,Ac,xc);
	}
	else if (N == 48 ) {
		CMatAdjMultNaiveCoeffAddBlockedT<48,8,8>(yc,alpha,Ac, xc);
	}
	else if (N == 64 ) {
		CMatAdjMultNaiveCoeffAddBlockedT<64,8,8>(yc, alpha, Ac, xc);
	}
	else {
		MasterLog(ERROR, "Matrix size %d not supported in CMatMultNaiveAdd" , N );
	}
}

}
