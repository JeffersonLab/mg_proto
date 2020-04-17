

#include <cstdio>

#include "MG_config.h"
#include "lattice/cmat_mult.h"
#include "utils/memory.h"
#include "utils/print_utils.h"
#include <complex>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <cassert>

namespace MG {

namespace {
	typedef int LAPACK_BLASINT;
#ifndef MGPROTO_USE_CBLAS
	extern "C" void cgemm_(const char *transa, const char *transb,
			LAPACK_BLASINT *m, LAPACK_BLASINT *n,
			LAPACK_BLASINT *k, std::complex<float> *alpha,
			const std::complex<float> *a, LAPACK_BLASINT *lda,
			const std::complex<float> *b, LAPACK_BLASINT *ldb,
			std::complex<float> *beta, std::complex<float> *c,
			LAPACK_BLASINT *ldc);
#else
	#include <cblas.h>
	CBLAS_TRANSPOSE toTrans(const char *trans) {
		const char t = *trans;
		if (t == 'n' || t == 'N') return CblasNoTrans;
		if (t == 't' || t == 'T') return CblasTrans;
		if (t == 'c' || t == 'C') return CblasConjTrans;
	}
#endif

	void XGEMM(const char *transa, const char *transb, LAPACK_BLASINT m,
			LAPACK_BLASINT n, LAPACK_BLASINT k, std::complex<float> alpha,
			const std::complex<float> *a, LAPACK_BLASINT lda, const std::complex<float> *b,
			LAPACK_BLASINT ldb, std::complex<float> beta, std::complex<float> *c,
			LAPACK_BLASINT ldc) {
		assert(c != a && c != b);
#ifndef MGPROTO_USE_CBLAS
		cgemm_(transa, transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
#else
		cblas_cgemm(CblasColMajor, toTrans(transa), toTrans(transb), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
#endif
	}
}

void CMatMultNaive(float* y,
				   const float* A,
				   const float* x,
				   IndexType N,
				   IndexType ncol)
{
	CMatMultCoeffAddNaive(0.0, y, 1.0, A, x, N, ncol);
}

void CMatMultAddNaive(float* y,
		const float* A,
		const float* x,
		IndexType N,
		IndexType ncol)
{
	CMatMultCoeffAddNaive(1.0, y, 1.0, A, x, N, ncol);
}

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
		IndexType xcols)
{
	if (fabs(beta) == 0.0) {
		for (IndexType j=0; j < xcols; ++j) 
			for (IndexType i=0; i < Arows; ++i) y[i + ldy*j] = 0.0;
	}

	XGEMM("N", "N", Arows, xcols, Acols, alpha, A, ldA, x, ldx, beta, y, ldy);
}


void CMatMultCoeffAddNaive(float beta,
		float* y,
		float alpha,
		const float* A,
		const float* x,
		IndexType N,
		IndexType ncol)
{
	if (fabs(beta) == 0.0) for (IndexType i=0; i < N*ncol*2; ++i) y[i] = 0.0;

	// Pretend these are arrays of complex numbers
	std::complex<float>* yc = reinterpret_cast<std::complex<float>*>(y);
	const std::complex<float>* Ac = reinterpret_cast<const std::complex<float>*>(A);
	const std::complex<float>* xc = reinterpret_cast<const std::complex<float>*>(x);

	XGEMM("N", "N", N, ncol, N, alpha, Ac, N, xc, N, beta, yc, N);
}


void CMatAdjMultNaive(float* y,
				   const float* A,
				   const float* x,
				   IndexType N,
					IndexType ncol)
{
	for (IndexType i=0; i < N*ncol*2; ++i) y[i] = 0.0;

	std::complex<float>* yc = reinterpret_cast<std::complex<float>*>(y);
	const std::complex<float>* Ac = reinterpret_cast<const std::complex<float>*>(A);
	const std::complex<float>* xc = reinterpret_cast<const std::complex<float>*>(x);

	XGEMM("C", "N", N, ncol, N, 1.0, Ac, N, xc, N, 0.0, yc, N);
}

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
		IndexType xcols)
{
	if (fabs(beta) == 0.0) {
		for (IndexType j=0; j < xcols; ++j) 
			for (IndexType i=0; i < Acols; ++i) y[i + ldy*j] = 0.0;
	}

	XGEMM("C", "N", Acols, xcols, Arows, alpha, A, ldA, x, ldx, beta, y, ldy);
}


void GcCMatMultGcNaive(float* y,
				   const float* A,
				   const float* x,
				   IndexType N,
					IndexType ncol)
{
	GcCMatMultGcCoeffAddNaive(0.0, y, 1.0, A, x, N, ncol);
}


void GcCMatMultGcCoeffAddNaive(float beta, float* y, float alpha,
				   const float* A,
				   const float* x,
				   IndexType N,
					IndexType ncol)
{
	if (fabs(beta) == 0.0) for (IndexType i=0; i < N*ncol*2; ++i) y[i] = 0.0;

	std::complex<float>* yc = reinterpret_cast<std::complex<float>*>(y);
	const std::complex<float>* Ac = reinterpret_cast<const std::complex<float>*>(A);
	const std::complex<float>* xc = reinterpret_cast<const std::complex<float>*>(x);

	XGEMM("N", "N", N/2, ncol, N/2,  alpha,  Ac,            N,  xc,      N, beta,  yc,      N);
	XGEMM("N", "N", N/2, ncol, N/2, -alpha, &Ac[N*N/2],     N, &xc[N/2], N, 1.0,   yc,      N);
	XGEMM("N", "N", N/2, ncol, N/2, -alpha, &Ac[N/2],       N, xc,       N, beta, &yc[N/2], N);
	XGEMM("N", "N", N/2, ncol, N/2,  alpha, &Ac[N*N/2+N/2], N, &xc[N/2], N, 1.0,  &yc[N/2], N);
}

}
