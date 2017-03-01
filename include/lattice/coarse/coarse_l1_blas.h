/*
 * coarse_l1_blas.h
 *
 *  Created on: Dec 12, 2016
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_COARSE_COARSE_L1_BLAS_H_
#define INCLUDE_LATTICE_COARSE_COARSE_L1_BLAS_H_

#include <complex>
#include "lattice/coarse/coarse_types.h"
using namespace MG;

namespace MG {

// x = x - y; followed by || x ||
double XmyNorm2Vec(CoarseSpinor& x, const CoarseSpinor& y);
double Norm2Vec(const CoarseSpinor& x);
std::complex<double> InnerProductVec(const CoarseSpinor& x, const CoarseSpinor& y);

void ZeroVec(CoarseSpinor& x);
void CopyVec(CoarseSpinor& x, const CoarseSpinor& y);
void ScaleVec(const float alpha, CoarseSpinor& x);
void ScaleVec(const std::complex<float>& alpha, CoarseSpinor& x);
void AxpyVec(const std::complex<float>& alpha, const CoarseSpinor& x, CoarseSpinor& y);
void YpeqxVec(const CoarseSpinor& x, CoarseSpinor& y);
void YmeqxVec(const CoarseSpinor& x, CoarseSpinor& y);
void AxpyVec(const float& alpha, const CoarseSpinor&x, CoarseSpinor& y);
void XmyzVec(const CoarseSpinor& x, const CoarseSpinor& y, CoarseSpinor& z);
void Gaussian(CoarseSpinor& v);
void ZeroGauge(CoarseGauge& gauge);

void BiCGStabPUpdate(const std::complex<float>& beta,
					 const CoarseSpinor& r,
					 const std::complex<float>& omega,
					 const CoarseSpinor& v,
					 CoarseSpinor& p);

void BiCGStabXUpdate(const std::complex<float>& omega,
					 const CoarseSpinor& r,
					 const std::complex<float>& alpha,
					 const CoarseSpinor& p,
					 CoarseSpinor& x);

}


#endif /* TEST_QDPXX_COARSE_L1_BLAS_H_ */
