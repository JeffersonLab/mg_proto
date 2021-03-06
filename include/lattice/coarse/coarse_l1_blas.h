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
#include "lattice/coarse/subset.h"
using namespace MG;

namespace MG {

// x = x - y; followed by || x ||
double XmyNorm2Vec(CoarseSpinor& x, const CoarseSpinor& y, const CBSubset& subset=SUBSET_ALL);
double Norm2Vec(const CoarseSpinor& x, const CBSubset& subset = SUBSET_ALL);
std::complex<double> InnerProductVec(const CoarseSpinor& x, const CoarseSpinor& y, const CBSubset& subset = SUBSET_ALL);

void ZeroVec(CoarseSpinor& x, const CBSubset& subset=SUBSET_ALL);
void CopyVec(CoarseSpinor& x, const CoarseSpinor& y, const CBSubset& subset=SUBSET_ALL);
void ScaleVec(const float alpha, CoarseSpinor& x, const CBSubset& subset=SUBSET_ALL);
void ScaleVec(const std::complex<float>& alpha, CoarseSpinor& x, const CBSubset& subset=SUBSET_ALL);
void AxpyVec(const std::complex<float>& alpha, const CoarseSpinor& x, CoarseSpinor& y, const CBSubset& subset=SUBSET_ALL);
void YpeqxVec(const CoarseSpinor& x, CoarseSpinor& y, const CBSubset& subset=SUBSET_ALL);
void YmeqxVec(const CoarseSpinor& x, CoarseSpinor& y,const CBSubset& subset=SUBSET_ALL);
void AxpyVec(const float& alpha, const CoarseSpinor&x, CoarseSpinor& y,const CBSubset& subset=SUBSET_ALL);
void XmyzVec(const CoarseSpinor& x, const CoarseSpinor& y, CoarseSpinor& z,const CBSubset& subset=SUBSET_ALL);
void Gaussian(CoarseSpinor& v, const CBSubset& subset=SUBSET_ALL);
void ZeroGauge(CoarseGauge& gauge);

void BiCGStabPUpdate(const std::complex<float>& beta,
					 const CoarseSpinor& r,
					 const std::complex<float>& omega,
					 const CoarseSpinor& v,
					 CoarseSpinor& p, const CBSubset& subset=SUBSET_ALL);

void BiCGStabXUpdate(const std::complex<float>& omega,
					 const CoarseSpinor& r,
					 const std::complex<float>& alpha,
					 const CoarseSpinor& p,
					 CoarseSpinor& x, const CBSubset& subset=SUBSET_ALL);


}


#endif /* TEST_QDPXX_COARSE_L1_BLAS_H_ */
