/*
 * qphix_blas_wrappers.h
 *
 *  Created on: Oct 17, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_QPHIX_QPHIX_BLAS_WRAPPERS_H_
#define INCLUDE_LATTICE_QPHIX_QPHIX_BLAS_WRAPPERS_H_

#include "lattice/qphix/qphix_types.h"
#include "lattice/coarse/subset.h"
namespace MG
{

// x = x - y; followed by || x ||
double XmyNorm2Vec(QPhiXSpinor& x, const QPhiXSpinor& y, const CBSubset& subset = SUBSET_ALL);
double Norm2Vec(const QPhiXSpinor& x, const CBSubset& subset = SUBSET_ALL);
std::complex<double> InnerProductVec(const QPhiXSpinor& x, const QPhiXSpinor& y,const CBSubset& subset = SUBSET_ALL);

void ZeroVec(QPhiXSpinor& x, const CBSubset& subset = SUBSET_ALL);
void CopyVec(QPhiXSpinor& x, const QPhiXSpinor& y, const CBSubset& subset = SUBSET_ALL);
void AxVec(const double alpha, QPhiXSpinor& x,const CBSubset& subset = SUBSET_ALL);
void AxpyVec(const std::complex<float>& alpha, const QPhiXSpinor& x, QPhiXSpinor& y,const CBSubset& subset = SUBSET_ALL);
void AxpyVec(const std::complex<double>& alpha, const QPhiXSpinor& x, QPhiXSpinor& y,const CBSubset& subset = SUBSET_ALL);
void AxpyVec(const double alpha, const QPhiXSpinor& x, QPhiXSpinor& y, const CBSubset& subset = SUBSET_ALL);
void Gaussian(QPhiXSpinor& v,const CBSubset& subset = SUBSET_ALL);
void YpeqXVec(const QPhiXSpinor& x, QPhiXSpinor& y,const CBSubset& subset = SUBSET_ALL);
void YmeqXVec(const QPhiXSpinor& x, QPhiXSpinor& y,const CBSubset& subset = SUBSET_ALL);

 // do we need these just now?
double XmyNorm2Vec(QPhiXSpinorF& x, const QPhiXSpinorF& y,const CBSubset& subset = SUBSET_ALL);
double Norm2Vec(const QPhiXSpinorF& x,const CBSubset& subset = SUBSET_ALL);
std::complex<double> InnerProductVec(const QPhiXSpinorF& x, const QPhiXSpinorF& y,const CBSubset& subset = SUBSET_ALL);

void ZeroVec(QPhiXSpinorF& x,const CBSubset& subset = SUBSET_ALL);

void CopyVec(QPhiXSpinorF& x, const QPhiXSpinorF& y,const CBSubset& subset = SUBSET_ALL);
void AxVec(const double alpha, QPhiXSpinorF& x,const CBSubset& subset = SUBSET_ALL);
void AxpyVec(const std::complex<float>& alpha, const QPhiXSpinorF& x, QPhiXSpinorF& y,const CBSubset& subset = SUBSET_ALL);
void AxpyVec(const std::complex<double>& alpha, const QPhiXSpinorF& x, QPhiXSpinorF& y,const CBSubset& subset = SUBSET_ALL);
void AxpyVec(const double alpha, const QPhiXSpinorF& x, QPhiXSpinorF& y,const CBSubset& subset = SUBSET_ALL);
void Gaussian(QPhiXSpinorF& v,const CBSubset& subset = SUBSET_ALL);
void YpeqXVec(const QPhiXSpinorF& x, QPhiXSpinorF& y,const CBSubset& subset = SUBSET_ALL);
void YmeqXVec(const QPhiXSpinorF& x, QPhiXSpinorF& y,const CBSubset& subset = SUBSET_ALL);

// Use overloading
void ConvertSpinor(const QPhiXSpinor& in, QPhiXSpinorF& out, const CBSubset& subset = SUBSET_ALL);
void ConvertSpinor(const QPhiXSpinorF& in, QPhiXSpinor& out, const CBSubset& subset = SUBSET_ALL);

}



#endif /* INCLUDE_LATTICE_QPHIX_QPHIX_BLAS_WRAPPERS_H_ */
