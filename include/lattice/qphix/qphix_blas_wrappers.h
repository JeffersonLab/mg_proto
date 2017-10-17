/*
 * qphix_blas_wrappers.h
 *
 *  Created on: Oct 17, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_QPHIX_QPHIX_BLAS_WRAPPERS_H_
#define INCLUDE_LATTICE_QPHIX_QPHIX_BLAS_WRAPPERS_H_

#include "lattice/qphix/qphix_types.h"

namespace MG
{

// x = x - y; followed by || x ||
double XmyNorm2Vec(QPhiXSpinor& x, const QPhiXSpinor& y);
double Norm2Vec(const QPhiXSpinor& x);
std::complex<double> InnerProductVec(const QPhiXSpinor& x, const QPhiXSpinor& y);

void ZeroVec(QPhiXSpinor& x);
void CopyVec(QPhiXSpinor& x, const QPhiXSpinor& y);
void AxpyVec(const std::complex<float>& alpha, const QPhiXSpinor& x, QPhiXSpinor& y);
void AxpyVec(const std::complex<double>& alpha, const QPhiXSpinor& x, QPhiXSpinor& y);
void Gaussian(QPhiXSpinor& v);

}



#endif /* INCLUDE_LATTICE_QPHIX_QPHIX_BLAS_WRAPPERS_H_ */
