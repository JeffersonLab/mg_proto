/*
 * qphix_blas_wrappers.h
 *
 *  Created on: Oct 17, 2017
 *      Author: bjoo
 */

#include "lattice/qphix/qphix_types.h"
#include "lattice/qphix/qphix_qdp_utils.h"
#include "lattice/qphix/qphix_blas_wrappers.h"
#include <qphix/blas_full_spinor.h>

using namespace QPhiX;

namespace MG
{
// x = x - y; followed by || x ||
double XmyNorm2Vec(QPhiXSpinor& x, const QPhiXSpinor& y)
{
  double ret_norm;
  const Geom& geom = y.getGeom();
  int n_blas_simt = geom.getNSIMT();
  xmy2Norm2Spinor<>( y.get(), x.get(), ret_norm,  geom, n_blas_simt);
  return ret_norm;

}

double Norm2Vec(const QPhiXSpinor& x)
{
  double ret_norm = 0;
  const Geom& geom = x.getGeom();
  int n_blas_simt = geom.getNSIMT();
  norm2Spinor(ret_norm, x.get(), geom,n_blas_simt);
  return ret_norm;

}

std::complex<double> InnerProductVec(const QPhiXSpinor& x, const QPhiXSpinor& y)
{
  double result[2];
  const Geom& geom = y.getGeom();
  int n_blas_simt = geom.getNSIMT();

  innerProductSpinor(result,x.get(),y.get(),geom, n_blas_simt);

  std::complex<double> ret_val(result[0],result[1]);
  return ret_val;
}
void ZeroVec(QPhiXSpinor& x)
{
  const Geom& geom = x.getGeom();
  int n_blas_simt = geom.getNSIMT();
  zeroSpinor(x.get(),geom,n_blas_simt);
}

void CopyVec(QPhiXSpinor& x, const QPhiXSpinor& y)
{
  const Geom& geom = y.getGeom();
  int n_blas_simt = geom.getNSIMT();
  copySpinor(x.get(),y.get(),geom,n_blas_simt);
}

void AxpyVec(const std::complex<float>& alpha, const QPhiXSpinor& x, QPhiXSpinor& y)
{
    const Geom& geom = y.getGeom();
   int n_blas_simt = geom.getNSIMT();

   double a[2] = { std::real(alpha), std::imag(alpha) };
   caxpySpinor(a, x.get(), y.get(), geom,n_blas_simt);
}

void AxpyVec(const std::complex<double>& alpha, const QPhiXSpinor& x, QPhiXSpinor& y)
{
  const Geom& geom = y.getGeom();
  int n_blas_simt = geom.getNSIMT();
  double a[2] = { std::real(alpha), std::imag(alpha) };
  caxpySpinor(a, x.get(), y.get(), geom,n_blas_simt);

}
void Gaussian(QPhiXSpinor& v)
{
  LatticeFermion x; gaussian(x);
  QDPSpinorToQPhiXSpinor(x,v);

}


}

