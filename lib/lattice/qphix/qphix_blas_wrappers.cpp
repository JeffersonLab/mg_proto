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
template<typename ST>
inline
double XmyNorm2VecT(ST& x, const ST& y)
{
  double ret_norm;
  const typename ST::GeomT& geom = y.getGeom();
  int n_blas_simt = geom.getNSIMT();
  xmy2Norm2Spinor<>( y.get(), x.get(), ret_norm,  geom, n_blas_simt);
  return ret_norm;
}


double XmyNorm2Vec(QPhiXSpinor& x, const QPhiXSpinor& y)
{
  return XmyNorm2VecT(x,y);
}

double XmyNorm2Vec(QPhiXSpinorF& x, const QPhiXSpinorF& y)
{
  return XmyNorm2VecT(x,y);
}


template<typename ST>
inline
double Norm2VecT(const ST& x)
{
  double ret_norm = 0;
  const typename ST::GeomT& geom = x.getGeom();
  int n_blas_simt = geom.getNSIMT();
  norm2Spinor(ret_norm, x.get(), geom,n_blas_simt);
  return ret_norm;
}

double
Norm2Vec(const QPhiXSpinor& x)
{
  return Norm2VecT(x);
}

double
Norm2Vec(const QPhiXSpinorF& x)
{
  return Norm2VecT(x);
}

template<typename ST>
inline
std::complex<double> InnerProductVecT(const ST& x, const ST& y)
{
  double result[2];
  const typename ST::GeomT& geom = y.getGeom();
  int n_blas_simt = geom.getNSIMT();

  innerProductSpinor(result,x.get(),y.get(),geom, n_blas_simt);

  std::complex<double> ret_val(result[0],result[1]);
  return ret_val;
}

std::complex<double> InnerProductVec(const QPhiXSpinor& x, const QPhiXSpinor& y)
{
  return InnerProductVecT(x,y);
}

std::complex<double> InnerProductVec(const QPhiXSpinorF& x, const QPhiXSpinorF& y)
{
  return InnerProductVecT(x,y);
}

template<typename ST>
void ZeroVecT(ST& x)
{
  const typename ST::GeomT& geom = x.getGeom();
  int n_blas_simt = geom.getNSIMT();
  zeroSpinor(x.get(),geom,n_blas_simt);
}

void ZeroVec(QPhiXSpinor& x) { ZeroVecT(x); }

void ZeroVec(QPhiXSpinorF& x) { ZeroVecT(x); }

template<typename ST>
inline
void CopyVecT(ST& x, const ST& y)
{
  const typename ST::GeomT& geom = y.getGeom();
  int n_blas_simt = geom.getNSIMT();
  copySpinor(x.get(),y.get(),geom,n_blas_simt);
}

void CopyVec(QPhiXSpinor& x, const QPhiXSpinor& y) { CopyVecT(x,y); }
void CopyVec(QPhiXSpinorF& x, const QPhiXSpinorF& y) { CopyVecT(x,y); }


template<typename ST>
inline
void AxVecT(const double alpha, ST& x)
{
  const typename ST::GeomT& geom = x.getGeom();
  int n_blas_simt = geom.getNSIMT();
  axSpinor<>(alpha, x.get(),geom,n_blas_simt);
}

void AxVec(const double alpha, QPhiXSpinor& x) { AxVecT(alpha,x); }
void AxVec(const double alpha, QPhiXSpinorF& x) { AxVecT(alpha,x); }


template<typename ST>
inline
void AxpyVecT(const double alpha, const ST& x, ST& y)
{
  const typename ST::GeomT& geom = y.getGeom();
  int n_blas_simt = geom.getNSIMT();

  double a[2] = { std::real(alpha), std::imag(alpha) };
  caxpySpinor(a, x.get(), y.get(), geom,n_blas_simt);
}

void AxpyVec(const double alpha, const QPhiXSpinor& x, QPhiXSpinor& y)
{
   AxpyVecT(alpha,x,y);

}

void AxpyVec(const double alpha, const QPhiXSpinorF& x, QPhiXSpinorF& y)
{
  AxpyVecT(alpha,x,y);
}

template<typename ST>
inline
void AxpyVecT(const std::complex<float>& alpha, const ST& x, ST& y)
{
  const typename ST::GeomT& geom = y.getGeom();
  int n_blas_simt = geom.getNSIMT();

  double a[2] = { std::real(alpha), std::imag(alpha) };
  caxpySpinor(a, x.get(), y.get(), geom,n_blas_simt);
}


void AxpyVec(const std::complex<float>& alpha, const QPhiXSpinor& x, QPhiXSpinor& y)
{
  AxpyVecT(alpha,x,y);
}

void AxpyVec(const std::complex<float>& alpha, const QPhiXSpinorF& x, QPhiXSpinorF& y)
{
  AxpyVecT(alpha,x,y);
}

template<typename ST>
inline
void AxpyVecT(const std::complex<double>& alpha, const ST& x, ST& y)
{
  const typename ST::GeomT& geom = y.getGeom();
  int n_blas_simt = geom.getNSIMT();
  double a[2] = { std::real(alpha), std::imag(alpha) };
  caxpySpinor(a, x.get(), y.get(), geom,n_blas_simt);

}

void AxpyVec(const std::complex<double>& alpha, const QPhiXSpinor& x, QPhiXSpinor& y)
{
  AxpyVecT(alpha,x,y);
}

void AxpyVec(const std::complex<double>& alpha, const QPhiXSpinorF& x, QPhiXSpinorF& y)
{
  AxpyVecT(alpha,x,y);
}



void Gaussian(QPhiXSpinor& v)
{
  LatticeFermion x; gaussian(x);
  QDPSpinorToQPhiXSpinor(x,v);

}
void Gaussian(QPhiXSpinorF& v)
{
  LatticeFermion x; gaussian(x);
  QDPSpinorToQPhiXSpinor(x,v);

}

template<typename S1, typename S2>
inline
void ConvertSpinorT(const S1& in, S2& out)
{
  const typename S1::GeomT& geom_in = in.getGeom();
  const typename S2::GeomT& geom_out = out.getGeom();
  const double scale_factor = 1;
  const int n_blas_threads = geom_out.getNSIMT();

  for(int cb=0; cb < 2; ++cb) {
    // QPhiX conversions
    convert( out.getCB(cb).get(), scale_factor, in.getCB(cb).get(),
        geom_out, geom_in,n_blas_threads);

  }
}

void ConvertSpinor(const QPhiXSpinor& in, QPhiXSpinorF& out)
{
  ConvertSpinorT(in,out);
}

void ConvertSpinor(const QPhiXSpinorF& in, QPhiXSpinor& out)
{
  ConvertSpinorT(in,out);
}

} // namespace
