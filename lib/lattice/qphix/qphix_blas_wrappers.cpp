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
#include "lattice/coarse/subset.h"

using namespace QPhiX;

namespace MG
{


// x = x - y; followed by || x ||
template<typename ST>
inline
double XmyNorm2VecT(ST& x, const ST& y, const CBSubset& subset)
{
  double ret_norm;
  const typename ST::GeomT& geom = y.getGeom();
  int n_blas_simt = geom.getNSIMT();
  xmy2Norm2Spinor<>( y.get(), x.get(), ret_norm,  geom, n_blas_simt, subset.start, subset.end);
  return ret_norm;
}


double XmyNorm2Vec(QPhiXSpinor& x, const QPhiXSpinor& y, const CBSubset& subset )
{
  return XmyNorm2VecT(x,y,subset);
}

double XmyNorm2Vec(QPhiXSpinorF& x, const QPhiXSpinorF& y,const CBSubset& subset)
{
  return XmyNorm2VecT(x,y,subset);
}


template<typename ST>
inline
double Norm2VecT(const ST& x, const CBSubset& subset)
{
  double ret_norm = 0;
  const typename ST::GeomT& geom = x.getGeom();
  int n_blas_simt = geom.getNSIMT();
  norm2Spinor(ret_norm, x.get(), geom,n_blas_simt, subset.start, subset.end);
  return ret_norm;
}

double
Norm2Vec(const QPhiXSpinor& x, const CBSubset& subset)
{
  return Norm2VecT(x, subset);
}

double
Norm2Vec(const QPhiXSpinorF& x, const CBSubset& subset)
{
  return Norm2VecT(x, subset);
}



template<typename ST>
inline
std::complex<double> InnerProductVecT(const ST& x, const ST& y, const CBSubset& subset)
{
  double result[2];
  const typename ST::GeomT& geom = y.getGeom();
  int n_blas_simt = geom.getNSIMT();

  innerProductSpinor(result,x.get(),y.get(),geom, n_blas_simt, subset.start, subset.end);

  std::complex<double> ret_val(result[0],result[1]);
  return ret_val;
}

std::complex<double> InnerProductVec(const QPhiXSpinor& x, const QPhiXSpinor& y, const CBSubset& subset )
{
  return InnerProductVecT(x,y, subset);
}

std::complex<double> InnerProductVec(const QPhiXSpinorF& x, const QPhiXSpinorF& y, const CBSubset& subset )
{
  return InnerProductVecT(x,y, subset);
}

template<typename ST>
void ZeroVecT(ST& x, const CBSubset& subset)
{
  const typename ST::GeomT& geom = x.getGeom();
  int n_blas_simt = geom.getNSIMT();
  zeroSpinor(x.get(),geom,n_blas_simt, subset.start, subset.end);
}

void ZeroVec(QPhiXSpinor& x, const CBSubset& subset ) { ZeroVecT(x,subset); }

void ZeroVec(QPhiXSpinorF& x, const CBSubset& subset) { ZeroVecT(x,subset); }

template<typename ST>
inline
void CopyVecT(ST& x, const ST& y,const CBSubset& subset)
{
  const typename ST::GeomT& geom = y.getGeom();
  int n_blas_simt = geom.getNSIMT();
  copySpinor(x.get(),y.get(),geom,n_blas_simt, subset.start, subset.end);
}

void CopyVec(QPhiXSpinor& x, const QPhiXSpinor& y, const CBSubset& subset ) { CopyVecT(x,y, subset); }
void CopyVec(QPhiXSpinorF& x, const QPhiXSpinorF& y, const CBSubset& subset) { CopyVecT(x,y, subset); }


template<typename ST>
inline
void AxVecT(const double alpha, ST& x, const CBSubset& subset )
{
  const typename ST::GeomT& geom = x.getGeom();
  int n_blas_simt = geom.getNSIMT();
  axSpinor<>(alpha, x.get(),geom,n_blas_simt, subset.start, subset.end);
}

void AxVec(const double alpha, QPhiXSpinor& x,const CBSubset& subset) { AxVecT(alpha,x,subset); }
void AxVec(const double alpha, QPhiXSpinorF& x, const CBSubset& subset) { AxVecT(alpha,x,subset); }


template<typename ST>
inline
void AxpyVecT(const double alpha, const ST& x, ST& y, const CBSubset& subset )
{
  const typename ST::GeomT& geom = y.getGeom();
  int n_blas_simt = geom.getNSIMT();

  double a[2] = { std::real(alpha), std::imag(alpha) };
  caxpySpinor(a, x.get(), y.get(), geom,n_blas_simt, subset.start, subset.end);
}

void AxpyVec(const double alpha, const QPhiXSpinor& x, QPhiXSpinor& y,const CBSubset& subset)
{
   AxpyVecT(alpha,x,y,subset);

}

void AxpyVec(const double alpha, const QPhiXSpinorF& x, QPhiXSpinorF& y, const CBSubset& subset )
{
  AxpyVecT(alpha,x,y,subset);
}

template<typename ST>
inline
void AxpyVecT(const std::complex<float>& alpha, const ST& x, ST& y, const CBSubset& subset )
{
  const typename ST::GeomT& geom = y.getGeom();
  int n_blas_simt = geom.getNSIMT();

  double a[2] = { std::real(alpha), std::imag(alpha) };
  caxpySpinor(a, x.get(), y.get(), geom,n_blas_simt, subset.start, subset.end);
}


void AxpyVec(const std::complex<float>& alpha, const QPhiXSpinor& x, QPhiXSpinor& y, const CBSubset& subset )
{
  AxpyVecT(alpha,x,y,subset);
}

void AxpyVec(const std::complex<float>& alpha, const QPhiXSpinorF& x, QPhiXSpinorF& y, const CBSubset& subset)
{
  AxpyVecT(alpha,x,y,subset);
}

template<typename ST>
inline
void AxpyVecT(const std::complex<double>& alpha, const ST& x, ST& y, const CBSubset& subset)
{
  const typename ST::GeomT& geom = y.getGeom();
  int n_blas_simt = geom.getNSIMT();
  double a[2] = { std::real(alpha), std::imag(alpha) };
  caxpySpinor(a, x.get(), y.get(), geom,n_blas_simt,subset.start, subset.end);

}

void AxpyVec(const std::complex<double>& alpha, const QPhiXSpinor& x, QPhiXSpinor& y, const CBSubset& subset )
{
  AxpyVecT(alpha,x,y, subset);
}

void AxpyVec(const std::complex<double>& alpha, const QPhiXSpinorF& x, QPhiXSpinorF& y, const CBSubset& subset)
{
  AxpyVecT(alpha,x,y, subset);
}



void Gaussian(QPhiXSpinor& v,const CBSubset& subset)
{
  LatticeFermion x; gaussian(x);
  QDPSpinorToQPhiXSpinor(x,v,subset);

}
void Gaussian(QPhiXSpinorF& v,const CBSubset& subset )
{
  LatticeFermion x; gaussian(x);
  QDPSpinorToQPhiXSpinor(x,v,subset);

}

template<typename S1, typename S2>
inline
void ConvertSpinorT(const S1& in, S2& out, const CBSubset& subset)
{
  const typename S1::GeomT& geom_in = in.getGeom();
  const typename S2::GeomT& geom_out = out.getGeom();
  const double scale_factor = 1;
  const int n_blas_threads = geom_out.getNSIMT();

  for(int cb=subset.start; cb < subset.end; ++cb) {
    // QPhiX conversions
    convert( out.getCB(cb).get(), scale_factor, in.getCB(cb).get(),
        geom_out, geom_in,n_blas_threads);

  }
}

void ConvertSpinor(const QPhiXSpinor& in, QPhiXSpinorF& out, const CBSubset& subset)
{
  ConvertSpinorT(in,out,subset);
}

void ConvertSpinor(const QPhiXSpinorF& in, QPhiXSpinor& out, const CBSubset& subset)
{
  ConvertSpinorT(in,out,subset);
}


template<typename ST>
inline
void  YpeqXVecT(const ST& x, ST& y, const CBSubset& subset )
{
  const typename ST::GeomT& geom = y.getGeom();
  int n_blas_simt = geom.getNSIMT();
  ypeqxSpinor(x.get(), y.get(), geom,n_blas_simt, subset.start,subset.end);
}

void YpeqXVec(const QPhiXSpinor& x, QPhiXSpinor& y, const CBSubset& subset )
{
  YpeqXVecT(x,y, subset);
}

void YpeqXVec(const QPhiXSpinorF& x, QPhiXSpinorF& y, const CBSubset& subset )
{
  YpeqXVecT(x,y,subset);
}

template<typename ST>
inline
void  YmeqXVecT(const ST& x, ST& y,const CBSubset& subset )
{
  const typename ST::GeomT& geom = y.getGeom();
  int n_blas_simt = geom.getNSIMT();
  ymeqxSpinor(x.get(), y.get(), geom,n_blas_simt, subset.start, subset.end);
}

void YmeqXVec(const QPhiXSpinor& x, QPhiXSpinor& y, const CBSubset& subset )
{
  YmeqXVecT(x,y, subset);
}

void YmeqXVec(const QPhiXSpinorF& x, QPhiXSpinorF& y,const CBSubset& subset )
{
  YmeqXVecT(x,y,subset);
}



} // namespace
