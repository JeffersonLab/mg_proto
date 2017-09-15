/*
 * kokkos_ops.h
 *
 *  Created on: Jul 26, 2017
 *      Author: bjoo
 */

#ifndef TEST_KOKKOS_KOKKOS_OPS_H_
#define TEST_KOKKOS_KOKKOS_OPS_H_

#include "kokkos_defaults.h"

namespace MG
{

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void ComplexCopy(MGComplex<T>& result, const MGComplex<T>& source)
{
	result=source;
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void ComplexZero(MGComplex<T>& result)
{
	result = MGComplex<T>(0,0);
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void Load(MGComplex<T>& result, const MGComplex<T>& source)
{
	result = source;
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void Store(MGComplex<T>& result, const MGComplex<T>& source)
{
	result = source;
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void Stream(MGComplex<T>& result, const MGComplex<T>& source)
{
	result = source;
}


template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexCMadd(MGComplex<T>& res, const MGComplex<T>& a, const MGComplex<T>& b)
{
  T res_re=( res.real() + a.real()*b.real() ) - a.imag()*b.imag();
  T res_im=( res.imag() + a.real()*b.imag() ) + a.imag()*b.real();

  res = MGComplex<T>(res_re,res_im);
  //	res += a*b; // Complex Multiplication
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexPeq(MGComplex<T>& res, const MGComplex<T>& a)
{
  T res_re = res.real() + a.real();
  T res_im = res.imag() + a.imag();

  res = MGComplex<T>(res_re,res_im);
  //res += a; // Complex += n
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexConjMadd(MGComplex<T>& res, const MGComplex<T>& a, const MGComplex<T>& b)
{
  T res_re = ( res.real() + a.real()*b.real() ) + a.imag()*b.imag();
  T res_im = ( res.imag() + a.real()*b.imag() ) - a.imag()*b.real();
  res = MGComplex<T>(res_re,res_im);
  //	res += Kokkos::conj(a)*b; // Complex Multiplication
}


template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_B( MGComplex<T>& res, const MGComplex<T>& a, const T& sign, const MGComplex<T>& b)
{
  T res_re = a.real() + sign*b.real();
  T res_im = a.imag() + sign*b.imag();
  res = MGComplex<T>(res_re,res_im);
}

 template<typename T, int sign>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_B( MGComplex<T>& res, const MGComplex<T>& a, const MGComplex<T>& b)
{
  const T fsign = static_cast<const T>(sign);
   T res_re = a.real() + fsign*b.real();
   T res_im = a.imag() + fsign*b.imag();
   res = MGComplex<T>(res_re,res_im);
 }


template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_iB( MGComplex<T>& res, const MGComplex<T>& a, const T& sign, const MGComplex<T>& b)
{
  T res_re =  a.real()-sign*b.imag();
  T res_im =  a.imag()+sign*b.real();
  res = MGComplex<T>(res_re,res_im);
}

 template<typename T, int sign>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_iB( MGComplex<T>& res, const MGComplex<T>& a, const MGComplex<T>& b)
{
  const T fsign=static_cast<const T>(sign);
  T res_re =  a.real()-sign*b.imag();
  T res_im =  a.imag()+sign*b.real();
  res = MGComplex<T>(res_re,res_im);
}



// a = -i b
template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_miB( MGComplex<T>& a, const T& sign, const MGComplex<T>& b)
{
  T res_re = a.real() + sign*b.imag();
  T res_im = a.imag() - sign*b.real();
  a = MGComplex<T>(res_re,res_im);
}

// a = -i b
 template<typename T, int sign>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_miB( MGComplex<T>& a, const MGComplex<T>& b)
{
  const T fsign = static_cast<const T>(sign);
  T res_re = a.real() + fsign*b.imag();
  T res_im = a.imag() - fsign*b.real();
  a = MGComplex<T>(res_re,res_im);
}


// a = b
template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_B( MGComplex<T>& a, const T& sign, const MGComplex<T>& b)
{
  T res_re = a.real() + sign*b.real();
  T res_im = a.imag() + sign*b.imag();
  a = MGComplex<T>(res_re,res_im);
}

// a = b
 template<typename T, int sign>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_B( MGComplex<T>& a, const MGComplex<T>& b)
{

  const T fsign = static_cast<T>(sign);
  T res_re = a.real() + fsign*b.real();
  T res_im = a.imag() + fsign*b.imag();
  a = MGComplex<T>(res_re,res_im);
}


} // namespace



#endif /* TEST_KOKKOS_KOKKOS_OPS_H_ */
