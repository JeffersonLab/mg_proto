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
	res += a*b; // Complex Multiplication
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexPeq(MGComplex<T>& res, const MGComplex<T>& a)
{
	res += a; // Complex Multiplication
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexConjMadd(MGComplex<T>& res, const MGComplex<T>& a, const MGComplex<T>& b)
{
	res += Kokkos::conj(a)*b; // Complex Multiplication
}
template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_B( MGComplex<T>& res, const MGComplex<T>& a, const T& sign, const MGComplex<T>& b)
{
	res.real() = a.real() + sign*b.real();
	res.imag() = a.imag() + sign*b.imag();
}


template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_iB( MGComplex<T>& res, const MGComplex<T>& a, const T& sign, const MGComplex<T>& b)
{
	res.real() = a.real()-sign*b.imag();
	res.imag() = a.imag()+sign*b.real();
}


// a = -i b
template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_miB( MGComplex<T>& a, const T& sign, const MGComplex<T>& b)
{
	a.real() += sign*b.imag();
	a.imag() -= sign*b.real();
}


// a = b
template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_B( MGComplex<T>& a, const T& sign, const MGComplex<T>& b)
{
	a.real() += sign*b.real();
	a.imag() += sign*b.imag();
}


} // namespace



#endif /* TEST_KOKKOS_KOKKOS_OPS_H_ */
