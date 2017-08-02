/*
 * kokkos_ops.h
 *
 *  Created on: Jul 26, 2017
 *      Author: bjoo
 */

#ifndef TEST_KOKKOS_KOKKOS_OPS_H_
#define TEST_KOKKOS_KOKKOS_OPS_H_

#include "my_complex.h"

namespace MG
{

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void ComplexCopy(MyComplex<T>& result, const MyComplex<T>& source)
{
	result=source;
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void ComplexZero(MyComplex<T>& result)
{
	result = MyComplex<T>(0,0);
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void Load(MyComplex<T>& result, const MyComplex<T>& source)
{
	result = source;
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void Store(MyComplex<T>& result, const MyComplex<T>& source)
{
	result = source;
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void Stream(MyComplex<T>& result, const MyComplex<T>& source)
{
	result = source;
}


template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexCMadd(MyComplex<T>& res, const MyComplex<T>& a, const MyComplex<T>& b)
{
	res += a*b; // Complex Multiplication
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexPeq(MyComplex<T>& res, const MyComplex<T>& a)
{
	res += a; // Complex Multiplication
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexConjMadd(MyComplex<T>& res, const MyComplex<T>& a, const MyComplex<T>& b)
{
	res += conj(a)*b; // Complex Multiplication
}
template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_B( MyComplex<T>& res, const MyComplex<T>& a, const T& sign, const MyComplex<T>& b)
{
	res.real() = a.real() + sign*b.real();
	res.imag() = a.imag() + sign*b.imag();
}


template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_iB( MyComplex<T>& res, const MyComplex<T>& a, const T& sign, const MyComplex<T>& b)
{
	res.real() =  a.real()-sign*b.imag();
	res.imag() =  a.imag()+sign*b.real();
}


// a = -i b
template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_miB( MyComplex<T>& a, const T& sign, const MyComplex<T>& b)
{
	a.real() += sign*b.imag();
	a.imag() -= sign*b.real();
}


// a = b
template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_B( MyComplex<T>& a, const T& sign, const MyComplex<T>& b)
{
	a.real() += sign*b.real();
	a.imag() += sign*b.imag();
}


} // namespace



#endif /* TEST_KOKKOS_KOKKOS_OPS_H_ */
