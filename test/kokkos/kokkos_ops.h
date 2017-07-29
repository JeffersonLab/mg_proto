/*
 * kokkos_ops.h
 *
 *  Created on: Jul 26, 2017
 *      Author: bjoo
 */

#ifndef TEST_KOKKOS_KOKKOS_OPS_H_
#define TEST_KOKKOS_KOKKOS_OPS_H_

#include <Kokkos_Complex.hpp>
namespace MG
{

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void ComplexCopy(Kokkos::complex<T>& result, const Kokkos::complex<T>& source)
{
	result=source;
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void ComplexZero(Kokkos::complex<T>& result)
{
	result = Kokkos::complex<T>(0,0);
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexCMadd(Kokkos::complex<T>& res, const Kokkos::complex<T>& a, const Kokkos::complex<T>& b)
{
	res += a*b; // Complex Multiplication
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexPeq(Kokkos::complex<T>& res, const Kokkos::complex<T>& a)
{
	res += a; // Complex Multiplication
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexConjMadd(Kokkos::complex<T>& res, const Kokkos::complex<T>& a, const Kokkos::complex<T>& b)
{
	res += Kokkos::conj(a)*b; // Complex Multiplication
}
template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_B( Kokkos::complex<T>& res, const Kokkos::complex<T>& a, const T& sign, const Kokkos::complex<T>& b)
{
	res.real() = a.real() + sign*b.real();
	res.imag() = a.imag() + sign*b.imag();
}


template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_iB( Kokkos::complex<T>& res, const Kokkos::complex<T>& a, const T& sign, const Kokkos::complex<T>& b)
{
	res.real() = a.real()-sign*b.imag();
	res.imag() = a.imag()+sign*b.real();
}


// a = -i b
template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_miB( Kokkos::complex<T>& a, const T& sign, const Kokkos::complex<T>& b)
{
	a.real() += sign*b.imag();
	a.imag() -= sign*b.real();
}


// a = b
template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_B( Kokkos::complex<T>& a, const T& sign, const Kokkos::complex<T>& b)
{
	a.real() += sign*b.real();
	a.imag() += sign*b.imag();
}


} // namespace



#endif /* TEST_KOKKOS_KOKKOS_OPS_H_ */
