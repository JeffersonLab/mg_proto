/*
 * kokkos_ops.h
 *
 *  Created on: Jul 26, 2017
 *      Author: bjoo
 */
#pragma once
#ifndef TEST_KOKKOS_KOKKOS_VECTPYE_H_
#define TEST_KOKKOS_KOKKOS_VECTYPE_H_

#include <Kokkos_Complex.hpp>
#include "kokkos_types.h"
namespace MG
{

template<typename T, int N>
struct SIMDComplex {

	Kokkos::complex<T> _data[N];

	constexpr int len() { return N; }

	inline
	void set(int l, const Kokkos::complex<T>& value)
	{
		_data[l] = value;
	}

	inline
	const Kokkos::complex<T>& operator()(int i) const
	{
		return _data[i];
	}
};

template<typename T, int N>
KOKKOS_FORCEINLINE_FUNCTION
void ComplexCopy(SIMDComplex<T,N>& result, const SIMDComplex<T,N>& source)
{
#pragma omp simd safelen(N)
	for(int i=0; i < N; ++i) {
		result._data[i] = source._data[i];
	}
}

template<typename T, int N>
KOKKOS_FORCEINLINE_FUNCTION
void ComplexZero(SIMDComplex<T,N>& result)
{
#pragma omp simd safelen(N)
	for(int i=0; i < N; ++i) {
		result.set(i,Kokkos::complex<T>(0,0));
	}
}

template<typename T, int N>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexPeq(SIMDComplex<T,N>& res, const SIMDComplex<T,N>& a)
{
#pragma omp simd safelen(N)
	for(int i=0; i < N; ++i)
		res._data[i] += a._data[i]; // Complex Multiplication
}


template<typename T, int N>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexCMadd(SIMDComplex<T,N>& res, const Kokkos::complex<T>& a, const SIMDComplex<T,N>& b)
{
#pragma omp simd safelen(N)
	for(int i=0; i < N; ++i)
		res._data[i] += a*b._data[i]; // Complex Multiplication
}

template<typename T, int N>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexConjMadd(SIMDComplex<T,N>& res, const Kokkos::complex<T>& a, const SIMDComplex<T,N>& b)
{
#pragma omp simd safelen(N)
	for(int i=0; i < N; ++i)
		res._data[i] += Kokkos::conj(a)*b(i); // Complex Multiplication
}



template<typename T, int N>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexCMadd(SIMDComplex<T,N>& res, const SIMDComplex<T,N>& a, const SIMDComplex<T,N>& b)
{
#pragma omp simd safelen(N)
	for(int i=0; i < N; ++i)
		res._data[i] += a(i)*b(i); // Complex Multiplication
}

template<typename T, int N>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexConjMadd(SIMDComplex<T,N>& res, const SIMDComplex<T,N>& a, const SIMDComplex<T,N>& b)
{
#pragma omp simd safelen(N)
	for(int i=0; i < N; ++i)
		res._data[i] += Kokkos::conj(a(i))*b(i); // Complex Multiplication
}

template<typename T, int N>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_B( SIMDComplex<T,N>& res, const SIMDComplex<T,N>& a, const T& sign, const SIMDComplex<T,N>& b)
{
#pragma omp simd safelen(N)
	for(int i=0; i < N; ++i) {
		res._data[i].real() = a(i).real() + sign*b(i).real();
		res._data[i].imag() = a(i).imag() + sign*b(i).imag();
	}
}

template<typename T, int N>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_iB( SIMDComplex<T,N>& res, const SIMDComplex<T,N>& a, const T& sign, const SIMDComplex<T,N>& b)
{
#pragma omp simd safelen(N)
	for(int i=0; i < N; ++i) {
		res._data[i].real() = a(i).real() - sign*b(i).imag();
		res._data[i].imag() = a(i).imag() + sign*b(i).real();
	}
}

// a = -i b
template<typename T,int N>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_miB( SIMDComplex<T,N>& a, const T& sign, const SIMDComplex<T,N>& b)
{
#pragma omp simd safelen(N)
	for(int i=0; i < N; ++i) {
		a._data[i].real() += sign*b(i).imag();
		a._data[i].imag() -= sign*b(i).real();
	}
}

// a = b
template<typename T, int N>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_B( SIMDComplex<T,N>& a, const T& sign, const SIMDComplex<T,N>& b)
{
#pragma omp simd safelen(N)
	for(int i=0; i < N; ++i) {
		a._data[i].real() += sign*b(i).real();
		a._data[i].imag() += sign*b(i).imag();
	}
}


} // namespace



#endif /* TEST_KOKKOS_KOKKOS_OPS_H_ */
