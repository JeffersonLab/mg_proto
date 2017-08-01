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

#include "MG_config.h"
#if defined(MG_USE_AVX512)
#include <immintrin.h>
#endif

namespace MG
{


// General
template<typename T, int N>
struct SIMDComplex {

  Kokkos::complex<T> _data[N] __attribute__((aligned(2*N*sizeof(T))));

	constexpr static int len() { return N; }

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
void Load(SIMDComplex<T,N>& result, const SIMDComplex<T,N>& source)
{
	T* dest = reinterpret_cast<T*>(&(result._data[0]));
	const T* src = reinterpret_cast<const T*>(&(source._data[0]));

#pragma omp simd safelen(2*N)
	for(int i=0; i < 2*N; ++i) {
		dest[i] = src[i];
	}
}

template<typename T, int N>
KOKKOS_FORCEINLINE_FUNCTION
void Store(SIMDComplex<T,N>& result, const SIMDComplex<T,N>& source)
{
	T* dest = reinterpret_cast<T*>(&(result._data[0]));
	const T* src = reinterpret_cast<const T*>(&(source._data[0]));

#pragma omp simd safelen(2*N)
	for(int i=0; i < 2*N; ++i) {
		dest[i] = src[i];
	}
}

template<typename T, int N>
KOKKOS_FORCEINLINE_FUNCTION
void Stream(SIMDComplex<T,N>& result, const SIMDComplex<T,N>& source)
{
	T* dest = reinterpret_cast<T*>(&(result._data[0]));
	const T* src = reinterpret_cast<const T*>(&(source._data[0]));

#pragma omp simd safelen(2*N)
	for(int i=0; i < 2*N; ++i) {
		dest[i] = src[i];
	}
}

template<typename T, int N>
KOKKOS_FORCEINLINE_FUNCTION
void ComplexZero(SIMDComplex<T,N>& result)
{
#pragma omp simd safelen(N)
	for(int i=0; i < N; ++i) {
		result._data[i]=Kokkos::complex<T>(0,0);
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



#if defined(MG_USE_AVX512)

// ----***** SPECIALIZED *****
template<>
  struct SIMDComplex<float,8> {

  explicit SIMDComplex<float,8>() {}

  union {
    Kokkos::complex<float> _data[8];
    __m512 _vdata;
  };

  constexpr static int len() { return 8; }

  inline
    void set(int l, const Kokkos::complex<float>& value)
  {
		_data[l] = value;
  }

  inline
    const Kokkos::complex<float>& operator()(int i) const
  {
    return _data[i];
  }
};

  template<>
  KOKKOS_FORCEINLINE_FUNCTION
  void Load<float,8>(SIMDComplex<float,8>& result, 
		     const SIMDComplex<float,8>& source)
  {
    void const* src = reinterpret_cast<void const*>(&(source._data[0]));
    
    result._vdata = _mm512_load_ps(src);
  }
  
  template<>
  KOKKOS_FORCEINLINE_FUNCTION
  void ComplexCopy<float,8>(SIMDComplex<float,8>& result, 
			    const SIMDComplex<float,8>& source)
  {
    result._vdata  = source._vdata;
  }

  template<>
  KOKKOS_FORCEINLINE_FUNCTION
  void Store<float,8>(SIMDComplex<float,8>& result, 
		      const SIMDComplex<float,8>& source)
  {
    void* dest = reinterpret_cast<void*>(&(result._data[0]));
    _mm512_store_ps(dest,source._vdata);
  }

  template<>
  KOKKOS_FORCEINLINE_FUNCTION
  void Stream<float,8>(SIMDComplex<float,8>& result, 
		      const SIMDComplex<float,8>& source)
  {
    void* dest = reinterpret_cast<void*>(&(result._data[0]));
    _mm512_stream_ps(dest,source._vdata);
  }
  

  template<>
  KOKKOS_FORCEINLINE_FUNCTION
  void ComplexZero<float,8>(SIMDComplex<float,8>& result)
  {
    result._vdata = _mm512_setzero_ps();
  }

  template<>
  KOKKOS_FORCEINLINE_FUNCTION
  void
  ComplexPeq<float,8>(SIMDComplex<float,8>& res, 
		      const SIMDComplex<float,8>& a)
  {
    res._vdata = _mm512_add_ps(res._vdata,a._vdata);
  }



template<>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_B<float,8>( SIMDComplex<float,8>& res, 
			    const SIMDComplex<float,8>& a, 
			    const float& sign, 
			    const SIMDComplex<float,8>& b)
{
  __m512 sgnvec = _mm512_set1_ps(sign);
  res._vdata = _mm512_fmadd_ps(sgnvec,b._vdata,a._vdata);
}

// a = b
template<>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_B( SIMDComplex<float,8>& a, 
		   const float& sign, 
		   const SIMDComplex<float,8>& b)
{
  __m512 sgnvec=_mm512_set1_ps(sign);
  a._vdata = _mm512_fmadd_ps( sgnvec, b._vdata, a._vdata);
}


template<>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexCMadd(SIMDComplex<float,8>& res, 
	     const Kokkos::complex<float>& a, 
	     const SIMDComplex<float,8>& b)
{
  __m512 avec_re = _mm512_set1_ps( a.real() );
  __m512 avec_im = _mm512_set1_ps( a.imag() );
  
  __m512 sgnvec = _mm512_set_ps( 1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1);
  __m512 perm_b = _mm512_mul_ps(sgnvec,_mm512_shuffle_ps(b._vdata,b._vdata,0xb1));
		
 
  res._vdata = _mm512_fmadd_ps( avec_re, b._vdata, res._vdata);
  res._vdata = _mm512_fmadd_ps( avec_im,perm_b, res._vdata);
}

  template<>
KOKKOS_FORCEINLINE_FUNCTION
void
  ComplexConjMadd<float,8>(SIMDComplex<float,8>& res, const Kokkos::complex<float>& a, 
		const SIMDComplex<float,8>& b)
{
  __m512 sgnvec2 = _mm512_set_ps(-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1);
  __m512 avec_re = _mm512_set1_ps( a.real() );
  __m512 avec_im = _mm512_set1_ps( a.imag() );
  __m512 perm_b = _mm512_mul_ps(sgnvec2,_mm512_shuffle_ps(b._vdata,b._vdata,0xb1));
  res._vdata = _mm512_fmadd_ps( avec_re, b._vdata, res._vdata);
  res._vdata = _mm512_fmadd_ps( avec_im, perm_b, res._vdata);
}


template<>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexCMadd<float,8>(SIMDComplex<float,8>& res, 
		      const SIMDComplex<float,8>& a, 
		      const SIMDComplex<float,8>& b)
{
  __m512 avec_re = _mm512_shuffle_ps( a._vdata, a._vdata, 0xa0 );
  __m512 avec_im = _mm512_shuffle_ps( a._vdata, a._vdata, 0xf5 );
  
  __m512 sgnvec = _mm512_set_ps( 1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1);
  __m512 perm_b = _mm512_mul_ps(sgnvec,_mm512_shuffle_ps(b._vdata,b._vdata,0xb1));

  res._vdata = _mm512_fmadd_ps( avec_re, b._vdata, res._vdata);
  res._vdata = _mm512_fmadd_ps( avec_im, perm_b, res._vdata);

}

template<>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexConjMadd<float,8>(SIMDComplex<float,8>& res, const SIMDComplex<float,8>& a, const SIMDComplex<float,8>& b)
{
  __m512 sgnvec2 = _mm512_set_ps(-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1);
  __m512 avec_re = _mm512_shuffle_ps( a._vdata, a._vdata, 0xa0 );
  __m512 avec_im = _mm512_shuffle_ps( a._vdata, a._vdata, 0xf5 );
  __m512 perm_b = _mm512_mul_ps(sgnvec2,_mm512_shuffle_ps(b._vdata,b._vdata,0xb1));

  res._vdata = _mm512_fmadd_ps( avec_re, b._vdata, res._vdata);
  res._vdata = _mm512_fmadd_ps( avec_im, perm_b, res._vdata);
}

template<>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_iB<float,8>( SIMDComplex<float,8>& res, 
			     const SIMDComplex<float,8>& a, 
			     const float& sign, 
			     const SIMDComplex<float,8>& b)
{
  __m512 perm_b = _mm512_shuffle_ps( b._vdata, b._vdata, 0xb1);
  __m512 sgnvec = _mm512_mul_ps( _mm512_set_ps(1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1),
				 _mm512_set1_ps(sign) );
  res._vdata = _mm512_fmadd_ps( sgnvec,perm_b, a._vdata);
}


// a = -i b
template<>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_miB<float,8>( SIMDComplex<float,8>& a, 
			      const float& sign, 
			      const SIMDComplex<float,8>& b)
{
  __m512 perm_b = _mm512_shuffle_ps( b._vdata, b._vdata, 0xb1);
  __m512 sgnvec2 = _mm512_mul_ps(
				 _mm512_set_ps(-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1),
				 _mm512_set1_ps(sign) );

  a._vdata = _mm512_fmadd_ps(sgnvec2,perm_b,a._vdata);
}

#endif

} // namespace



#endif /* TEST_KOKKOS_KOKKOS_OPS_H_ */