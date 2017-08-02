/*
 * kokkos_ops.h
 *
 *  Created on: Jul 26, 2017
 *      Author: bjoo
 */
#pragma once
#ifndef TEST_KOKKOS_KOKKOS_VECTPYE_H_
#define TEST_KOKKOS_KOKKOS_VECTYPE_H_

#include "my_complex.h"
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

  MyComplex<T> _data[N] __attribute__((aligned(2*N*sizeof(T))));

	constexpr static int len() { return N; }

	inline
	void set(int l, const MyComplex<T>& value)
	{
		_data[l] = value;
	}

	inline
	const MyComplex<T>& operator()(int i) const
	{
		return _data[i];
	}
};

template<typename T, int N>
KOKKOS_FORCEINLINE_FUNCTION
void ComplexCopy(SIMDComplex<T,N>& result, const SIMDComplex<T,N>& source)
{
  MyComplex<T>* resdata = result._data;
  const MyComplex<T>* srcdata = source._data;

#pragma omp simd safelen(N)
	for(int i=0; i < N; ++i) {
		resdata[i] = srcdata[i];
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
  MyComplex<T>* dest = reinterpret_cast<MyComplex<T>*>(&(result._data[0]));
#pragma omp simd safelen(N) 
	for(int i=0; i < N; ++i) {
	  dest[i]=MyComplex<T>(0,0);
	}
}

template<typename T, int N>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexPeq(SIMDComplex<T,N>& res, const SIMDComplex<T,N>& a)
{
  MyComplex<T>* resdata = reinterpret_cast<MyComplex<T>*>(&(res._data[0]));
  const MyComplex<T>* adata = reinterpret_cast<const MyComplex<T>*>(&(a._data[0]));

#pragma omp simd safelen(N) 
	for(int i=0; i < N; ++i)
		resdata[i] += adata[i]; // Complex Multiplication
}


template<typename T, int N>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexCMadd(SIMDComplex<T,N>& res, const MyComplex<T>& a, const SIMDComplex<T,N>& b)
{
  MyComplex<T>* resdata = reinterpret_cast<MyComplex<T>*>(&(res._data[0]));

  const MyComplex<T>* bdata = reinterpret_cast<const MyComplex<T>*>(&(b._data[0]));

#pragma omp simd safelen(N) 
	for(int i=0; i < N; ++i)
		resdata[i] += a*bdata[i]; // Complex Multiplication
}

template<typename T, int N>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexConjMadd(SIMDComplex<T,N>& res, const MyComplex<T>& a, const SIMDComplex<T,N>& b)
{
  MyComplex<T>* resdata = reinterpret_cast<MyComplex<T>*>(&(res._data[0]));

  const MyComplex<T>* bdata = reinterpret_cast<const MyComplex<T>*>(&(b._data[0]));

#pragma omp simd safelen(N) 
	for(int i=0; i < N; ++i)
		resdata[i] += conj(a)*bdata[i]; // Complex Multiplication
}



template<typename T, int N>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexCMadd(SIMDComplex<T,N>& res, const SIMDComplex<T,N>& a, const SIMDComplex<T,N>& b)
{
  MyComplex<T>* resdata = reinterpret_cast<MyComplex<T>*>(&(res._data[0]));
  const MyComplex<T>* adata = reinterpret_cast<const MyComplex<T>*>(&(a._data[0]));
  const MyComplex<T>* bdata = reinterpret_cast<const MyComplex<T>*>(&(b._data[0]));

#pragma omp simd safelen(N) 
	for(int i=0; i < N; ++i)
		resdata[i] += adata[i]*bdata[i]; // Complex Multiplication
}

template<typename T, int N>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexConjMadd(SIMDComplex<T,N>& res, const SIMDComplex<T,N>& a, const SIMDComplex<T,N>& b)
{
  MyComplex<T>* resdata = reinterpret_cast<MyComplex<T>*>(&(res._data[0]));
  const MyComplex<T>* adata = reinterpret_cast<const MyComplex<T>*>(&(a._data[0]));
  const MyComplex<T>* bdata = reinterpret_cast<const MyComplex<T>*>(&(b._data[0]));

#pragma omp simd safelen(N) 
	for(int i=0; i < N; ++i)
		resdata[i] +=conj(adata[i])*bdata[i]; // Complex Multiplication
}

template<typename T, int N>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_B( SIMDComplex<T,N>& res, const SIMDComplex<T,N>& a, const T& sign, const SIMDComplex<T,N>& b)
{
  MyComplex<T>* resdata = reinterpret_cast<MyComplex<T>*>(&(res._data[0]));
  const MyComplex<T>* adata = reinterpret_cast<const MyComplex<T>*>(&(a._data[0]));
  const MyComplex<T>* bdata = reinterpret_cast<const MyComplex<T>*>(&(b._data[0]));

#pragma omp simd safelen(N) 
	for(int i=0; i < N; ++i) {
		resdata[i].real() =  adata[i].real() + sign*bdata[i].real() ;
		resdata[i].imag() =  adata[i].imag() + sign*bdata[i].imag() ;
	}
}

template<typename T, int N>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_iB( SIMDComplex<T,N>& res, const SIMDComplex<T,N>& a, const T& sign, const SIMDComplex<T,N>& b)
{
  MyComplex<T>* resdata = reinterpret_cast<MyComplex<T>*>(&(res._data[0]));
  const MyComplex<T>* adata = reinterpret_cast<const MyComplex<T>*>(&(a._data[0]));
  const MyComplex<T>* bdata = reinterpret_cast<const MyComplex<T>*>(&(b._data[0]));


#pragma omp simd safelen(N) 
	for(int i=0; i < N; ++i) {
		resdata[i].real() =  adata[i].real() - sign*bdata[i].imag() ;
		resdata[i].imag() =  adata[i].imag() + sign*bdata[i].real() ;
	}
}

// a = -i b
template<typename T,int N>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_miB( SIMDComplex<T,N>& a, const T& sign, const SIMDComplex<T,N>& b)
{
  MyComplex<T>* adata = reinterpret_cast<MyComplex<T>*>(&(a._data[0]));
  const MyComplex<T>* bdata = reinterpret_cast<const MyComplex<T>*>(&(b._data[0]));


#pragma omp simd safelen(N) 
	for(int i=0; i < N; ++i) {
		adata[i].real() +=  sign*bdata[i].imag();
		adata[i].imag() -=  sign*bdata[i].real();
	}
}

// a = b
template<typename T, int N>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_B( SIMDComplex<T,N>& a, const T& sign, const SIMDComplex<T,N>& b)
{
  MyComplex<T>* adata = reinterpret_cast<MyComplex<T>*>(&(a._data[0]));
  const MyComplex<T>* bdata = reinterpret_cast<const MyComplex<T>*>(&(b._data[0]));

#pragma omp simd safelen(N) 
	for(int i=0; i < N; ++i) {
		adata[i].real() += sign*bdata[i].real();
		adata[i].imag() += sign*bdata[i].imag();
	}
}


/****** OMP **** SPECIALIZED */

template<>
KOKKOS_FORCEINLINE_FUNCTION
void ComplexCopy(SIMDComplex<float,8>& result, const SIMDComplex<float,8>& source)
{
  MyComplex<float>* resdata = result._data;
  const MyComplex<float>* srcdata = source._data;

#pragma omp simd safelen(8) aligned( resdata,srcdata : 64 )
	for(int i=0; i < 8; ++i) {
		resdata[i] = srcdata[i];
	}
}

template<>
KOKKOS_FORCEINLINE_FUNCTION
void Load(SIMDComplex<float,8>& result, const SIMDComplex<float,8>& source)
{
	float* dest = reinterpret_cast<float*>(&(result._data[0]));
	const float* src = reinterpret_cast<const float*>(&(source._data[0]));

#pragma omp simd safelen(2*8) aligned( dest,src : 64 )
	for(int i=0; i < 2*8; ++i) {
		dest[i] = src[i];
	}
}

template<>
KOKKOS_FORCEINLINE_FUNCTION
void Store(SIMDComplex<float,8>& result, const SIMDComplex<float,8>& source)
{
	float* dest = reinterpret_cast<float*>(&(result._data[0]));
	const float* src = reinterpret_cast<const float*>(&(source._data[0]));

#pragma omp simd safelen(2*8) aligned( dest,src : 64 )
	for(int i=0; i < 2*8; ++i) {
		dest[i] = src[i];
	}
}

template<>
KOKKOS_FORCEINLINE_FUNCTION
void Stream(SIMDComplex<float,8>& result, const SIMDComplex<float,8>& source)
{
	float* dest = reinterpret_cast<float*>(&(result._data[0]));
	const float* src = reinterpret_cast<const float*>(&(source._data[0]));

#pragma omp simd safelen(2*8) aligned(dest,src : 64 )
	for(int i=0; i < 2*8; ++i) {
		dest[i] = src[i];
	}
}

template<>
KOKKOS_FORCEINLINE_FUNCTION
void ComplexZero(SIMDComplex<float,8>& result)
{
  MyComplex<float>* dest = reinterpret_cast<MyComplex<float>*>(&(result._data[0]));
#pragma omp simd safelen(8) aligned(dest : 64)
	for(int i=0; i < 8; ++i) {
	  dest[i]=MyComplex<float>(0,0);
	}
}

template<>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexPeq(SIMDComplex<float,8>& res, const SIMDComplex<float,8>& a)
{
  MyComplex<float>* resdata = reinterpret_cast<MyComplex<float>*>(&(res._data[0]));
  const MyComplex<float>* adata = reinterpret_cast<const MyComplex<float>*>(&(a._data[0]));

#pragma omp simd safelen(8) aligned(resdata,adata : 64)
	for(int i=0; i < 8; ++i)
		resdata[i] += adata[i]; // Complex Multiplication
}


template<>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexCMadd(SIMDComplex<float,8>& res, const MyComplex<float>& a, const SIMDComplex<float,8>& b)
{
  MyComplex<float>* resdata = reinterpret_cast<MyComplex<float>*>(&(res._data[0]));

  const MyComplex<float>* bdata = reinterpret_cast<const MyComplex<float>*>(&(b._data[0]));

#pragma omp simd safelen(8) aligned(resdata,bdata: 64)
	for(int i=0; i < 8; ++i)
		resdata[i] += a*bdata[i]; // Complex Multiplication
}

template<>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexConjMadd(SIMDComplex<float,8>& res, const MyComplex<float>& a, const SIMDComplex<float,8>& b)
{
  MyComplex<float>* resdata = reinterpret_cast<MyComplex<float>*>(&(res._data[0]));

  const MyComplex<float>* bdata = reinterpret_cast<const MyComplex<float>*>(&(b._data[0]));

#pragma omp simd safelen(8) aligned(resdata,bdata: 64)
	for(int i=0; i < 8; ++i)
		resdata[i] += conj(a)*bdata[i]; // Complex Multiplication
}



template<>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexCMadd(SIMDComplex<float,8>& res, const SIMDComplex<float,8>& a, const SIMDComplex<float,8>& b)
{
  MyComplex<float>* resdata = reinterpret_cast<MyComplex<float>*>(&(res._data[0]));
  const MyComplex<float>* adata = reinterpret_cast<const MyComplex<float>*>(&(a._data[0]));
  const MyComplex<float>* bdata = reinterpret_cast<const MyComplex<float>*>(&(b._data[0]));

#pragma omp simd safelen(8) aligned(resdata,adata,bdata: 64)
	for(int i=0; i < 8; ++i)
		resdata[i] += adata[i]*bdata[i]; // Complex Multiplication
}

template<>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexConjMadd(SIMDComplex<float,8>& res, const SIMDComplex<float,8>& a, const SIMDComplex<float,8>& b)
{
  MyComplex<float>* resdata = reinterpret_cast<MyComplex<float>*>(&(res._data[0]));
  const MyComplex<float>* adata = reinterpret_cast<const MyComplex<float>*>(&(a._data[0]));
  const MyComplex<float>* bdata = reinterpret_cast<const MyComplex<float>*>(&(b._data[0]));

#pragma omp simd safelen(8) aligned(resdata,adata,bdata: 64)
	for(int i=0; i < 8; ++i)
		resdata[i] += conj(adata[i])*bdata[i]; // Complex Multiplication
}

template<>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_B( SIMDComplex<float,8>& res, const SIMDComplex<float,8>& a, const float& sign, const SIMDComplex<float,8>& b)
{
  MyComplex<float>* resdata = reinterpret_cast<MyComplex<float>*>(&(res._data[0]));
  const MyComplex<float>* adata = reinterpret_cast<const MyComplex<float>*>(&(a._data[0]));
  const MyComplex<float>* bdata = reinterpret_cast<const MyComplex<float>*>(&(b._data[0]));

#pragma omp simd safelen(8) aligned( resdata, adata, bdata : 64 )
	for(int i=0; i < 8; ++i) {
		resdata[i].real() += sign*bdata[i].real() ;
		resdata[i].imag() += sign*bdata[i].imag() ;
	}
}

template<>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_iB( SIMDComplex<float,8>& res, const SIMDComplex<float,8>& a, const float& sign, const SIMDComplex<float,8>& b)
{
  MyComplex<float>* resdata = reinterpret_cast<MyComplex<float>*>(&(res._data[0]));
  const MyComplex<float>* adata = reinterpret_cast<const MyComplex<float>*>(&(a._data[0]));
  const MyComplex<float>* bdata = reinterpret_cast<const MyComplex<float>*>(&(b._data[0]));


#pragma omp simd safelen(8) aligned( resdata, adata, bdata : 64 )
	for(int i=0; i < 8; ++i) {
		resdata[i].real() -= sign*bdata[i].imag() ;
		resdata[i].imag() += sign*bdata[i].real() ;
	}
}

// a = -i b
template<>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_miB( SIMDComplex<float,8>& a, const float& sign, const SIMDComplex<float,8>& b)
{
  MyComplex<float>* adata = reinterpret_cast<MyComplex<float>*>(&(a._data[0]));
  const MyComplex<float>* bdata = reinterpret_cast<const MyComplex<float>*>(&(b._data[0]));


#pragma omp simd safelen(8) aligned( adata,bdata : 64 )
	for(int i=0; i < 8; ++i) {
		adata[i].real() += sign*bdata[i].imag();
		adata[i].imag() -= sign*bdata[i].real();
	}
}

// a = b
template<>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_B( SIMDComplex<float,8>& a, const float& sign, const SIMDComplex<float,8>& b)
{
  MyComplex<float>* adata = reinterpret_cast<MyComplex<float>*>(&(a._data[0]));
  const MyComplex<float>* bdata = reinterpret_cast<const MyComplex<float>*>(&(b._data[0]));

#pragma omp simd safelen(8) aligned( adata,bdata : 64 )
	for(int i=0; i < 8; ++i) {
		adata[i].real() += sign*bdata[i].real();
		adata[i].imag() += sign*bdata[i].imag();
	}
}





#if defined(MG_USE_AVX512)

// ----***** SPECIALIZED *****
template<>
  struct SIMDComplex<float,8> {

  explicit SIMDComplex<float,8>() {}

  union {
    MyComplex<float> _data[8];
    __m512 _vdata;
  };

  constexpr static int len() { return 8; }

  inline
    void set(int l, const MyComplex<float>& value)
  {
		_data[l] = value;
  }

  inline
    const MyComplex<float>& operator()(int i) const
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
	     const MyComplex<float>& a, 
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
  ComplexConjMadd<float,8>(SIMDComplex<float,8>& res, const MyComplexy<float>& a, 
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
