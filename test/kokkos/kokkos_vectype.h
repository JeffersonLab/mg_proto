/*
 * kokkos_ops.h
 *
 *  Created on: Jul 26, 2017
 *      Author: bjoo
 */
#pragma once
#ifndef TEST_KOKKOS_KOKKOS_VECTPYE_H_
#define TEST_KOKKOS_KOKKOS_VECTYPE_H_

#include "kokkos_defaults.h"
#include "kokkos_types.h"

#include "MG_config.h"
#if defined(MG_USE_AVX512)
#include <immintrin.h>
#endif


#include <Kokkos_Core.hpp>


namespace MG
{


// General
template<typename T, int N>
  struct alignas(2*sizeof(T)) SIMDComplex {
  MGComplex<T> _data[N]; 
  constexpr static int len() { return N; }
  
  KOKKOS_INLINE_FUNCTION
  void set(int l, const MGComplex<T>& value)
  {
    _data[l] = value;
  }
  
  KOKKOS_INLINE_FUNCTION
  const MGComplex<T>& operator()(int i) const
  {
    return _data[i];
  }
  
  KOKKOS_INLINE_FUNCTION
  MGComplex<T>& operator()(int i) {
    return _data[i];
  }
};

 // On the GPU only one elemen per 'VectorThread'
template<typename T, int N>
  struct alignas(2*sizeof(T)) GPUThreadSIMDComplex {

  MGComplex<T> _data;

  // This is the vector length so still N
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr static  int len() { return N; }
  
  // Ignore l
  KOKKOS_INLINE_FUNCTION
  void set(int l, const MGComplex<T>& value)
  {
    _data = value;
  }
  
  // Ignore i
  KOKKOS_FORCEINLINE_FUNCTION
  const MGComplex<T>& operator()(int i) const
  {
    return _data;
  }
  
  // Ignore i
  KOKKOS_FORCEINLINE_FUNCTION
  MGComplex<T>& operator()(int i) {
    return _data;
  }
};

  // THIS IS WHERE WE INTRODUCE SOME NONPORTABILITY
  // ThreadSIMDComplex ***MUST** only be instantiated in 
  // a Kokkos parallel region
#ifdef KOKKOS_HAVE_CUDA
  template<typename T, int N> 
  using ThreadSIMDComplex = GPUThreadSIMDComplex<T,N>;
#else
  template<typename T, int N>
  using ThreadSIMDComplex = SIMDComplex<T,N>;
#endif

// T1 must support indexing with operator()
  template<typename T, int N, template <typename,int> class T1, template <typename,int> class T2>
KOKKOS_FORCEINLINE_FUNCTION
   void ComplexCopy(T1<T,N>& result, const T2<T,N>& source)
{
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) {
      result(i) = source(i);
    });
}

 template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
KOKKOS_FORCEINLINE_FUNCTION
   void Load(T1<T,N>& result, const T2<T,N>& source)
{
  
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) {
      result(i) = source(i);
  //    T* res=reinterpret_cast<T*>(&(result(i)));
  //    const T* src = reinterpret_cast<const T*>(&(source(i)));
  //    res[0] = src[0];
  //    res[1] = src[1];

    });
}

 template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
KOKKOS_FORCEINLINE_FUNCTION
   void Store(T1<T,N>& result, const T2<T,N>& source)
{
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) {
      result(i) = source(i);
#if 0
      T* res=reinterpret_cast<T*>(&(result(i)));
      const T* src = reinterpret_cast<const T*>(&(source(i)));
      res[0] = src[0];
      res[1] = src[1];
#endif
    });
}

 template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
   KOKKOS_FORCEINLINE_FUNCTION
 void Stream(T1<T,N>& result, const T2<T,N>& source)
 {
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) {
      result(i) = source(i);
#if 0
      T* res=reinterpret_cast<T*>(&(result(i)));
      const T* src = reinterpret_cast<const T*>(&(source(i)));
      res[0] = src[0];
      res[1] = src[1];
#endif
    });
}

  template<typename T, int N, template<typename,int> class T1>
KOKKOS_FORCEINLINE_FUNCTION
void ComplexZero(T1<T,N>& result)
{
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) {
      result(i)=MGComplex<T>(0,0);
#if 0
      T* res=reinterpret_cast<T*>(&(result(i)));
      res[0] = 0;
      res[1] = 0;
#endif

    });
}

  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexPeq(T1<T,N>& res, const T2<T,N>& a)
{
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) {
      // res(i) += a(i); // Complex Addition
      T* r = reinterpret_cast<T*>( &(res(i)) );
      const T* src = reinterpret_cast<const T*>(&(a(i)));
      r[0] += src[0];
      r[1] += src[1];
    });
}


  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexCMadd(T1<T,N>& res, const MGComplex<T>& a, const T2<T,N>& b)
{
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) {
      // res(i)+= a*b(i); // Complex Multiplication
      T* r = reinterpret_cast<T*>(&(res(i)));
      const T* ap = reinterpret_cast<const T*>(&(a));
      const T* bp = reinterpret_cast<const T*>(&(b(i)));
      
      r[0] += ap[0]*bp[0];
      r[1] += ap[1]*bp[0];
      r[0] -= ap[1]*bp[1];
      r[1] += ap[0]*bp[1];

    });
}


  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexConjMadd(T1<T,N>& res, const MGComplex<T>& a, const T2<T,N>& b)
{
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) {
      // res(i) += Kokkos::conj(a)*b(i); // Complex Multiplication
      T* r = reinterpret_cast<T*>(&(res(i)));
      const T* ap = reinterpret_cast<const T*>(&(a));
      const T* bp = reinterpret_cast<const T*>(&(b(i)));
      r[0] += ap[0]*bp[0];
      r[0] += ap[1]*bp[1];
      r[1] -= ap[1]*bp[0];
      r[1] += ap[0]*bp[1];

    });

}



  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2, template<typename,int> class T3>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexCMadd(T1<T,N>& res, const T2<T,N>& a, const T3<T,N>& b)
{
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) { 
      // res(i) += a(i)*b(i); // Complex Multiplication
      T* r = reinterpret_cast<T*>(&(res(i)));
      const T* ap = reinterpret_cast<const T*>(&(a(i)));
      const T* bp = reinterpret_cast<const T*>(&(b(i)));
      
      r[0] += ap[0]*bp[0];
      r[1] += ap[1]*bp[0];
      r[0] -= ap[1]*bp[1];
      r[1] += ap[0]*bp[1];

    });
}

  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2, template<typename,int> class T3>
KOKKOS_FORCEINLINE_FUNCTION
void
  ComplexConjMadd(T1<T,N>& res, const T2<T,N>& a, const T3<T,N>& b)
{
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) { 
      // res(i) += Kokkos::conj(a(i))*b(i); // Complex Multiplication
      T* r = reinterpret_cast<T*>(&(res(i)));
      const T* ap = reinterpret_cast<const T*>(&(a(i)));
      const T* bp = reinterpret_cast<const T*>(&(b(i)));
      r[0] += ap[0]*bp[0];
      r[0] += ap[1]*bp[1];
      r[1] -= ap[1]*bp[0];
      r[1] += ap[0]*bp[1];

    });
}

  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2, template<typename,int> class T3>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_B( T1<T,N>& res, const T2<T,N>& a, const T& sign, const T3<T,N>& b)
{
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) { 
      //      res(i).real() = a(i).real() + sign*b(i).real();
      //      res(i).imag() = a(i).imag() + sign*b(i).imag();
      T* r=reinterpret_cast<T*>(&(res(i)));
      const T* ap=reinterpret_cast<const T*>(&(a(i)));
      const T* bp=reinterpret_cast<const T*>(&(b(i)));

      r[0] = ap[0] + sign * bp[0];
      r[1] = ap[1] + sign * bp[1];
    });

}

  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2, template<typename,int> class T3>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_iB( T1<T,N>& res, const T2<T,N>& a, const T& sign, const T3<T,N>& b)
{
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) {
      //res(i).real() = a(i).real() - sign*b(i).imag();
      //res(i).imag() = a(i).imag() + sign*b(i).real();
      T* r=reinterpret_cast<T*>(&(res(i)));
      const T* ap=reinterpret_cast<const T*>(&(a(i)));
      const T* bp=reinterpret_cast<const T*>(&(b(i)));

      r[0] = ap[0] - sign * bp[1];
      r[1] = ap[1] + sign * bp[0];

    });
}

// a = -i b
  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_miB( T1<T,N>& a, const T& sign, const T2<T,N>& b)
{
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) {
      T* ap=reinterpret_cast<T*>(&(a(i)));
      const T* bp=reinterpret_cast<const T*>(&(b(i)));

      ap[0] += sign*bp[1];
      ap[1] -= sign*bp[0];
    });
}

    
// a = b
  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
KOKKOS_FORCEINLINE_FUNCTION
  void A_peq_sign_B( T1<T,N>& a, const T& sign, const T2<T,N>& b)
{
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) { 
     T* ap=reinterpret_cast<T*>(&(a(i)));
      const T* bp=reinterpret_cast<const T*>(&(b(i)));

      //      a(i).real() += sign*b(i).real();
      // a(i).imag() += sign*b(i).imag();
      ap[0] += sign*bp[0];
      ap[1] += sign*bp[1];
    });
}



#if defined(MG_USE_AVX512)

// ----***** SPECIALIZED *****
template<>
  struct SIMDComplex<float,8> {

  explicit SIMDComplex<float,8>() {}

  union {
    Complex<float> _data[8];
    __m512 _vdata;
  };

  constexpr static int len() { return 8; }

  inline
    void set(int l, const Complex<float>& value)
  {
		_data[l] = value;
  }

  inline
    const Complex<float>& operator()(int i) const
  {
    return _data[i];
  }

  inline
  Complex<float>& operator()(int i) {
    return _data[i];
  }
};

  template<>
  KOKKOS_FORCEINLINE_FUNCTION
  void Load<float,8,SIMDComplex,SIMDComplex>(SIMDComplex<float,8>& result, 
		     const SIMDComplex<float,8>& source)
  {
    void const* src = reinterpret_cast<void const*>(&(source._data[0]));
    
    result._vdata = _mm512_load_ps(src);
  }
  
  template<>
  KOKKOS_FORCEINLINE_FUNCTION
  void ComplexCopy<float,8,SIMDComplex,SIMDComplex>(SIMDComplex<float,8>& result, 
			    const SIMDComplex<float,8>& source)
  {
    result._vdata  = source._vdata;
  }

  template<>
  KOKKOS_FORCEINLINE_FUNCTION
  void Store<float,8,SIMDComplex,SIMDComplex>(SIMDComplex<float,8>& result, 
		      const SIMDComplex<float,8>& source)
  {
    void* dest = reinterpret_cast<void*>(&(result._data[0]));
    _mm512_store_ps(dest,source._vdata);
  }

  template<>
  KOKKOS_FORCEINLINE_FUNCTION
  void Stream<float,8,SIMDComplex,SIMDComplex>(SIMDComplex<float,8>& result, 
		      const SIMDComplex<float,8>& source)
  {
    void* dest = reinterpret_cast<void*>(&(result._data[0]));
    _mm512_stream_ps(dest,source._vdata);
  }
  

  template<>
  KOKKOS_FORCEINLINE_FUNCTION
  void ComplexZero<float,8,SIMDComplex>(SIMDComplex<float,8>& result)
  {
    result._vdata = _mm512_setzero_ps();
  }

  template<>
  KOKKOS_FORCEINLINE_FUNCTION
  void
  ComplexPeq<float,8,SIMDComplex,SIMDComplex>(SIMDComplex<float,8>& res, 
		      const SIMDComplex<float,8>& a)
  {
    res._vdata = _mm512_add_ps(res._vdata,a._vdata);
  }



template<>
KOKKOS_FORCEINLINE_FUNCTION
void 
A_add_sign_B<float,8,SIMDComplex,SIMDComplex,SIMDComplex>( SIMDComplex<float,8>& res, 
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
void 
A_peq_sign_B<float,8,SIMDComplex,SIMDComplex>( SIMDComplex<float,8>& a, 
		   const float& sign, 
		   const SIMDComplex<float,8>& b)
{
  __m512 sgnvec=_mm512_set1_ps(sign);
  a._vdata = _mm512_fmadd_ps( sgnvec, b._vdata, a._vdata);
}


template<>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexCMadd<float,8,SIMDComplex,SIMDComplex>(SIMDComplex<float,8>& res, 
	     const Complex<float>& a, 
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
  ComplexConjMadd<float,8,SIMDComplex,SIMDComplex>(SIMDComplex<float,8>& res, const Complex<float>& a, 
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
ComplexCMadd<float,8,SIMDComplex,SIMDComplex,SIMDComplex>(SIMDComplex<float,8>& res, 
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
ComplexConjMadd<float,8,SIMDComplex,SIMDComplex,SIMDComplex>(SIMDComplex<float,8>& res, const SIMDComplex<float,8>& a, const SIMDComplex<float,8>& b)
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
void A_add_sign_iB<float,8,SIMDComplex,SIMDComplex,SIMDComplex>( SIMDComplex<float,8>& res, 
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
void A_peq_sign_miB<float,8,SIMDComplex,SIMDComplex>( SIMDComplex<float,8>& a, 
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
