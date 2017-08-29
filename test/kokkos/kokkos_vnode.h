#ifndef TEST_KOKKOS_VNODE_H
#define TEST_KOKKOS_VNODE_H

//#ifdef KOKKOS_HAVE_CUDA
//#include <sm_30_intrinsics.h>
//#endif
#include "MG_config.h"
#include "kokkos_traits.h"
#include "kokkos_vectype.h"
#include "lattice/lattice_info.h"

#include <assert.h>
namespace MG {


template<typename T, int N>
struct VNode;

template<typename T>
  struct VNode<T,1> {
  using VecType =  ThreadSIMDComplex<typename BaseType<T>::Type,1>;

  static constexpr int VecLen = 1 ;
  static constexpr int nDim = 0;

  static constexpr int Dim0 = 1;
  static constexpr int Dim1 = 1;
  static constexpr int Dim2 = 1;
  static constexpr int Dim3 = 1;


  static
  KOKKOS_FORCEINLINE_FUNCTION
  void  permuteT(VecType& vec) {}

  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteZ(VecType& vec) {}

  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteY(VecType& vec) {}

  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteX(VecType& vec) {}

 }; // Struct Vector Length = 1


template<typename T>
struct VNode<T,2> {
  using VecType =  ThreadSIMDComplex<typename BaseType<T>::Type,2>;
  static constexpr int VecLen =  2;
  static constexpr int NDim = 1;

  static constexpr int Dim0 = 1;
  static constexpr int Dim1 = 1;
  static constexpr int Dim2 = 1;
  static constexpr int Dim3 = 2;


  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteT(VecType& vec) 
  {
	  // permute ab->ba
	  auto tmp = vec(0);
	  vec(0)=vec(1);
	  vec(1)=tmp;
  }

  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteZ(VecType& vec) 
  {

  }

  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteY(VecType& vec) 
  {

  }

  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteX(VecType& vec) 
  {

  }


}; // Struct Vector Length = 2


template<typename T>
struct VNode<T,4> {

  using VecType = ThreadSIMDComplex<typename BaseType<T>::Type,4>;

  static constexpr int VecLen =  4;
  static constexpr int NDim = 2;

  static constexpr int Dim0 = 1;
  static constexpr int Dim1 = 1;
  static constexpr int Dim2 = 2;
  static constexpr int Dim3 = 2;


  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteT(VecType& vec) 
  {
	  // Permute  abcd -> cdba

	  auto tmp0=vec(0);
	  auto tmp1=vec(1);
	  vec(0)=vec(2);
	  vec(1)=vec(3);
	  vec(2)=tmp0;
	  vec(3)=tmp1;

  }

  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteZ(VecType& vec) 
  {

	  // permute: ab cd -> ba dc
	  auto tmp=vec(0);
	  vec(0)=vec(1);
	  vec(1)=tmp;

	  auto tmp2=vec(2);
	  vec(2)=vec(3);
	  vec(3)=tmp2;
  }

  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteY(VecType& vec) 
  {
  }

  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteX(VecType& vec) 
  {
  }


};   // struct vector length = 4

template<typename T>
struct VNode<T,8> {

  using VecType = ThreadSIMDComplex<typename BaseType<T>::Type,8>;

  static constexpr int VecLen = 8;
  static constexpr int NDim = 3;

  static constexpr int Dim0 = 1;
  static constexpr int Dim1 = 2;
  static constexpr int Dim2 = 2;
  static constexpr int Dim3 = 2;


  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteT(VecType& vec) 
  {
	  // Permute ab cd ef gh -> ef gh ab cd
	  auto tmp0 = vec(0);
	  auto tmp1 = vec(1);
	  auto tmp2 = vec(2);
	  auto tmp3 = vec(3);

	  vec(0)=vec(4);
	  vec(1)=vec(5);
	  vec(2)=vec(6);
	  vec(3)=vec(7);
	  vec(4)=tmp0;
	  vec(5)=tmp1;
	  vec(6)=tmp2;
	  vec(7)=tmp3;

  }

  static
    KOKKOS_FORCEINLINE_FUNCTION
    void permuteZ(VecType& vec) 
  {
	  // permute ab cd ef gh -> cd ab gh ef
	  auto tmp0 = vec(0);
	  auto tmp1 = vec(1);
	  vec(0)=vec(2);
	  vec(1)=vec(3);
	  vec(2)=tmp0;
	  vec(3)=tmp1;

	  auto tmp2 = vec(4);
	  auto tmp3 = vec(5);
	  vec(4)=vec(6);
	  vec(5)=vec(7);
	  vec(6)=tmp2;
	  vec(7)=tmp3;

  }

  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteY(VecType& vec) 
  {
	  // permute ab cd ef gh -> ba dc fe hg
	  auto tmp0 = vec(0);
	  vec(0)=vec(1);
	  vec(1)=tmp0;

	  auto tmp1 =vec(2);
	  vec(2)=vec(3);
	  vec(3)=tmp1;

	  auto tmp2 = vec(4);
	  vec(4)=vec(5);
	  vec(5)=tmp2;

	  auto tmp3 = vec(6);
	  vec(6)=vec(7);
	  vec(7)=tmp3;

  }

  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteX(VecType& vec) 
  {
  }
}; // struct vector length = 8
} // namespace

#if defined(MG_USE_AVX512)
#include <immintrin.h>

namespace MG {
template<>
struct VNode<MGComplex<float>,8> {

  using VecType = SIMDComplex<float,8>;

  static constexpr int VecLen = 8;
  static constexpr int NDim = 3;

  static constexpr int Dim0 = 1;
  static constexpr int Dim1 = 2;
  static constexpr int Dim2 = 2;
  static constexpr int Dim3 = 2;


  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteT(VecType& vec)
  {
	  // printf(".");
	  // Permute ab cd ef gh -> ef gh ab cd

	  vec._vdata = _mm512_permutexvar_ps(_mm512_set_epi32(7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8), vec._vdata);

#if 0
	  auto tmp0 = vec(0);
	  auto tmp1 = vec(1);
	  auto tmp2 = vec(2);
	  auto tmp3 = vec(3);

	  vec(0)=vec(4);
	  vec(1)=vec(5);
	  vec(2)=vec(6);
	  vec(3)=vec(7);
	  vec(4)=tmp0;
	  vec(5)=tmp1;
	  vec(6)=tmp2;
	  vec(7)=tmp3;
#endif
  }

  static
    KOKKOS_FORCEINLINE_FUNCTION
    void permuteZ(VecType& vec)
  {
	  vec._vdata = _mm512_permutexvar_ps(_mm512_set_epi32(11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4), vec._vdata);

#if 0
	  // permute ab cd ef gh -> cd ab gh ef
	  auto tmp0 = vec(0);
	  auto tmp1 = vec(1);
	  vec(0)=vec(2);
	  vec(1)=vec(3);
	  vec(2)=tmp0;
	  vec(3)=tmp1;

	  auto tmp2 = vec(4);
	  auto tmp3 = vec(5);
	  vec(4)=vec(6);
	  vec(5)=vec(7);
	  vec(6)=tmp2;
	  vec(7)=tmp3;
#endif
  }

  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteY(VecType& vec)
  {
	  vec._vdata = _mm512_permutexvar_ps(_mm512_set_epi32(13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2), vec._vdata);

#if 0
	  // permute ab cd ef gh -> ba dc fe hg
	  auto tmp0 = vec(0);
	  vec(0)=vec(1);
	  vec(1)=tmp0;

	  auto tmp1 =vec(2);
	  vec(2)=vec(3);
	  vec(3)=tmp1;

	  auto tmp2 = vec(4);
	  vec(4)=vec(5);
	  vec(5)=tmp2;

	  auto tmp3 = vec(6);
	  vec(6)=vec(7);
	  vec(7)=tmp3;
#endif
  }

  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteX(VecType& vec)
  {
  }
}; // struct vector length = 8


#endif

} // Namespace


#endif
