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
  void  permuteT(VecType& vec_out, const VecType& vec_in) {
	  vec_out(0) = vec_in(0);
  }

  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteZ(VecType& vec_out, const VecType& vec_in) {
	  vec_out(0) = vec_in(0);
  }

  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteY(VecType& vec_out, const VecType& vec_in) {
	  vec_out(0) = vec_in(0);
  }

  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteX(VecType& vec_out, const VecType& vec_in) {
	  vec_out(0) = vec_in(0);
  }

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
  void permuteT(VecType& vec_out, const VecType& vec_in)
  {
	  vec_out(0)=vec_in(1);
	  vec_out(1)=vec_in(0);
  }

  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteZ(VecType& vec_out, const VecType& vec_in)
  {
	  vec_out(0) = vec_in(0);
	  vec_out(1) = vec_in(1);
  }

  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteY(VecType& vec_out, const VecType& vec_in)
  {
	  vec_out(0) = vec_in(0);
	  vec_out(1) = vec_in(1);

  }

  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteX(VecType& vec_out, const VecType& vec_in)
  {
	  vec_out(0) = vec_in(0);
	  vec_out(1) = vec_in(1);

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
  void permuteT(VecType& vec_out, const VecType& vec_in)
  {
	  // Permute  abcd -> cdba


	  vec_out(0)=vec_in(2);
	  vec_out(1)=vec_in(3);
	  vec_out(2)=vec_in(0);
	  vec_out(3)=vec_in(1);

  }

  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteZ(VecType& vec_out, const VecType& vec_in)
  {

	  // permute: ab cd -> ba dc
	  vec_out(0)=vec_in(1);
	  vec_out(1)=vec_in(0);

	  vec_out(2)=vec_in(3);
	  vec_out(3)=vec_in(2);
  }

  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteY(VecType& vec_out, const VecType& vec_in)
  {
	  vec_out(0)=vec_in(0);
	  vec_out(1)=vec_in(1);
	  vec_out(2)=vec_in(2);
	  vec_out(3)=vec_in(3);
  }

  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteX(VecType& vec_out, const VecType& vec_in)
  {
	  vec_out(0)=vec_in(0);
	  vec_out(1)=vec_in(1);
	  vec_out(2)=vec_in(2);
	  vec_out(3)=vec_in(3);
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
  void permuteT(VecType& vec_out, const VecType& vec_in)
  {
	  // Permute ab cd ef gh -> ef gh ab cd

	  vec_out(0)=vec_in(4);
	  vec_out(1)=vec_in(5);
	  vec_out(2)=vec_in(6);
	  vec_out(3)=vec_in(7);
	  vec_out(4)=vec_in(0);
	  vec_out(5)=vec_in(1);
	  vec_out(6)=vec_in(2);
	  vec_out(7)=vec_in(3);

  }

  static
    KOKKOS_FORCEINLINE_FUNCTION
    void permuteZ(VecType& vec_out, const VecType& vec_in)
  {
	  vec_out(0)=vec_in(2);
	  vec_out(1)=vec_in(3);
	  vec_out(2)=vec_in(0);
	  vec_out(3)=vec_in(1);

	  vec_out(4)=vec_in(6);
	  vec_out(5)=vec_in(7);
	  vec_out(6)=vec_in(4);
	  vec_out(7)=vec_in(5);

  }

  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteY(VecType& vec_out, const VecType& vec_in)
  {
	  // permute ab cd ef gh -> ba dc fe hg
	  vec_out(0)=vec_in(1);
	  vec_out(1)=vec_in(0);

	  vec_out(2)=vec_in(3);
	  vec_out(3)=vec_in(2);

	  vec_out(4)=vec_in(5);
	  vec_out(5)=vec_in(4);

	  vec_out(6)=vec_in(7);
	  vec_out(7)=vec_in(6);

  }

  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteX(VecType& vec_out, const VecType& vec_in)
  {
	  vec_out(0)=vec_in(0);
	  vec_out(1)=vec_in(1);
	  vec_out(2)=vec_in(2);
	  vec_out(3)=vec_in(3);
	  vec_out(4)=vec_in(4);
	  vec_out(5)=vec_in(5);
	  vec_out(6)=vec_in(6);
	  vec_out(7)=vec_in(7);

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
  void permuteT(VecType& vec_out, const VecType& vec_in)
  {
	  // printf(".");
	  // Permute ab cd ef gh -> ef gh ab cd

	  vec_out._vdata = _mm512_permutexvar_ps(_mm512_set_epi32(7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8), vec_in._vdata);

  }

  static
    KOKKOS_FORCEINLINE_FUNCTION
    void permuteZ(VecType& vec_out, const VecType& vec_in)
  {
	  vec_out._vdata = _mm512_permutexvar_ps(_mm512_set_epi32(11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4), vec_in._vdata);

  }

  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteY(VecType& vec_out, const VecType& vec_in)
  {
	  vec_out._vdata = _mm512_permutexvar_ps(_mm512_set_epi32(13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2), vec_in._vdata);

  }

  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteX(VecType& vec_out, const VecType& vec_in)
  {
	  vec_out._vdata = vec_in._vdata;

  }
}; // struct vector length = 8

} // Namespace
#endif // AVX512


#endif
