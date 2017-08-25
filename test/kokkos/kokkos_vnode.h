#ifndef TEST_KOKKOS_VNODE_H
#define TEST_KOKKOS_VNODE_H

//#ifdef KOKKOS_HAVE_CUDA
//#include <sm_30_intrinsics.h>
//#endif

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

 };


template<typename T>
struct VNode<T,2> {
  using VecType =  ThreadSIMDComplex<typename BaseType<T>::Type,2>;
  static constexpr int VecLen =  2;
  static constexpr int NDim = 1;

  static constexpr int Dim0 = 1;
  static constexpr int Dim1 = 1;
  static constexpr int Dim2 = 1;
  static constexpr int Dim3 = 2;

#if 0
  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteT(VecType& vec) 
  {
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
#endif

};


template<typename T>
struct VNode<T,4> {

  using VecType = ThreadSIMDComplex<typename BaseType<T>::Type,4>;

  static constexpr int VecLen =  4;
  static constexpr int NDim = 2;

  static constexpr int Dim0 = 1;
  static constexpr int Dim1 = 1;
  static constexpr int Dim2 = 2;
  static constexpr int Dim3 = 2;

#if 0
  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteT(VecType& vec) 
  {

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
#endif

};

template<typename T>
struct VNode<T,8> {

  using VecType = ThreadSIMDComplex<typename BaseType<T>::Type,8>;

  static constexpr int VecLen = 8;
  static constexpr int NDim = 3;

  static constexpr int Dim0 = 1;
  static constexpr int Dim1 = 2;
  static constexpr int Dim2 = 2;
  static constexpr int Dim3 = 2;

#if 0 
  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteT(VecType& vec) 
  {
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
#endif
};


template<typename T>
struct VNode<T,16> {

  using VecType = ThreadSIMDComplex<typename BaseType<T>::Type,16>;

  static constexpr int VecLen = 16; 
  static constexpr int NDim =4; 

  static constexpr int Dim0 = 2;
  static constexpr int Dim1 = 2;
  static constexpr int Dim2 = 2;
  static constexpr int Dim3 = 2;

#if 0
  static
  KOKKOS_FORCEINLINE_FUNCTION
  void permuteT(VecType& vec) 
  {
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
#endif

};



}

#endif
