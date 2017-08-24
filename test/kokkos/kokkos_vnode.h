#ifndef TEST_KOKKOS_VNODE_H
#define TEST_KOKKOS_VNODE_H


#include "kokkos_vectype.h"
#include "lattice/lattice_info.h"
#include "utils/print_utils.h"

namespace MG {

template<typename T, int N>
struct VNode;

template<typename T>
  struct VNode<T,1> {
  using VecType =  ThreadSIMDComplex<typename BaseType<T>::Type,1>;

  static constexpr int vecLen = 1 ;
  static constexpr int nDim = 0;

  static const  int Dims[4];
 };
 
template<typename T>
const  int VNode<T,1>::Dims[4] = { 1,1,1,1 };

template<typename T>
struct VNode<T,2> {

  using VecType =  ThreadSIMDComplex<typename BaseType<T>::Type,2>;

  static constexpr int VecLen =  2;
  static constexpr int NDim = 1;

  static const int Dims[4];
};

template<typename T>
const  int VNode<T,2>::Dims[4] = { 1,1,1,2 };


template<typename T>
struct VNode<T,4> {

  using VecType = ThreadSIMDComplex<typename BaseType<T>::Type,4>;

  static constexpr int VecLen =  4;
  static constexpr int NDim = 2;

  static const int Dims[4];
};

template<typename T>
const  int VNode<T,4>::Dims[4] = { 1,1,2,2 };

template<typename T>
struct VNode<T,8> {

  using VecType = ThreadSIMDComplex<typename BaseType<T>::Type,8>;

  static constexpr int VecLen = 8;
  static constexpr int NDim = 3;

  static const  int Dims[4];
};

template<typename T>
const  int VNode<T,8>::Dims[4] = { 1,2,2,2 };

template<typename T>
struct VNode<T,16> {

  using VecType = ThreadSIMDComplex<typename BaseType<T>::Type,16>;

  static constexpr int VecLen = 16; 
  static constexpr int NDim =4; 

  static const int Dims[4];
};

template<typename T>
const  int VNode<T,16>::Dims[4] = { 2,2,2,2 };




}

#endif
