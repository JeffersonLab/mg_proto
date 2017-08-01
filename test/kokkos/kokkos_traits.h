/*
 * kokkos_traits.h
 *
 *  Created on: Jul 30, 2017
 *      Author: bjoo
 */
#pragma once
#ifndef TEST_KOKKOS_KOKKOS_TRAITS_H_
#define TEST_KOKKOS_KOKKOS_TRAITS_H_

#include "Kokkos_Complex.hpp"
#include "kokkos_vectype.h"

namespace MG {
template<typename T>
struct BaseType {
};

template<typename T>
struct BaseType<Kokkos::complex<T>>{
	typedef T Type;
};

template<typename T, int N >
struct BaseType< SIMDComplex<T, N> > {
	typedef T Type;
};

template<typename T>
  struct Veclen {
  };

template<typename T>
struct Veclen<Kokkos::complex<T>> {
  static const int value = 1;
 };

 template<typename T, int N>
  struct Veclen<SIMDComplex<T,N>> { 
  static const int value = N;
 };


}



#endif /* TEST_KOKKOS_KOKKOS_TRAITS_H_ */
