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


}



#endif /* TEST_KOKKOS_KOKKOS_TRAITS_H_ */
