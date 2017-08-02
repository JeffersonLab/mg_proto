#pragma once
#ifndef MY_COMPLEX_H
#define MY_COMPLEX_H

#if 0

#include <complex>

namespace MG { 
template<typename T>
using MyComplex = std::complex<T>;
}
using std::conj;

#else 

#include "Kokkos_Complex.hpp"
namespace MG {
template<typename T>
using MyComplex = Kokkos::complex<T>;
using Kokkos::conj;
}

#endif
#endif
