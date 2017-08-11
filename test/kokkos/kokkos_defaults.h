/*
 * kokkos_defaults.h
 *
 *  Created on: May 23, 2017
 *      Author: bjoo
 */

#ifndef TEST_KOKKOS_KOKKOS_DEFAULTS_H_
#define TEST_KOKKOS_KOKKOS_DEFAULTS_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include "./my_complex.h"


namespace MG
{
 

template<typename T>
using MGComplex = Balint::complex<T>;

#if defined(KOKKOS_HAVE_CUDA)
  using ExecSpace = Kokkos::Cuda::execution_space;
  using MemorySpace = Kokkos::Cuda::memory_space;

#if 1
  using Layout = Kokkos::LayoutRight;
  using GaugeLayout = Kokkos::LayoutRight;
  using NeighLayout = Kokkos::LayoutRight;
#else

  using Layout = Kokkos::LayoutLeft;
  using GaugeLayout = Kokkos::LayoutLeft;
  using NeighLayout = Kokkos::LayoutLeft;
#endif

#else
  using ExecSpace = Kokkos::OpenMP::execution_space;
  using MemorySpace = Kokkos::OpenMP::memory_space;
  using Layout = Kokkos::LayoutRight;
  using NeighLayout = Kokkos::OpenMP::array_layout;
#endif

using ThreadExecPolicy =  Kokkos::TeamPolicy<ExecSpace,Kokkos::LaunchBounds<128,1>>;
//using ThreadExecPolicy =  Kokkos::TeamPolicy<ExecSpace>;
using TeamHandle =  ThreadExecPolicy::member_type;
using VectorPolicy = Kokkos::Impl::ThreadVectorRangeBoundariesStruct<int,TeamHandle>;
  
}

#if defined(__CUDACC__) 
#define K_ALIGN(N) __align__(N)
#elif defined(__GNUC__) || defined(__INTEL_COMPILER)
#define K_ALIGN(N) __attribute__((aligned(N)))
#else
#error "Unsupported compiler. Please add ALIGN macro declaraiton to kokkos_defaults.h"
#endif


#endif /* TEST_KOKKOS_KOKKOS_DEFAULTS_H_ */
