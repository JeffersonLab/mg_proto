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

#if defined(KOKKOS_HAVE_CUDA)
#include "./my_complex.h"
#endif

namespace MG
{

#if defined(KOKKOS_HAVE_CUDA)
template<typename T>
using MGComplex = Balint::complex<T>;
#else
template<typename T>
using MGComplex = Kokkos::complex<T>;
#endif

#if defined(KOKKOS_HAVE_CUDA)
  using ExecSpace = Kokkos::Cuda::execution_space;
  using MemorySpace = Kokkos::Cuda::memory_space;

#if 0
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
  using GaugeLayout = Kokkos::LayoutRight;
  using NeighLayout = Kokkos::OpenMP::array_layout;
#endif

#if defined(KOKKOS_HAVE_CUDA)
using ThreadExecPolicy =  Kokkos::TeamPolicy<ExecSpace,Kokkos::LaunchBounds<128,1>>;
#else
using ThreadExecPolicy = Kokkos::TeamPolicy<ExecSpace>;
#endif


using TeamHandle =  ThreadExecPolicy::member_type;
using VectorPolicy = Kokkos::Impl::ThreadVectorRangeBoundariesStruct<int,TeamHandle>;

#if defined(KOKKOS_HAVE_CUDA)
  // Try an N-dimensional threading policy for cache blocking
 using MDPolicy =  Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<4,
		 Kokkos::Experimental::Iterate::Left,Kokkos::Experimental::Iterate::Left>>;

#else 
  // Try an N-dimensional threading policy for cache blocking
 using MDPolicy =  Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<4,
		 Kokkos::Experimental::Iterate::Left,Kokkos::Experimental::Iterate::Left>>;

#endif  
}

#if defined(__CUDACC__) 
#define K_ALIGN(N) __align__(N)
#elif defined(__GNUC__) || defined(__INTEL_COMPILER)
#define K_ALIGN(N) __attribute__((aligned(N)))
#else
#error "Unsupported compiler. Please add ALIGN macro declaraiton to kokkos_defaults.h"
#endif


#endif /* TEST_KOKKOS_KOKKOS_DEFAULTS_H_ */
