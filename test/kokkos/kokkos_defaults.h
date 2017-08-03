/*
 * kokkos_defaults.h
 *
 *  Created on: May 23, 2017
 *      Author: bjoo
 */

#ifndef TEST_KOKKOS_KOKKOS_DEFAULTS_H_
#define TEST_KOKKOS_KOKKOS_DEFAULTS_H_

#include <Kokkos_Core.hpp>
namespace MG
{
#if defined(KOKKOS_HAVE_CUDA)
  using ExecSpace = Kokkos::Cuda::execution_space;
  using MemorySpace = Kokkos::Cuda::memory_space;
  using Layout = Kokkos::LayoutRight;
  using NeighLayout = Kokkos::Cuda::array_layout;

#else
  using ExecSpace = Kokkos::OpenMP::execution_space;
  using MemorySpace = Kokkos::OpenMP::memory_space;
  using Layout = Kokkos::LayoutRight;
  using NeighLayout = Kokkos::OpenMP::array_layout;
#endif

using ThreadExecPolicy =  Kokkos::TeamPolicy<ExecSpace>;
using TeamHandle =  ThreadExecPolicy::member_type;
using VectorPolicy = Kokkos::Impl::ThreadVectorRangeBoundariesStruct<int,TeamHandle>;
  
}



#endif /* TEST_KOKKOS_KOKKOS_DEFAULTS_H_ */
