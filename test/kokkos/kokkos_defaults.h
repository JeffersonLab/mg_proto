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
	// Set a Default Layout
	using Layout =  Kokkos::DefaultExecutionSpace::array_layout;
	using RangePolicy = Kokkos::RangePolicy<>;

}



#endif /* TEST_KOKKOS_KOKKOS_DEFAULTS_H_ */
