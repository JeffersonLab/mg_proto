/*
 * basic_typetraits.h
 *
 *  Created on: Dec 8, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_BASIC_TYPETRAITS_H_
#define INCLUDE_LATTICE_BASIC_TYPETRAITS_H_

#include <type_traits>
#include "lattice/constants.h"

namespace MG {
//* Some basic traits about basic types */

template<typename T,  class Enable=void>
struct SizeTraits;

template<typename T>
struct SizeTraits<T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {

	static
	inline
	constexpr
	IndexType IndexSpaceSize(void) { return 1; }

	static
	inline
	constexpr
	size_t MemorySize(void) { return sizeof(T); }
};


}


#endif /* INCLUDE_LATTICE_BASIC_TYPETRAITS_H_ */
