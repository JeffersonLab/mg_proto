/*
 * layout_traits.h
 *
 *  Created on: Nov 10, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_LAYOUT_TRAITS_H_
#define INCLUDE_LATTICE_LAYOUT_TRAITS_H_

namespace MGGeometry {

template<typename L>
struct
LayoutTraits {
	const bool has_subview = false;
	typedef void subview_layout_type;
};



}



#endif /* INCLUDE_LATTICE_LAYOUT_TRAITS_H_ */
