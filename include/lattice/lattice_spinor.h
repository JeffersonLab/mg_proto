/*
 * spinor.h
 *
 *  Created on: Oct 20, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_LATTICE_SPINOR_H_
#define INCLUDE_LATTICE_LATTICE_SPINOR_H_

#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/geometry_utils.h"
#include "utils/memory.h"
#include <memory>

#include "lattice/compact_cb_aos_spinor_layout.h"
#include "lattice/cb_soa_spinor_layout.h"
#include "lattice/block_cb_soa_spinor_layout.h"
#include "lattice/layout_container.h"

namespace MGGeometry {

  // Typedefs;
#if 0
  using LatticeSpinorF = GenericLayoutContainer<float,CompactCBAOSSpinorLayout, MGUtils::REGULAR>;
  using LatticeSpinorD = GenericLayoutContainer<double,CompactCBAOSSpinorLayout, MGUtils::REGULAR>;
#else
  using LatticeSpinorF = GenericLayoutContainer<float, CBSOASpinorLayout, MGUtils::REGULAR>;
  using LatticeSpinorD = GenericLayoutContainer<double, CBSOASpinorLayout, MGUtils::REGULAR>;
  using LatticeSpinorIndex = GenericLayoutContainer<IndexType, CBSOASpinorLayout, MGUtils::REGULAR>;



  using LatticeBlockSpinorF = GenericLayoutContainer<float,BlockAggregateVectorLayout, MGUtils::REGULAR>;
  using LatticeBlockSpinorIndex = GenericLayoutContainer<IndexType,BlockAggregateVectorLayout, MGUtils::REGULAR>;
  using LatticeBlocSpinorD = GenericLayoutContainer<double, BlockAggregateVectorLayout, MGUtils::REGULAR>;

  using LatticeBlockSpinorArrayF = GenericLayoutContainer<float,BlockAggregateVectorArrayLayout, MGUtils::REGULAR>;
  using LatticeBlockSpinorArrayD = GenericLayoutContainer<double,BlockAggregateVectorArrayLayout, MGUtils::REGULAR>;
  using LatticeBlockSpinorArrayIndex = GenericLayoutContainer<IndexType,BlockAggregateVectorArrayLayout, MGUtils::REGULAR>;

  #endif



}




#endif /* INCLUDE_LATTICE_LATTICE_SPINOR_H_ */
