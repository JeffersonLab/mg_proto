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
#include "lattice/aggregation.h"
#include "utils/memory.h"
#include <memory>

#include "lattice/layouts/cb_soa_spinor_layout.h"
#include "lattice/layouts/block_cb_soa_spinor_layout.h"
#include "lattice/layout_container.h"
#include "lattice/layouts/cb_soa_spinor_specific.h"


namespace MG {

  // Typedefs;
  using LatticeSpinorF = LatticeLayoutContainer<float,CBSOASpinorLayout<float>>;
  using LatticeSpinorD = LatticeLayoutContainer<double,CBSOASpinorLayout<double>>;
  using LatticeSpinorIndex = LatticeLayoutContainer<IndexType,CBSOASpinorLayout<IndexType>>;


  using LatticeBlockSpinorF = AggregateLayoutContainer<float,BlockAggregateVectorLayout<float>>;
  using LatticeBlocSpinorD = AggregateLayoutContainer<double,BlockAggregateVectorLayout<double>>;
  using LatticeBlockSpinorIndex = AggregateLayoutContainer<IndexType,BlockAggregateVectorLayout<IndexType>>;
}




#endif /* INCLUDE_LATTICE_LATTICE_SPINOR_H_ */
