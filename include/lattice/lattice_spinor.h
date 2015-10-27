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
#include "lattice/generic_spinor.h"

namespace MGGeometry {

  // Typedefs;
#if 0
  using LatticeSpinorF = GeneralLatticeSpinor<float,CompactCBAOSSpinorLayout<float>, MGUtils::REGULAR>;
  using LatticeSpinorD = GeneralLatticeSpinor<double,CompactCBAOSSpinorLayout<double>, MGUtils::REGULAR>;
#else
  using LatticeSpinorF = GeneralLatticeSpinor<float, CBSOASpinorLayout<float>, MGUtils::REGULAR>;
  using LatticeSpinorD = GeneralLatticeSpinor<double, CBSOASpinorLayout<double>, MGUtils::REGULAR>;
#endif



}




#endif /* INCLUDE_LATTICE_LATTICE_SPINOR_H_ */
