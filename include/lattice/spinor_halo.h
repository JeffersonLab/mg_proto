/*
 * spinor_halo.h
 *
 *  Created on: Mar 7, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_SPINOR_HALO_H_
#define INCLUDE_LATTICE_SPINOR_HALO_H_

#if defined(MG_USE_QMP_SPINOR_HALO)
#include "lattice/spinor_halo_qmp.h"
#else
#include "lattice/spinor_halo_single.h"
#endif

#endif /* INCLUDE_LATTICE_SPINOR_HALO_H_ */
