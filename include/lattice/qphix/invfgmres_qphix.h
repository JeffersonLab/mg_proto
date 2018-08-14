/*
 * invfgmres_qphix.h
 *
 *  Created on: Oct 17, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_QPHIX_INVFGMRES_QPHIX_H_
#define INCLUDE_LATTICE_QPHIX_INVFGMRES_QPHIX_H_

#include  "lattice/qphix/qphix_types.h"
#include  "lattice/qphix/qphix_blas_wrappers.h"
#include  "lattice/invfgmres_generic.h"
#include  "lattice/unprec_wrapper.h"

namespace MG {

  using FGMRESSolverQPhiX = FGMRESGeneric::FGMRESSolverGeneric<QPhiXSpinor,QPhiXGauge>;
  using FGMRESSolverQPhiXF = FGMRESGeneric::FGMRESSolverGeneric<QPhiXSpinorF,QPhiXGaugeF>;

  using UnprecFGMRESSolverQPhiXWrapper =  UnprecWrapper<QPhiXSpinor,QPhiXGauge,FGMRESGeneric::FGMRESSolverGeneric<QPhiXSpinor,QPhiXGauge>,UnprecLinearSolver>;
  using UnprecFGMRESSolverQPhiXFWrapper =  UnprecWrapper<QPhiXSpinorF,QPhiXGaugeF,FGMRESGeneric::FGMRESSolverGeneric<QPhiXSpinor,QPhiXGauge>,UnprecLinearSolver>;

}



#endif /* INCLUDE_LATTICE_QPHIX_INVFGMRES_QPHIX_H_ */
