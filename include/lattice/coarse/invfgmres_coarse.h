/*
 * invfgmres_coarse.h
 *
 *  Created on: Oct 17, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_COARSE_INVFGMRES_COARSE_H_
#define INCLUDE_LATTICE_COARSE_INVFGMRES_COARSE_H_

#include  "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_l1_blas.h"
#include  "lattice/invfgmres_generic.h"
#include  "lattice/unprec_solver_wrappers.h"
namespace MG {

  using FGMRESSolverCoarse = FGMRESGeneric::FGMRESSolverGeneric<CoarseSpinor,CoarseGauge>;

  using UnprecFGMRESSolverCoarseWrapper =  UnprecLinearSolverWrapper<CoarseSpinor,CoarseGauge,
		  	  	  	  	  	  	  	  	  	  	  	  	  FGMRESGeneric::FGMRESSolverGeneric<CoarseSpinor,CoarseGauge>>;

}




#endif /* INCLUDE_LATTICE_COARSE_INVFGMRES_COARSE_H_ */
