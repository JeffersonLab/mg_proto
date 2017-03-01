/*
 * fgmres_common.h
 *
 *  Created on: Jan 12, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_FGMRES_COMMON_H_
#define INCLUDE_LATTICE_FGMRES_COMMON_H_

#include "lattice/constants.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"

namespace MG {

//! Params for FGMRESDR inverter
 /*! \ingroup invert */
 struct FGMRESParams : public MG::LinearSolverParamsBase {
 public:
	 int NKrylov;
 };

};




#endif /* TEST_QDPXX_FGMRES_COMMON_H_ */
