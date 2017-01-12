/*
 * fgmres_common.h
 *
 *  Created on: Jan 12, 2017
 *      Author: bjoo
 */

#ifndef TEST_QDPXX_FGMRES_COMMON_H_
#define TEST_QDPXX_FGMRES_COMMON_H_

#include "qdp.h"
#include "lattice/constants.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"

namespace MGTesting {

//! Params for FGMRESDR inverter
 /*! \ingroup invert */
 struct FGMRESParams : public MG::LinearSolverParamsBase {
 public:
	 int NKrylov;
 };

};




#endif /* TEST_QDPXX_FGMRES_COMMON_H_ */
