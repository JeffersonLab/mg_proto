/*
 * mr_params.h
 *
 *  Created on: Feb 23, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_MR_PARAMS_H_
#define INCLUDE_LATTICE_MR_PARAMS_H_

#include "lattice/solver.h"

namespace MGTesting {

class MRSolverParams : public MG::LinearSolverParamsBase {
 public:
	  double Omega; // OverRelaxation

  };

};
#endif /* TEST_QDPXX_MR_PARAMS_H_ */
