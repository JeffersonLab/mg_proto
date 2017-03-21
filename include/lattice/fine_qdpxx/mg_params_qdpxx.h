/*
 * mg_params_qdpxx.h
 *
 *  Created on: Mar 21, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_FINE_QDPXX_MG_PARAMS_QDPXX_H_
#define INCLUDE_LATTICE_FINE_QDPXX_MG_PARAMS_QDPXX_H_

#include "lattice/constants.h"
#include "lattice/solver.h"
#include "lattice/mr_params.h"
#include "lattice/fgmres_common.h"

namespace MG {

struct VCycleParams {
	// Pre Smoother Params
	MRSolverParams pre_smoother_params;
	FGMRESParams bottom_solver_params;
	MRSolverParams post_smoother_params;
	LinearSolverParamsBase cycle_params;
};

struct SetupParams {
	int n_levels;
	std::vector<int> n_vecs;
	std::vector< IndexArray > block_sizes;
	std::vector< int > null_solver_max_iter;
	std::vector< double > null_solver_rsd_target;

};

}; // Namespace


#endif /* INCLUDE_LATTICE_FINE_QDPXX_MG_PARAMS_QDPXX_H_ */
