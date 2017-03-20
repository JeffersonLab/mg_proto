/*
 * mg_level_qdpxx.h
 *
 *  Created on: Mar 15, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_FINE_QDPXX_MG_LEVEL_QDPXX_H_
#define INCLUDE_LATTICE_FINE_QDPXX_MG_LEVEL_QDPXX_H_

#include "lattice/linear_operator.h"
#include "lattice/solver.h"
#include "lattice/coarse/block.h"
#include "lattice/mg_level_coarse.h"
#include "lattice/linear_operator.h"
#include "lattice/fine_qdpxx/wilson_clover_linear_operator.h"
#include <memory>

namespace MG {

struct MGLevelQDPXX {
	std::shared_ptr<LatticeInfo> info;
	QDP::multi1d<QDP::LatticeFermion> null_vecs;
	std::shared_ptr< LinearSolver< QDP::LatticeFermion,QDP::multi1d<QDP::LatticeColorMatrix> > > null_solver;           // Solver for NULL on this level
	std::shared_ptr< LinearSolver< QDP::LatticeFermion,QDP::multi1d<QDP::LatticeColorMatrix> > > pre_smoother;
	std::shared_ptr< LinearSolver< QDP::LatticeFermion,QDP::multi1d<QDP::LatticeColorMatrix> > > post_smoother;
	std::shared_ptr< QDPWilsonCloverLinearOperator > M;
	std::vector<Block> blocklist;

	~MGLevelQDPXX() {}
};

struct MultigridLevels {
	int n_levels;
	MGLevelQDPXX fine_level;
	std::vector<MGLevelCoarse> coarse_levels;
};

struct SetupParams {
	int n_levels;
	std::vector<int> n_vecs;
	IndexArray local_lattice_size;
	std::vector< IndexArray > block_sizes;
	std::vector< int > null_solver_max_iter;
	std::vector< double > null_solver_rsd_target;

};

void SetupQDPXXToCoarse(const SetupParams& p, std::shared_ptr<QDPWilsonCloverLinearOperator> M_fine,
						MGLevelQDPXX& fine_level, MGLevelCoarse& coarse_level);

void SetupCoarseToCoarse(const SetupParams& p, std::shared_ptr< CoarseWilsonCloverLinearOperator > M_fine, int fine_level_id,
						MGLevelCoarse& fine_level, MGLevelCoarse& coarse_level);

void SetupMGLevels(const SetupParams& p, MultigridLevels& mg_levels,
			std::shared_ptr<QDPWilsonCloverLinearOperator> M_fine);


}



#endif /* INCLUDE_LATTICE_FINE_QDPXX_MG_LEVEL_QDPXX_H_ */
