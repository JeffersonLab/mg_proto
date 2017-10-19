/*
 * mg_level_qphix.h
 *
 *  Created on: Oct 19, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_QPHIX_MG_LEVEL_QPHIX_H_
#define INCLUDE_LATTICE_QPHIX_MG_LEVEL_QPHIX_H_


#include <lattice/coarse/invbicgstab_qphix.h>
#include <memory>
#include "lattice/qphix/qphix_types.h"
#include "lattice/coarse/block.h"
#include "lattice/solver.h"

#include "lattice/qphix/qphix_clover_linear_operator.h"

namespace MG {
  struct MGLevelQPhiX {
  std::shared_ptr<const LatticeInfo> info;
  std::shared_ptr<QPhiXGaugeF> gauge;
  std::vector<std::shared_ptr<QPhiXSpinorF> > null_vecs; // NULL Vectors -- in single prec
  std::vector<Block> blocklist;
  std::shared_ptr< const BiCGStabSolverQPhiXF > null_solver;           // Solver for NULL on this level;
  std::shared_ptr< const QPhiXWilsonCloverLinearOperatorF > M;

  ~MGLevelQPhiX() {}
};
}


#endif /* INCLUDE_LATTICE_QPHIX_MG_LEVEL_QPHIX_H_ */
