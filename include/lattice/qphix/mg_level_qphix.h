/*
 * mg_level_qphix.h
 *
 *  Created on: Oct 19, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_QPHIX_MG_LEVEL_QPHIX_H_
#define INCLUDE_LATTICE_QPHIX_MG_LEVEL_QPHIX_H_


#include <lattice/qphix/invbicgstab_qphix.h>
#include <memory>
#include "lattice/qphix/qphix_types.h"
#include "lattice/coarse/block.h"
#include "lattice/mg_level_coarse.h"
#include "lattice/solver.h"

#include "lattice/qphix/qphix_clover_linear_operator.h"

namespace MG {

  template<typename SpinorT, typename SolverT, typename LinOpT>
  struct MGLevelQPhiXT {
    using Spinor = SpinorT;
    using LinOp =  LinOpT;

    std::shared_ptr<const LatticeInfo> info;
    std::vector<std::shared_ptr<SpinorT> > null_vecs; // NULL Vectors -- in single prec
    std::vector<Block> blocklist;
    std::shared_ptr<  const SolverT > null_solver;           // Solver for NULL on this level;
    std::shared_ptr<  LinOpT> M;


  ~MGLevelQPhiXT() {}
  };

  using MGLevelQPhiX = MGLevelQPhiXT<QPhiXSpinor,BiCGStabSolverQPhiX,QPhiXWilsonCloverLinearOperator>;
  using MGLevelQPhiXF = MGLevelQPhiXT<QPhiXSpinorF,BiCGStabSolverQPhiXF,QPhiXWilsonCloverLinearOperatorF>;

  struct QPhiXMultigridLevels {
    int n_levels;
    MGLevelQPhiXF fine_level;
    std::vector<MGLevelCoarse> coarse_levels;
  };

  void SetupQPhiXToCoarseGenerateVecs(const SetupParams& p, std::shared_ptr<QPhiXWilsonCloverLinearOperator> M_fine,
               MGLevelQPhiX& fine_level, MGLevelCoarse& coarse_level);

   void SetupQPhiXToCoarseGenerateVecs(const SetupParams& p, std::shared_ptr<QPhiXWilsonCloverLinearOperatorF> M_fine,
                 MGLevelQPhiXF& fine_level, MGLevelCoarse& coarse_level);

   void SetupQPhiXToCoarseVecsIn(const SetupParams& p, std::shared_ptr<QPhiXWilsonCloverLinearOperator> M_fine,
                MGLevelQPhiX& fine_level, MGLevelCoarse& coarse_level);

    void SetupQPhiXToCoarseVecsIn(const SetupParams& p, std::shared_ptr<QPhiXWilsonCloverLinearOperatorF> M_fine,
                  MGLevelQPhiXF& fine_level, MGLevelCoarse& coarse_level);

  void SetupQPhiXToCoarse(const SetupParams& p, std::shared_ptr<QPhiXWilsonCloverLinearOperator> M_fine,
              MGLevelQPhiX& fine_level, MGLevelCoarse& coarse_level);

  void SetupQPhiXToCoarse(const SetupParams& p, std::shared_ptr<QPhiXWilsonCloverLinearOperatorF> M_fine,
                MGLevelQPhiXF& fine_level, MGLevelCoarse& coarse_level);



  void SetupQPhiXMGLevels(const SetupParams& p, QPhiXMultigridLevels& mg_levels,
        std::shared_ptr<QPhiXWilsonCloverLinearOperatorF> M_fine_single);

}


#endif /* INCLUDE_LATTICE_QPHIX_MG_LEVEL_QPHIX_H_ */
