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
#include "lattice/qphix/qphix_eo_clover_linear_operator.h"
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

  template<typename LinOp>
  using MGLevelQPhiXLinOp = MGLevelQPhiXT<QPhiXSpinor,BiCGStabSolverQPhiX,LinOp>;

  template<typename LinOpF>
  using MGLevelQPhiXLinOpF = MGLevelQPhiXT<QPhiXSpinorF,BiCGStabSolverQPhiXF,LinOpF>;


  using MGLevelQPhiX = MGLevelQPhiXT<QPhiXSpinor,BiCGStabSolverQPhiX,QPhiXWilsonCloverLinearOperator>;

  using MGLevelQPhiXF = MGLevelQPhiXT<QPhiXSpinorF,BiCGStabSolverQPhiXF,QPhiXWilsonCloverLinearOperatorF>;

  using MGLevelQPhiXEO= MGLevelQPhiXT<QPhiXSpinor,BiCGStabSolverQPhiX,QPhiXWilsonCloverEOLinearOperator>;

  using MGLevelQPhiXFEO = MGLevelQPhiXT<QPhiXSpinorF,BiCGStabSolverQPhiXF,QPhiXWilsonCloverEOLinearOperatorF>;

  template<typename LinOpF>
  struct QPhiXMultigridLevelsT {
    int n_levels;
    MGLevelQPhiXLinOpF<LinOpF> fine_level;
    std::vector<MGLevelCoarse> coarse_levels;
  };

  using QPhiXMultigridLevels = QPhiXMultigridLevelsT<QPhiXWilsonCloverLinearOperatorF>;
  using QPhiXMultigridLevelsEO = QPhiXMultigridLevelsT<QPhiXWilsonCloverEOLinearOperatorF>;

  // Non EO Versions
  void SetupQPhiXToCoarseGenerateVecs(const SetupParams& p,
		  	  const std::shared_ptr<QPhiXWilsonCloverLinearOperator>& M_fine,
              MGLevelQPhiXLinOp<QPhiXWilsonCloverLinearOperator>& fine_level,
			  MGLevelCoarse& coarse_level);

   void SetupQPhiXToCoarseGenerateVecs(const SetupParams& p,
		   	   const std::shared_ptr<QPhiXWilsonCloverLinearOperatorF>& M_fine,
               MGLevelQPhiXLinOpF<QPhiXWilsonCloverLinearOperatorF>& fine_level,
			   MGLevelCoarse& coarse_level);

   void SetupQPhiXToCoarseVecsIn(const SetupParams& p,
		   	   const std::shared_ptr<QPhiXWilsonCloverLinearOperator>& M_fine,
                MGLevelQPhiXLinOp<QPhiXWilsonCloverLinearOperator>& fine_level,
				MGLevelCoarse& coarse_level);

    void SetupQPhiXToCoarseVecsIn(const SetupParams& p,
    			const std::shared_ptr<QPhiXWilsonCloverLinearOperatorF>& M_fine,
				MGLevelQPhiXLinOpF<QPhiXWilsonCloverLinearOperatorF>& fine_level,
				MGLevelCoarse& coarse_level);

  void SetupQPhiXToCoarse(const SetupParams& p,
		  	  	  const std::shared_ptr<QPhiXWilsonCloverLinearOperator>& M_fine,
              	  MGLevelQPhiXLinOp<QPhiXWilsonCloverLinearOperator>& fine_level,
				  MGLevelCoarse& coarse_level);

  void SetupQPhiXToCoarse(const SetupParams& p,
		  	  	  const std::shared_ptr<QPhiXWilsonCloverLinearOperatorF>& M_fine,
				  MGLevelQPhiXLinOpF<QPhiXWilsonCloverLinearOperatorF>& fine_level,
				  MGLevelCoarse& coarse_level);



  void SetupQPhiXMGLevels(const SetupParams& p, QPhiXMultigridLevels& mg_levels,
        const std::shared_ptr<QPhiXWilsonCloverLinearOperatorF>& M_fine_single);


  // EO Versions
  void SetupQPhiXToCoarseGenerateVecs(const SetupParams& p,
		  const std::shared_ptr<QPhiXWilsonCloverEOLinearOperator>& M_fine,
		  MGLevelQPhiXLinOp<QPhiXWilsonCloverEOLinearOperator>& fine_level,
		  MGLevelCoarse& coarse_level);

     void SetupQPhiXToCoarseGenerateVecs(const SetupParams& p,
    		 const std::shared_ptr<QPhiXWilsonCloverEOLinearOperatorF>& M_fine,
             MGLevelQPhiXLinOpF<QPhiXWilsonCloverEOLinearOperatorF>& fine_level,
			 MGLevelCoarse& coarse_level);

     void SetupQPhiXToCoarseVecsIn(const SetupParams& p,
    		 const std::shared_ptr<QPhiXWilsonCloverEOLinearOperator>& M_fine,
             MGLevelQPhiXLinOp<QPhiXWilsonCloverEOLinearOperator>& fine_level,
			 MGLevelCoarse& coarse_level);

      void SetupQPhiXToCoarseVecsIn(const SetupParams& p,
    		 const std::shared_ptr<QPhiXWilsonCloverEOLinearOperatorF>& M_fine,
             MGLevelQPhiXLinOpF<QPhiXWilsonCloverEOLinearOperatorF>& fine_level,
			 MGLevelCoarse& coarse_level);

    void SetupQPhiXToCoarse(const SetupParams& p,
    		const std::shared_ptr<QPhiXWilsonCloverEOLinearOperator>& M_fine,
			MGLevelQPhiXLinOp<QPhiXWilsonCloverEOLinearOperator>& fine_level,
			MGLevelCoarse& coarse_level);

    void SetupQPhiXToCoarse(const SetupParams& p,
    			const std::shared_ptr<QPhiXWilsonCloverEOLinearOperatorF>& M_fine,
                MGLevelQPhiXLinOpF<QPhiXWilsonCloverEOLinearOperatorF>& fine_level,
				MGLevelCoarse& coarse_level);

  void SetupQPhiXMGLevels(const SetupParams& p,
		  QPhiXMultigridLevelsEO& mg_levels,
          const std::shared_ptr<QPhiXWilsonCloverEOLinearOperatorF>& M_fine);

}


#endif /* INCLUDE_LATTICE_QPHIX_MG_LEVEL_QPHIX_H_ */
