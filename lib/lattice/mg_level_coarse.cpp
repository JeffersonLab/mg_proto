/*
 * mg_level_coarse.cpp
 *
 *  Created on: Aug 29, 2018
 *      Author: bjoo
 */
#include "lattice/mg_level_coarse.h"

namespace MG
{

// These need to be moved into a .cc file. Right now they are with QDPXX (shriek!!!)
  void SetupCoarseToCoarse(const SetupParams& p, std::shared_ptr< const CoarseWilsonCloverLinearOperator > M_fine, int fine_level_id,
              MGLevelCoarse& fine_level, MGLevelCoarse& coarse_level)
  {
	  SetupCoarseToCoarseT<>(p,M_fine,fine_level_id, fine_level, coarse_level);
  }

  void SetupCoarseToCoarse(const SetupParams& p, std::shared_ptr< const CoarseEOWilsonCloverLinearOperator > M_fine, int fine_level_id,
                MGLevelCoarseEO& fine_level, MGLevelCoarseEO& coarse_level)
  {
	  SetupCoarseToCoarseT<>(p,M_fine,fine_level_id, fine_level, coarse_level);
  }
}
