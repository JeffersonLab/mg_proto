/*
 * vcycle_recursive_qdpxx.h
 *
 *  Created on: Mar 21, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_FINE_QDPXX_VCYCLE_RECURSIVE_QDPXX_H_
#define INCLUDE_LATTICE_FINE_QDPXX_VCYCLE_RECURSIVE_QDPXX_H_

#include <vector>
#include <memory>
#include "lattice/solver.h"
#include "lattice/fine_qdpxx/vcycle_qdpxx_coarse.h"

using namespace QDP;

namespace MG {

class VCycleRecursiveQDPXX :  public LinearSolver<LatticeFermion, multi1d<LatticeColorMatrix> >
{
public:
	VCycleRecursiveQDPXX( const std::vector<VCycleParams>& vcycle_params,
						  const MultigridLevels& mg_levels );


	std::vector<LinearSolverResults> operator()(LatticeFermion& out, const LatticeFermion& in, ResiduumType resid_type = RELATIVE ) const;

	const LatticeInfo& GetInfo() const { return *_mg_levels.fine_level.info; }
	const CBSubset& GetSubset() const { return SUBSET_ALL; }

private:

	const std::vector<VCycleParams> _vcycle_params;
	const MultigridLevels& _mg_levels;

	std::shared_ptr< const Smoother<LatticeFermion,multi1d<LatticeColorMatrix> > > _pre_smoother;
	std::shared_ptr< const Smoother<LatticeFermion,multi1d<LatticeColorMatrix> > > _post_smoother;
	std::shared_ptr< const VCycleQDPCoarse2 > _toplevel_vcycle;

	std::vector< std::shared_ptr< const Smoother< CoarseSpinor, CoarseGauge > > >       _coarse_presmoother;
	std::vector< std::shared_ptr< const Smoother< CoarseSpinor, CoarseGauge > > >       _coarse_postsmoother;
	std::vector< std::shared_ptr< const LinearSolver< CoarseSpinor, CoarseGauge > > >   _coarse_vcycle;
	std::vector< std::shared_ptr< const LinearSolver< CoarseSpinor, CoarseGauge > > >   _bottom_solver;

};

}
#endif /* INCLUDE_LATTICE_FINE_QDPXX_VCYCLE_RECURSIVE_QDPXX_H_ */
