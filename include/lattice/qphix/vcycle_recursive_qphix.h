/*
 * vcycle_recursive_qphix.h
 *
 *  Created on: Mar 21, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_QPHIX_VCYCLE_RECURSIVE_QPHIX_H_
#define INCLUDE_LATTICE_QPHIX_VCYCLE_RECURSIVE_QPHIX_H_

#include <vector>
#include <memory>
#include "lattice/solver.h"
#include "lattice/qphix/vcycle_qphix_coarse.h"
#include "lattice/qphix/mg_level_qphix.h"

using namespace QDP;

namespace MG {

class VCycleRecursiveQPhiX :  public LinearSolver<QPhiXSpinor, QPhiXGauge >
{
public:
	VCycleRecursiveQPhiX( const std::vector<VCycleParams>& vcycle_params,
						  const QPhiXMultigridLevels& mg_levels );


	LinearSolverResults operator()(QPhiXSpinor& out, const QPhiXSpinor& in,
	    ResiduumType resid_type = RELATIVE ) const;

private:

	const std::vector<VCycleParams> _vcycle_params;
	const QPhiXMultigridLevels& _mg_levels;

	std::shared_ptr< const Smoother<QPhiXSpinorF,QPhiXGaugeF > > _pre_smoother;
	std::shared_ptr< const Smoother<QPhiXSpinorF,QPhiXGaugeF> > _post_smoother;
	std::shared_ptr< const VCycleQPhiXCoarse2 > _toplevel_vcycle;

	std::vector< std::shared_ptr< const Smoother< CoarseSpinor, CoarseGauge > > >       _coarse_presmoother;
	std::vector< std::shared_ptr< const Smoother< CoarseSpinor, CoarseGauge > > >       _coarse_postsmoother;
	std::vector< std::shared_ptr< const LinearSolver< CoarseSpinor, CoarseGauge > > >   _coarse_vcycle;
	std::vector< std::shared_ptr< const LinearSolver< CoarseSpinor, CoarseGauge > > >   _bottom_solver;

};

class VCycleRecursiveQPhiXEO :  public LinearSolver<QPhiXSpinor, QPhiXGauge >
{
public:
	VCycleRecursiveQPhiXEO( const std::vector<VCycleParams>& vcycle_params,
						  const QPhiXMultigridLevelsEO& mg_levels );


	LinearSolverResults operator()(QPhiXSpinor& out, const QPhiXSpinor& in,
	    ResiduumType resid_type = RELATIVE ) const;

private:

	const std::vector<VCycleParams> _vcycle_params;
	const QPhiXMultigridLevelsEO& _mg_levels;

	std::shared_ptr< const Smoother<QPhiXSpinorF,QPhiXGaugeF > > _pre_smoother;
	std::shared_ptr< const Smoother<QPhiXSpinorF,QPhiXGaugeF> > _post_smoother;
	std::shared_ptr< const VCycleQPhiXCoarseEO2 > _toplevel_vcycle;

	std::vector< std::shared_ptr< const Smoother< CoarseSpinor, CoarseGauge > > >       _coarse_presmoother;
	std::vector< std::shared_ptr< const Smoother< CoarseSpinor, CoarseGauge > > >       _coarse_postsmoother;
	std::vector< std::shared_ptr< const LinearSolver< CoarseSpinor, CoarseGauge > > >   _coarse_vcycle;
	std::vector< std::shared_ptr< const LinearSolver< CoarseSpinor, CoarseGauge > > >   _bottom_solver;

};
};
#endif /* INCLUDE_LATTICE_QPHIX_VCYCLE_RECURSIVE_QPHIX_H_ */
