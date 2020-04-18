/*
 * vcycle_qdpxx_coarse.h
 *
 *  Created on: Jan 13, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_COARSE_VCYCLE_COARSE_H_
#define INCLUDE_LATTICE_COARSE_VCYCLE_COARSE_H_

#include "MG_config.h"

#include <lattice/coarse/invfgmres_coarse.h>
#include <lattice/coarse/invmr_coarse.h>
#include "lattice/constants.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"
#include "lattice/fgmres_common.h"
#include "lattice/coarse/aggregate_block_coarse.h"
#include "lattice/coarse/coarse_transfer.h"
#include "utils/print_utils.h"
#include "lattice/coarse/subset.h"

#ifdef MG_ENABLE_TIMERS
#include "utils/timer.h"
#endif

namespace MG {

class VCycleCoarse : public LinearSolver<CoarseSpinor,CoarseGauge>
{
public:
	std::vector<LinearSolverResults> operator()(CoarseSpinor& out, const CoarseSpinor& in, ResiduumType resid_type = RELATIVE ) const
	{
		assert(out.GetNCol() == in.GetNCol());
		IndexType ncol = out.GetNCol();

		const LatticeInfo& info=out.GetInfo();
		{
			const LatticeInfo& info_in = in.GetInfo();
			const LatticeInfo& M_info = _M_fine.GetInfo();
			AssertCompatible(info, info_in);
			AssertCompatible(info, M_info);
		}

		std::vector<LinearSolverResults> res(ncol);

		CoarseSpinor tmp(info, ncol);  // Use these to compute residua
		CoarseSpinor r(info, ncol);    //

		int level = _M_fine.GetLevel();

		std::vector<double> norm2_in(ncol);


		// Initialize
		ZeroVec(out);  // Work with zero intial guess
		CopyVec(r,in);
		std::vector<double> norm2_r = Norm2Vec(r);
		norm2_in = norm2_r;

		std::vector<double> target(ncol, _param.RsdTarget);
		bool continueP = false;
		for (int col=0; col < ncol; ++col) {
			if ( resid_type == RELATIVE ) {
				target[col] *= std::sqrt(norm2_r[col]);
			}
			if ( std::sqrt( norm2_r[col]) > target[col]) continueP = true;
		}

		// Check if converged already
		if ( !continueP || _param.MaxIter <= 0  ) {
			for (int col=0; col < ncol; ++col) {
				res[col].resid_type = resid_type;
				res[col].n_count = 0;
				res[col].resid = std::sqrt(norm2_r[col]);
				if( resid_type == RELATIVE ) {
					res[col].resid /= std::sqrt(norm2_r[col]);
				}
			}
			return res;
		}

		if( _param.VerboseP ) {
			for (int col=0; col < ncol; ++col) {
				if( resid_type == RELATIVE) {
					MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d col=%d"
							"Initial || r ||/|| b || =%16.8e  Target=%16.8e",
							level, col, std::sqrt(norm2_r[col]/norm2_in[col]), _param.RsdTarget);

				}
				else {
					MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d col=%d"
							"Initial || r ||=%16.8e  Target=%16.8e",
							level, col, std::sqrt(norm2_r[col]), _param.RsdTarget);
				}
			}
		}

		// At this point we have to do at least one iteration
		int iter = 0;

		while ( iter < _param.MaxIter) {

			++iter;


			CoarseSpinor delta(info, ncol);
			ZeroVec(delta);

			// Smoother does not compute a residuum
			_pre_smoother(delta,r);

			// Update solution
			// out += delta;
			YpeqxVec(delta,out);

			// Update residuum
			_M_fine(tmp,delta, LINOP_OP);

			// r -= tmp;
			YmeqxVec(tmp,r);

			if ( _param.VerboseP ) {
				std::vector<double> norm2_pre_presmooth=Norm2Vec(r);
				for (int col=0; col < ncol; ++col) {
					if( resid_type == RELATIVE ) {
						MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d col=%d "
								"After Pre-Smoothing || r ||/|| b ||=%16.8e Target=%16.8e",
								level, iter, col, std::sqrt(norm2_pre_presmooth[col]/norm2_in[col]), _param.RsdTarget);
					}
					else {
						MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d col=%d"
								"After Pre-Smoothing || r ||=%16.8e Target=%16.8e",
								level, iter, col, std::sqrt(norm2_pre_presmooth[col]), _param.RsdTarget);
					}
				}
			}

			CoarseSpinor coarse_in(_coarse_info, ncol);

			// Coarsen r
			_Transfer.R(r,coarse_in);

			CoarseSpinor coarse_delta(_coarse_info, ncol);
			ZeroVec(coarse_delta);
			_bottom_solver(coarse_delta,coarse_in);

			// Reuse Smoothed Delta as temporary for prolongating coarse delta back to fine
			_Transfer.P(coarse_delta, delta);

			// Update solution
			//			out += delta;
			YpeqxVec(delta,out);

			// Update residuum
			_M_fine(tmp, delta, LINOP_OP);
			// r -= tmp;
			YmeqxVec(tmp,r);

			if ( _param.VerboseP ) {
				std::vector<double> norm2_pre_postsmooth=Norm2Vec(r);
				for (int col=0; col < ncol; ++col) {
					if( resid_type == RELATIVE ) {
						MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d col=%d "
								"After Coarse Solve || r ||/|| b ||=%16.8e Target=%16.8e",
								level, iter, col, std::sqrt(norm2_pre_postsmooth[col]/norm2_in[col]), _param.RsdTarget);
					}
					else {
						MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d col=%d "
								"After Coarse Solve || r ||=%16.8e Target=%16.8e",
								level, iter, col, std::sqrt(norm2_pre_postsmooth[col]), _param.RsdTarget);
					}
				}
			}

			// delta = zero;
			ZeroVec(delta);
			_post_smoother(delta,r);

			// Update full solution
			// out += delta;
			YpeqxVec(delta,out);
			_M_fine(tmp,delta,LINOP_OP);
			//r -= tmp;
			YmeqxVec(tmp,r);
			norm2_r = Norm2Vec(r);

			if( _param.VerboseP ) {
				for (int col=0; col < ncol; ++col) {
					if( resid_type == RELATIVE ) {
						MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d col=%d "
								"After Post-Smoothing || r ||/|| b ||=%16.8e Target=%16.8e",
								level, iter, col, std::sqrt(norm2_r[col]/norm2_in[col]), _param.RsdTarget);
					}
					else {
						MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d "
								"After Post-Smoothing || r ||=%16.8e Target=%16.8e",
								level, iter, std::sqrt(norm2_r[col]), _param.RsdTarget);
					}
				}
			}

			// Check convergence
			continueP = false;
			for (int col=0; col < ncol; ++col)
				if ( std::sqrt(norm2_r[col]) >= target[col] ) continueP = true;
			if (!continueP) break;
		}

		for (int col=0; col < ncol; ++col)  {
			res[col].resid_type = resid_type;
			res[col].n_count = iter;
			res[col].resid= std::sqrt(norm2_r[col]);
			if( resid_type == RELATIVE ) {
				res[col].resid /= std::sqrt(norm2_in[col]);
			}
		}
		return res;
	}

	VCycleCoarse(const LatticeInfo& coarse_info,
				const std::vector<Block>& my_blocks,
				const std::vector<std::shared_ptr<CoarseSpinor> >& vecs,
					 const LinearOperator<CoarseSpinor, CoarseGauge>& M_fine,
					 const Smoother<CoarseSpinor,CoarseGauge>& pre_smoother,
					 const Smoother<CoarseSpinor,CoarseGauge>& post_smoother,
					 const LinearSolver<CoarseSpinor,CoarseGauge>& bottom_solver,
					 const LinearSolverParamsBase& param) : _coarse_info(coarse_info),
							 	 	 	 	 	 	 	 	_my_blocks(my_blocks),
															_vecs(vecs),
															_M_fine(M_fine),
															_pre_smoother(pre_smoother),
															_post_smoother(post_smoother),
															_bottom_solver(bottom_solver),
															_param(param),
															_Transfer(my_blocks,vecs){}


private:
	const LatticeInfo& _coarse_info;
	const std::vector<Block>& _my_blocks;
	const std::vector< std::shared_ptr<CoarseSpinor> >& _vecs;
	const LinearOperator< CoarseSpinor, CoarseGauge>& _M_fine;
	const Smoother<CoarseSpinor,CoarseGauge>& _pre_smoother;
	const Smoother<CoarseSpinor,CoarseGauge>& _post_smoother;
	const LinearSolver<CoarseSpinor,CoarseGauge>& _bottom_solver;
	const LinearSolverParamsBase& _param;
	const CoarseTransfer _Transfer;
};

class VCycleCoarseEO : public LinearSolver<CoarseSpinor,CoarseGauge>
{
public:
	std::vector<LinearSolverResults> operator()(CoarseSpinor& out, const CoarseSpinor& in, ResiduumType resid_type = RELATIVE ) const
	{

		assert(out.GetNCol() == in.GetNCol());
		IndexType ncol = out.GetNCol();

		const LatticeInfo& info=out.GetInfo();
		{
			const LatticeInfo& info_in = in.GetInfo();
			const LatticeInfo& M_info = _M_fine.GetInfo();
			AssertCompatible(info, info_in);
			AssertCompatible(info, M_info);
		}

		std::vector<LinearSolverResults> res(ncol);

		CoarseSpinor tmp(info, ncol);  // Use these to compute residua
		CoarseSpinor r(info, ncol);    //

		int level = _M_fine.GetLevel();
		const CBSubset& subset = _M_fine.GetSubset();

		std::vector<double> norm2_in, norm2_r;


		// Initialize
		ZeroVec(out);  // Work with zero intial guess
		ZeroVec(r);

		// We are using a coarse EO preconditioner where
		// M [ x_e ] = [   0      ]
		//   [ x_o ] = [ A_oo b_o ]
		//
		// gives the same solution as S x_o = b_o
		// THe solver above doesn't know this. It just calls the preconditioner.
		// So we prep the RHS here.
		_M_fine.M_diag(r,in, ODD);

		// Typically we will always do this on the full vectors since this is
		// an 'unprec' operation until we exit but here we know r_e = 0 so
		// we can norm only on the subset
		norm2_r = Norm2Vec(r,SUBSET_ODD);
		norm2_in = norm2_r;

		std::vector<double> target(ncol, _param.RsdTarget);
		bool continueP = false;
		for (int col=0; col < ncol; ++col) {
			if ( resid_type == RELATIVE ) {
				target[col] *= std::sqrt(norm2_r[col]);
			}
			if ( std::sqrt( norm2_r[col]) > target[col]) continueP = true;
		}

		// Check if converged already
		if ( !continueP || _param.MaxIter <= 0  ) {
			for (int col=0; col < ncol; ++col) {
				res[col].resid_type = resid_type;
				res[col].n_count = 0;
				res[col].resid = std::sqrt(norm2_r[col]);
				if( resid_type == RELATIVE ) {
					res[col].resid /= std::sqrt(norm2_r[col]);
				}
			}
			return res;
		}

		if( _param.VerboseP ) {
			for (int col=0; col < ncol; ++col) {
				if( resid_type == RELATIVE) {
					MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d col=%d"
							"Initial || r ||/|| b || =%16.8e  Target=%16.8e",
							level, col, std::sqrt(norm2_r[col]/norm2_in[col]), _param.RsdTarget);

				}
				else {
					MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d col=%d"
							"Initial || r ||=%16.8e  Target=%16.8e",
							level, col, std::sqrt(norm2_r[col]), _param.RsdTarget);
				}
			}
		}

		// At this point we have to do at least one iteration
		int iter = 0;

		while ( iter < _param.MaxIter) {

			++iter;


			CoarseSpinor delta(info, ncol);
			ZeroVec(delta);

			// Smoother does not compute a residuum
			// It is an 'unprec' smoother tho it may use internally a wrapped even-odd
			_pre_smoother(delta,r);

			// Update solution
			// out += delta;
			YpeqxVec(delta,out);

			// Update unprec residuum
			_M_fine.unprecOp(tmp,delta, LINOP_OP);

			// r -= tmp;
			YmeqxVec(tmp,r);

			if ( _param.VerboseP ) {
				std::vector<double> norm2_pre_presmooth=Norm2Vec(r);
				for (int col=0; col < ncol; ++col) {
					if( resid_type == RELATIVE ) {
						MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d col=%d"
								"After Pre-Smoothing || r ||/|| b ||=%16.8e Target=%16.8e",
								level, iter, col, std::sqrt(norm2_pre_presmooth[col]/norm2_in[col]), _param.RsdTarget);
					}
					else {
						MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d col=%d"
								"After Pre-Smoothing || r ||=%16.8e Target=%16.8e",
								level, iter, col, std::sqrt(norm2_pre_presmooth[col]), _param.RsdTarget);
					}
				}
			}

			CoarseSpinor coarse_in(_coarse_info, ncol);

			// Coarsen r
			_Transfer.R(r,coarse_in);

			CoarseSpinor coarse_delta(_coarse_info);
			ZeroVec(coarse_delta);

			// Again, this is an unprec solve, tho it may be a wrapped even-odd
			_bottom_solver(coarse_delta,coarse_in);

			// Reuse Smoothed Delta as temporary for prolongating coarse delta back to fine
			_Transfer.P(coarse_delta, delta);

			// Update solution
			//			out += delta;
			YpeqxVec(delta,out);

			// Update residuum
			_M_fine.unprecOp(tmp, delta, LINOP_OP);
			// r -= tmp;
			YmeqxVec(tmp,r);

			if ( _param.VerboseP ) {
				std::vector<double> norm2_pre_postsmooth=Norm2Vec(r);
				for (int col=0; col < ncol; ++col) {
					if( resid_type == RELATIVE ) {
						MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d col=%d "
								"After Coarse Solve || r ||/|| b ||=%16.8e Target=%16.8e",
								level, iter, col, std::sqrt(norm2_pre_postsmooth[col]/norm2_in[col]), _param.RsdTarget);
					}
					else {
						MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d col=%d "
								"After Coarse Solve || r ||=%16.8e Target=%16.8e",
								level, iter, col, std::sqrt(norm2_pre_postsmooth[col]), _param.RsdTarget);
					}
				}
			}

			// delta = zero;
			ZeroVec(delta);
			_post_smoother(delta,r);

			// Update full solution
			// out += delta;
			YpeqxVec(delta,out);
			_M_fine.unprecOp(tmp,delta,LINOP_OP);
			//r -= tmp;
			YmeqxVec(tmp,r);
			norm2_r = Norm2Vec(r);

			if( _param.VerboseP ) {
				for (int col=0; col < ncol; ++col) {
					if( resid_type == RELATIVE ) {
						MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d col=%d "
								"After Post-Smoothing || r ||/|| b ||=%16.8e Target=%16.8e",
								level, iter, col, std::sqrt(norm2_r[col]/norm2_in[col]), _param.RsdTarget);
					}
					else {
						MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d "
								"After Post-Smoothing || r ||=%16.8e Target=%16.8e",
								level, iter, std::sqrt(norm2_r[col]), _param.RsdTarget);
					}
				}
			}


			// Check convergence
			continueP = false;
			for (int col=0; col < ncol; ++col)
				if ( std::sqrt(norm2_r[col]) >= target[col] ) continueP = true;
			if (!continueP) break;
		}

		for (int col=0; col < ncol; ++col)  {
			res[col].resid_type = resid_type;
			res[col].n_count = iter;
			res[col].resid= std::sqrt(norm2_r[col]);
			if( resid_type == RELATIVE ) {
				res[col].resid /= std::sqrt(norm2_in[col]);
			}
		}

		// We only need the odd part of the result since we are preconditioning a Schur System.
		ZeroVec(out,SUBSET_EVEN); // (keep subset odd)
		return res;
	}

	VCycleCoarseEO(const LatticeInfo& coarse_info,
				const std::vector<Block>& my_blocks,
				const std::vector<std::shared_ptr<CoarseSpinor> >& vecs,
					 const EOLinearOperator<CoarseSpinor, CoarseGauge>& M_fine,
					 const Smoother<CoarseSpinor,CoarseGauge>& pre_smoother,
					 const Smoother<CoarseSpinor,CoarseGauge>& post_smoother,
					 const LinearSolver<CoarseSpinor,CoarseGauge>& bottom_solver,
					 const LinearSolverParamsBase& param) : _coarse_info(coarse_info),
							 	 	 	 	 	 	 	 	_my_blocks(my_blocks),
															_vecs(vecs),
															_M_fine(M_fine),
															_pre_smoother(pre_smoother),
															_post_smoother(post_smoother),
															_bottom_solver(bottom_solver),
															_param(param),
															_Transfer(my_blocks,vecs){}


private:
	const LatticeInfo& _coarse_info;
	const std::vector<Block>& _my_blocks;
	const std::vector< std::shared_ptr<CoarseSpinor> >& _vecs;
	const EOLinearOperator< CoarseSpinor, CoarseGauge>& _M_fine;
	const Smoother<CoarseSpinor,CoarseGauge>& _pre_smoother;
	const Smoother<CoarseSpinor,CoarseGauge>& _post_smoother;
	const LinearSolver<CoarseSpinor,CoarseGauge>& _bottom_solver;
	const LinearSolverParamsBase& _param;
	const CoarseTransfer _Transfer;
};

class VCycleCoarseEO2 : public LinearSolver<CoarseSpinor,CoarseGauge>
{
public:
	std::vector<LinearSolverResults> operator()(CoarseSpinor& out, const CoarseSpinor& in, ResiduumType resid_type = RELATIVE ) const
	{
		assert(out.GetNCol() == in.GetNCol());
		IndexType ncol = out.GetNCol();

		int level = _M_fine.GetLevel();

		Timer::TimerAPI::startTimer("VCycleCoarseEO2/operator()/level"+std::to_string(level));
		const LatticeInfo& info=out.GetInfo();
		{
			const LatticeInfo& info_in = in.GetInfo();
			const LatticeInfo& M_info = _M_fine.GetInfo();
			AssertCompatible(info, info_in);
			AssertCompatible(info, M_info);
		}

		std::vector<LinearSolverResults> res;

		CoarseSpinor tmp(info, ncol);  // Use these to compute residua
		CoarseSpinor r(info, ncol);    //


		const CBSubset& subset = _M_fine.GetSubset();

		std::vector<double> norm2_in(ncol), norm2_r(ncol);


		// Initialize
		ZeroVec(out,subset);  // Work with zero intial guess
		ZeroVec(r);

		CopyVec(r,in,subset);


		// We are using a coarse EO2 preconditioner where
		// we smooth with an unwrapped M_fine

		// Typically we will always do this on the full vectors since this is
		// an 'unprec' operation until we exit but here we know r_e = 0 so
		// we can norm only on the subset
		norm2_r = Norm2Vec(r,SUBSET_ODD);
		norm2_in = norm2_r;

		std::vector<double> target(ncol, _param.RsdTarget);
		bool continueP = false;
		for (int col=0; col < ncol; ++col) {
			if ( resid_type == RELATIVE ) {
				target[col] *= std::sqrt(norm2_r[col]);
			}
			if ( std::sqrt( norm2_r[col]) > target[col]) continueP = true;
		}

		// Check if converged already
		if ( !continueP || _param.MaxIter <= 0  ) {
			for (int col=0; col < ncol; ++col) {
				res[col].resid_type = resid_type;
				res[col].n_count = 0;
				res[col].resid = std::sqrt(norm2_r[col]);
				if( resid_type == RELATIVE ) {
					res[col].resid /= std::sqrt(norm2_r[col]);
				}
			}
			Timer::TimerAPI::stopTimer("VCycleCoarseEO2/operator()/level"+std::to_string(level));
			return res;
		}

		if( _param.VerboseP ) {
			for (int col=0; col < ncol; ++col) {
				if( resid_type == RELATIVE) {
					MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d col=%d"
							"Initial || r ||/|| b || =%16.8e  Target=%16.8e",
							level, col, std::sqrt(norm2_r[col]/norm2_in[col]), _param.RsdTarget);

				}
				else {
					MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d col=%d"
							"Initial || r ||=%16.8e  Target=%16.8e",
							level, col, std::sqrt(norm2_r[col]), _param.RsdTarget);
				}
			}
		}


		// At this point we have to do at least one iteration
		int iter = 0;

		while ( iter < _param.MaxIter) {

			++iter;


			CoarseSpinor delta(info, ncol);
			Timer::TimerAPI::startTimer("VCycleCoarseEO2/presmooth/level"+std::to_string(level));
			ZeroVec(delta,subset);
			// Smoother does not compute a residuum
			// It is an 'unprec' smoother tho it may use internally a wrapped even-odd
			_pre_smoother(delta,r);
			Timer::TimerAPI::stopTimer("VCycleCoarseEO2/presmooth/level"+std::to_string(level));


			Timer::TimerAPI::startTimer("VCycleCoarseEO2/update/level"+std::to_string(level));
			// Update solution
			// out += delta;
			YpeqxVec(delta,out,subset);
			// Update prec residuum
			_M_fine(tmp,delta, LINOP_OP);
			// r -= tmp;
			YmeqxVec(tmp,r,subset);
			Timer::TimerAPI::stopTimer("VCycleCoarseEO2/update/level"+std::to_string(level));

			if ( _param.VerboseP ) {
				std::vector<double> norm2_pre_presmooth=Norm2Vec(r);
				for (int col=0; col < ncol; ++col) {
					if( resid_type == RELATIVE ) {
						MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d col=%d"
								"After Pre-Smoothing || r ||/|| b ||=%16.8e Target=%16.8e",
								level, iter, col, std::sqrt(norm2_pre_presmooth[col]/norm2_in[col]), _param.RsdTarget);
					}
					else {
						MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d col=%d"
								"After Pre-Smoothing || r ||=%16.8e Target=%16.8e",
								level, iter, col, std::sqrt(norm2_pre_presmooth[col]), _param.RsdTarget);
					}
				}
			}


			Timer::TimerAPI::startTimer("VCycleCoarseEO2/restrictFrom/level"+std::to_string(level));
			CoarseSpinor coarse_in(_coarse_info, ncol);
			_Transfer.R(r,ODD,coarse_in);
			 Timer::TimerAPI::stopTimer("VCycleCoarseEO2/restrictFrom/level"+std::to_string(level));
			CoarseSpinor coarse_delta(_coarse_info, ncol);
            
			Timer::TimerAPI::startTimer("VCycleCoarseEO2/bottom_solve/level"+std::to_string(level));
			ZeroVec(coarse_delta);
			// Again, this is an unprec solve, tho it may be a wrapped even-odd
			_bottom_solver(coarse_delta,coarse_in);
			Timer::TimerAPI::stopTimer("VCycleCoarseEO2/bottom_solve/level"+std::to_string(level));

			Timer::TimerAPI::startTimer("VCycleCoarseEO2/prolongateTo/level"+std::to_string(level));
			// Reuse Smoothed Delta as temporary for prolongating coarse delta back to fine
			_Transfer.P(coarse_delta, ODD, delta);
			Timer::TimerAPI::stopTimer("VCycleCoarseEO2/prolongateTo/level"+std::to_string(level));

			Timer::TimerAPI::startTimer("VCycleCoarseEO2/update/level"+std::to_string(level));
			// Update solution
			//			out += delta;
			YpeqxVec(delta,out, subset);
			// Update residuum
			_M_fine(tmp, delta, LINOP_OP);
			// r -= tmp;
			YmeqxVec(tmp,r,subset);
			Timer::TimerAPI::stopTimer("VCycleCoarseEO2/update/level"+std::to_string(level));

			if ( _param.VerboseP ) {
				std::vector<double> norm2_pre_postsmooth=Norm2Vec(r);
				for (int col=0; col < ncol; ++col) {
					if( resid_type == RELATIVE ) {
						MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d col=%d "
								"After Coarse Solve || r ||/|| b ||=%16.8e Target=%16.8e",
								level, iter, col, std::sqrt(norm2_pre_postsmooth[col]/norm2_in[col]), _param.RsdTarget);
					}
					else {
						MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d col=%d "
								"After Coarse Solve || r ||=%16.8e Target=%16.8e",
								level, iter, col, std::sqrt(norm2_pre_postsmooth[col]), _param.RsdTarget);
					}
				}
			}


			Timer::TimerAPI::startTimer("VCycleCoarseEO2/postsmooth/level"+std::to_string(level));
			// delta = zero;
			ZeroVec(delta,subset);
			_post_smoother(delta,r);
			Timer::TimerAPI::stopTimer("VCycleCoarseEO2/postsmooth/level"+std::to_string(level));

			Timer::TimerAPI::startTimer("VCycleCoarseEO2/update/level"+std::to_string(level));
			// Update full solution
			// out += delta;
			YpeqxVec(delta,out,subset);
			_M_fine(tmp,delta,LINOP_OP);
			//r -= tmp;
			YmeqxVec(tmp,r,subset);
			norm2_r = Norm2Vec(r,subset);
			Timer::TimerAPI::stopTimer("VCycleCoarseEO2/update/level"+std::to_string(level));

			if( _param.VerboseP ) {
				for (int col=0; col < ncol; ++col) {
					if( resid_type == RELATIVE ) {
						MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d col=%d "
								"After Post-Smoothing || r ||/|| b ||=%16.8e Target=%16.8e",
								level, iter, col, std::sqrt(norm2_r[col]/norm2_in[col]), _param.RsdTarget);
					}
					else {
						MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d "
								"After Post-Smoothing || r ||=%16.8e Target=%16.8e",
								level, iter, std::sqrt(norm2_r[col]), _param.RsdTarget);
					}
				}
			}

			// Check convergence
			continueP = false;
			for (int col=0; col < ncol; ++col)
				if ( std::sqrt(norm2_r[col]) >= target[col] ) continueP = true;
			if (!continueP) break;
		}

		for (int col=0; col < ncol; ++col)  {
			res[col].resid_type = resid_type;
			res[col].n_count = iter;
			res[col].resid= std::sqrt(norm2_r[col]);
			if( resid_type == RELATIVE ) {
				res[col].resid /= std::sqrt(norm2_in[col]);
			}
		}

		// should remain odd throughout
		//	ZeroVec(out,SUBSET_EVEN); // (keep subset odd)
		Timer::TimerAPI::stopTimer("VCycleCoarseEO2/operator()/level"+std::to_string(level));
		return res;
	}

	VCycleCoarseEO2(const LatticeInfo& coarse_info,
			const std::vector<Block>& my_blocks,
			const std::vector<std::shared_ptr<CoarseSpinor> >& vecs,
			const EOLinearOperator<CoarseSpinor, CoarseGauge>& M_fine,
			const Smoother<CoarseSpinor,CoarseGauge>& pre_smoother,
			const Smoother<CoarseSpinor,CoarseGauge>& post_smoother,
			const LinearSolver<CoarseSpinor,CoarseGauge>& bottom_solver,
			const LinearSolverParamsBase& param, bool apply_clover=true) : _coarse_info(coarse_info),
					_my_blocks(my_blocks),
					_vecs(vecs),
					_M_fine(M_fine),
					_pre_smoother(pre_smoother),
					_post_smoother(post_smoother),
					_bottom_solver(bottom_solver),
					_param(param),
					_Transfer(my_blocks,vecs) {
                                int level = _M_fine.GetLevel();
                                Timer::TimerAPI::addTimer("VCycleCoarseEO2/operator()/level"+std::to_string(level));
                                Timer::TimerAPI::addTimer("VCycleCoarseEO2/restrictFrom/level"+std::to_string(level));
                                Timer::TimerAPI::addTimer("VCycleCoarseEO2/prolongateTo/level"+std::to_string(level));
                                Timer::TimerAPI::addTimer("VCycleCoarseEO2/presmooth/level"+std::to_string(level));
                                Timer::TimerAPI::addTimer("VCycleCoarseEO2/postsmooth/level"+std::to_string(level));
                                Timer::TimerAPI::addTimer("VCycleCoarseEO2/bottom_solve/level"+std::to_string(level));
                                Timer::TimerAPI::addTimer("VCycleCoarseEO2/update/level"+std::to_string(level));
			}


private:
	const LatticeInfo& _coarse_info;
	const std::vector<Block>& _my_blocks;
	const std::vector< std::shared_ptr<CoarseSpinor> >& _vecs;
	const EOLinearOperator< CoarseSpinor, CoarseGauge>& _M_fine;
	const Smoother<CoarseSpinor,CoarseGauge>& _pre_smoother;
	const Smoother<CoarseSpinor,CoarseGauge>& _post_smoother;
	const LinearSolver<CoarseSpinor,CoarseGauge>& _bottom_solver;
	const LinearSolverParamsBase& _param;
	const CoarseTransfer _Transfer;
};


}



#endif /* TEST_QDPXX_VCYCLE_QDPXX_COARSE_H_ */
