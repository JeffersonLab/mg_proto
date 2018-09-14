/*
 * vcycle_qdpxx_coarse.h
 *
 *  Created on: Jan 13, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_COARSE_VCYCLE_COARSE_H_
#define INCLUDE_LATTICE_COARSE_VCYCLE_COARSE_H_

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

#ifdef ENABLE_TIMERS
#include "utils/timer.h"
#endif

namespace MG {

class VCycleCoarse : public LinearSolver<CoarseSpinor,CoarseGauge>
{
public:
	LinearSolverResults operator()(CoarseSpinor& out, const CoarseSpinor& in, ResiduumType resid_type = RELATIVE ) const
	{

		const LatticeInfo& info=out.GetInfo();
		{
			const LatticeInfo& info_in = in.GetInfo();
			const LatticeInfo& M_info = _M_fine.GetInfo();
			AssertCompatible(info, info_in);
			AssertCompatible(info, M_info);
		}

		LinearSolverResults res;

		CoarseSpinor tmp(info);  // Use these to compute residua
		CoarseSpinor r(info);    //

		int level = _M_fine.GetLevel();

		double norm_in, norm_r;


		// Initialize
		ZeroVec(out);  // Work with zero intial guess
		CopyVec(r,in);
		norm_r = sqrt(Norm2Vec(r));
		norm_in = norm_r;

		double target = _param.RsdTarget;
		if ( resid_type == RELATIVE ) {
			target *= norm_r;
		}

		// Check if converged already
		if (  norm_r <= target || _param.MaxIter <= 0  ) {
			res.resid_type = resid_type;
			res.n_count = 0;
			res.resid = norm_r;
			if( resid_type == RELATIVE ) {
				res.resid /= norm_r;
			}
			return res;
		}

		if( _param.VerboseP ) {
		  if( resid_type == RELATIVE) {
        MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d "
            "Initial || r ||/|| b || =%16.8e  Target=%16.8e",
            level, norm_r/norm_in, _param.RsdTarget);

		  }
		  else {
		    MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d "
		        "Initial || r ||=%16.8e  Target=%16.8e",
		        level, norm_r, _param.RsdTarget);
		  }
		}

		// At this point we have to do at least one iteration
		int iter = 0;

		bool continueP = true;
		while ( continueP ) {

			++iter;


			CoarseSpinor delta(info);
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
				double norm_pre_presmooth=sqrt(Norm2Vec(r));
				if( resid_type == RELATIVE ) {
          MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d "
              "After Pre-Smoothing || r ||/|| b ||=%16.8e Target=%16.8e",
              level, iter, norm_pre_presmooth/norm_in, _param.RsdTarget);
				}
				else {
				  MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d "
				      "After Pre-Smoothing || r ||=%16.8e Target=%16.8e",
				      level, iter, norm_pre_presmooth, _param.RsdTarget);
				}
			}

			CoarseSpinor coarse_in(_coarse_info);

			// Coarsen r
			_Transfer.R(r,coarse_in);

			CoarseSpinor coarse_delta(_coarse_info);
			ZeroVec(coarse_delta);
			LinearSolverResults coarse_res =_bottom_solver(coarse_delta,coarse_in);

			// Reuse Smoothed Delta as temporary for prolongating coarse delta back to fine
			_Transfer.P(coarse_delta, delta);

			// Update solution
			//			out += delta;
			YpeqxVec(delta,out);

			// Update residuum
			_M_fine(tmp, delta, LINOP_OP);
			// r -= tmp;
			YmeqxVec(tmp,r);

			if( _param.VerboseP ) {
				double norm_pre_postsmooth = sqrt(Norm2Vec(r));
	      if( resid_type == RELATIVE ) {
	          MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d "
	              "After Coarse Solve || r ||/|| b ||=%16.8e Target=%16.8e",
	              level, iter, norm_pre_postsmooth/norm_in, _param.RsdTarget);
	        }
	        else {
	          MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d "
	              "After Coarse Solve || r ||=%16.8e Target=%16.8e",
	              level, iter, norm_pre_postsmooth, _param.RsdTarget);
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
			norm_r = sqrt(Norm2Vec(r));

			if( _param.VerboseP ) {
        if( resid_type == RELATIVE ) {
            MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d "
                "After Post-Smoothing || r ||/|| b ||=%16.8e Target=%16.8e",
                level, iter, norm_r/norm_in, _param.RsdTarget);
          }
          else {
            MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d "
                "After Post-Smoothing || r ||=%16.8e Target=%16.8e",
                level, iter, norm_r, _param.RsdTarget);
          }
			}

			// Check convergence
			continueP = ( iter < _param.MaxIter ) &&  ( norm_r > target );
		}

		res.resid_type = resid_type;
		res.n_count = iter;
		res.resid= norm_r;
		if( resid_type == RELATIVE ) {
			res.resid /= norm_in;
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
	LinearSolverResults operator()(CoarseSpinor& out, const CoarseSpinor& in, ResiduumType resid_type = RELATIVE ) const
	{

		const LatticeInfo& info=out.GetInfo();
		{
			const LatticeInfo& info_in = in.GetInfo();
			const LatticeInfo& M_info = _M_fine.GetInfo();
			AssertCompatible(info, info_in);
			AssertCompatible(info, M_info);
		}

		LinearSolverResults res;

		CoarseSpinor tmp(info);  // Use these to compute residua
		CoarseSpinor r(info);    //

		int level = _M_fine.GetLevel();
		const CBSubset& subset = _M_fine.GetSubset();

		double norm_in, norm_r;


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
		norm_r = sqrt(Norm2Vec(r,SUBSET_ODD));
		norm_in = norm_r;

		double target = _param.RsdTarget;
		if ( resid_type == RELATIVE ) {
			target *= norm_r;
		}

		// Check if converged already
		if (  norm_r <= target || _param.MaxIter <= 0  ) {
			res.resid_type = resid_type;
			res.n_count = 0;
			res.resid = norm_r;
			if( resid_type == RELATIVE ) {
				res.resid /= norm_r;
			}
			return res;
		}

		if( _param.VerboseP ) {
		  if( resid_type == RELATIVE) {
        MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d "
            "Initial || r ||/|| b || =%16.8e  Target=%16.8e",
            level, norm_r/norm_in, _param.RsdTarget);

		  }
		  else {
		    MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d "
		        "Initial || r ||=%16.8e  Target=%16.8e",
		        level, norm_r, _param.RsdTarget);
		  }
		}

		// At this point we have to do at least one iteration
		int iter = 0;

		bool continueP = true;
		while ( continueP ) {

			++iter;


			CoarseSpinor delta(info);
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
				double norm_pre_presmooth=sqrt(Norm2Vec(r));
				if( resid_type == RELATIVE ) {
          MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d "
              "After Pre-Smoothing || r ||/|| b ||=%16.8e Target=%16.8e",
              level, iter, norm_pre_presmooth/norm_in, _param.RsdTarget);
				}
				else {
				  MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d "
				      "After Pre-Smoothing || r ||=%16.8e Target=%16.8e",
				      level, iter, norm_pre_presmooth, _param.RsdTarget);
				}
			}

			CoarseSpinor coarse_in(_coarse_info);

			// Coarsen r
			_Transfer.R(r,coarse_in);

			CoarseSpinor coarse_delta(_coarse_info);
			ZeroVec(coarse_delta);

			// Again, this is an unprec solve, tho it may be a wrapped even-odd
			LinearSolverResults coarse_res =_bottom_solver(coarse_delta,coarse_in);

			// Reuse Smoothed Delta as temporary for prolongating coarse delta back to fine
			_Transfer.P(coarse_delta, delta);

			// Update solution
			//			out += delta;
			YpeqxVec(delta,out);

			// Update residuum
			_M_fine.unprecOp(tmp, delta, LINOP_OP);
			// r -= tmp;
			YmeqxVec(tmp,r);

			if( _param.VerboseP ) {
				double norm_pre_postsmooth = sqrt(Norm2Vec(r));
	      if( resid_type == RELATIVE ) {
	          MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d "
	              "After Coarse Solve || r ||/|| b ||=%16.8e Target=%16.8e",
	              level, iter, norm_pre_postsmooth/norm_in, _param.RsdTarget);
	        }
	        else {
	          MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d "
	              "After Coarse Solve || r ||=%16.8e Target=%16.8e",
	              level, iter, norm_pre_postsmooth, _param.RsdTarget);
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
			norm_r = sqrt(Norm2Vec(r));

			if( _param.VerboseP ) {
        if( resid_type == RELATIVE ) {
            MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d "
                "After Post-Smoothing || r ||/|| b ||=%16.8e Target=%16.8e",
                level, iter, norm_r/norm_in, _param.RsdTarget);
          }
          else {
            MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d "
                "After Post-Smoothing || r ||=%16.8e Target=%16.8e",
                level, iter, norm_r, _param.RsdTarget);
          }
			}

			// Check convergence
			continueP = ( iter < _param.MaxIter ) &&  ( norm_r > target );
		}

		res.resid_type = resid_type;
		res.n_count = iter;
		res.resid= norm_r;
		if( resid_type == RELATIVE ) {
			res.resid /= norm_in;
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
	LinearSolverResults operator()(CoarseSpinor& out, const CoarseSpinor& in, ResiduumType resid_type = RELATIVE ) const
	{

		const LatticeInfo& info=out.GetInfo();
		{
			const LatticeInfo& info_in = in.GetInfo();
			const LatticeInfo& M_info = _M_fine.GetInfo();
			AssertCompatible(info, info_in);
			AssertCompatible(info, M_info);
		}

		LinearSolverResults res;

		CoarseSpinor tmp(info);  // Use these to compute residua
		CoarseSpinor r(info);    //

		int level = _M_fine.GetLevel();
		const CBSubset& subset = _M_fine.GetSubset();

		double norm_in, norm_r;


		// Initialize
		ZeroVec(out,subset);  // Work with zero intial guess
		ZeroVec(r);

		CopyVec(r,in,subset);


		// We are using a coarse EO2 preconditioner where
		// we smooth with an unwrapped M_fine

		// Typically we will always do this on the full vectors since this is
		// an 'unprec' operation until we exit but here we know r_e = 0 so
		// we can norm only on the subset
		norm_r = sqrt(Norm2Vec(r,SUBSET_ODD));
		norm_in = norm_r;

		double target = _param.RsdTarget;
		if ( resid_type == RELATIVE ) {
			target *= norm_r;
		}

		// Check if converged already
		if (  norm_r <= target || _param.MaxIter <= 0  ) {
			res.resid_type = resid_type;
			res.n_count = 0;
			res.resid = norm_r;
			if( resid_type == RELATIVE ) {
				res.resid /= norm_r;
			}
			return res;
		}

		if( _param.VerboseP ) {
			if( resid_type == RELATIVE) {
				MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d "
						"Initial || r ||/|| b || =%16.8e  Target=%16.8e",
						level, norm_r/norm_in, _param.RsdTarget);

			}
			else {
				MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d "
						"Initial || r ||=%16.8e  Target=%16.8e",
						level, norm_r, _param.RsdTarget);
			}
		}

		// At this point we have to do at least one iteration
		int iter = 0;

		bool continueP = true;
		while ( continueP ) {

			++iter;


			CoarseSpinor delta(info);
			ZeroVec(delta,subset);

			// Smoother does not compute a residuum
			// It is an 'unprec' smoother tho it may use internally a wrapped even-odd
			_pre_smoother(delta,r);

			// Update solution
			// out += delta;
			YpeqxVec(delta,out,subset);

			// Update prec residuum
			_M_fine(tmp,delta, LINOP_OP);

			// r -= tmp;
			YmeqxVec(tmp,r,subset);

			if ( _param.VerboseP ) {
				double norm_pre_presmooth=sqrt(Norm2Vec(r,subset));
				if( resid_type == RELATIVE ) {
					MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d "
							"After Pre-Smoothing || r ||/|| b ||=%16.8e Target=%16.8e",
							level, iter, norm_pre_presmooth/norm_in, _param.RsdTarget);
				}
				else {
					MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d "
							"After Pre-Smoothing || r ||=%16.8e Target=%16.8e",
							level, iter, norm_pre_presmooth, _param.RsdTarget);
				}
			}

			CoarseSpinor coarse_in(_coarse_info);
			_Transfer.R(r,ODD,coarse_in);

			CoarseSpinor coarse_delta(_coarse_info);
			ZeroVec(coarse_delta);

			// Again, this is an unprec solve, tho it may be a wrapped even-odd
			LinearSolverResults coarse_res =_bottom_solver(coarse_delta,coarse_in);

			// Reuse Smoothed Delta as temporary for prolongating coarse delta back to fine
			_Transfer.P(coarse_delta, ODD, delta);

			// Update solution
			//			out += delta;
			YpeqxVec(delta,out, subset);

			// Update residuum
			_M_fine(tmp, delta, LINOP_OP);
			// r -= tmp;
			YmeqxVec(tmp,r,subset);

			if( _param.VerboseP ) {
				double norm_pre_postsmooth = sqrt(Norm2Vec(r,subset));
				if( resid_type == RELATIVE ) {
					MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d "
							"After Coarse Solve || r ||/|| b ||=%16.8e Target=%16.8e",
							level, iter, norm_pre_postsmooth/norm_in, _param.RsdTarget);
				}
				else {
					MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d "
							"After Coarse Solve || r ||=%16.8e Target=%16.8e",
							level, iter, norm_pre_postsmooth, _param.RsdTarget);
				}
			}

			// delta = zero;
			ZeroVec(delta,subset);
			_post_smoother(delta,r);

			// Update full solution
			// out += delta;
			YpeqxVec(delta,out,subset);
			_M_fine(tmp,delta,LINOP_OP);
			//r -= tmp;
			YmeqxVec(tmp,r,subset);
			norm_r = sqrt(Norm2Vec(r,subset));

			if( _param.VerboseP ) {
				if( resid_type == RELATIVE ) {
					MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d "
							"After Post-Smoothing || r ||/|| b ||=%16.8e Target=%16.8e",
							level, iter, norm_r/norm_in, _param.RsdTarget);
				}
				else {
					MasterLog(INFO, "VCYCLE (COARSE->COARSE): level=%d iter=%d "
							"After Post-Smoothing || r ||=%16.8e Target=%16.8e",
							level, iter, norm_r, _param.RsdTarget);
				}
			}

			// Check convergence
			continueP = ( iter < _param.MaxIter ) &&  ( norm_r > target );
		}

		res.resid_type = resid_type;
		res.n_count = iter;
		res.resid= norm_r;
		if( resid_type == RELATIVE ) {
			res.resid /= norm_in;
		}
		// should remain odd throughout
		//	ZeroVec(out,SUBSET_EVEN); // (keep subset odd)
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
					_Transfer(my_blocks,vecs) {}


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
