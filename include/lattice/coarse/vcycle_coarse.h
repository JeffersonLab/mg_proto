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
#include "utils/print_utils.h"

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
			restrictSpinor(_my_blocks, _vecs, r,coarse_in);

			CoarseSpinor coarse_delta(_coarse_info);
			ZeroVec(coarse_delta);
			LinearSolverResults coarse_res =_bottom_solver(coarse_delta,coarse_in);

			// Reuse Smoothed Delta as temporary for prolongating coarse delta back to fine
			prolongateSpinor(_my_blocks, _vecs, coarse_delta, delta);

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
															_param(param) {}


private:
	const LatticeInfo& _coarse_info;
	const std::vector<Block>& _my_blocks;
	const std::vector< std::shared_ptr<CoarseSpinor> >& _vecs;
	const LinearOperator< CoarseSpinor, CoarseGauge>& _M_fine;
	const Smoother<CoarseSpinor,CoarseGauge>& _pre_smoother;
	const Smoother<CoarseSpinor,CoarseGauge>& _post_smoother;
	const LinearSolver<CoarseSpinor,CoarseGauge>& _bottom_solver;
	const LinearSolverParamsBase& _param;
};


}



#endif /* TEST_QDPXX_VCYCLE_QDPXX_COARSE_H_ */
