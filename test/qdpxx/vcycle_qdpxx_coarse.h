/*
 * vcycle_qdpxx_coarse.h
 *
 *  Created on: Jan 13, 2017
 *      Author: bjoo
 */

#ifndef TEST_QDPXX_VCYCLE_QDPXX_COARSE_H_
#define TEST_QDPXX_VCYCLE_QDPXX_COARSE_H_
#include "qdp.h"
#include "lattice/constants.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"
#include "invmr.h"
#include "lattice/invfgmres_coarse.h"
#include "lattice/fgmres_common.h"
#include "aggregate_block_qdpxx.h"
#include "utils/print_utils.h"
using namespace QDP;
using namespace MG;

namespace MGTesting {

class VCycleQDPCoarse2 : public LinearSolver<LatticeFermion, multi1d<LatticeColorMatrix> >
{
public:
	LinearSolverResults operator()(LatticeFermion& out, const LatticeFermion& in, ResiduumType resid_type = RELATIVE ) const
	{

		LinearSolverResults res;

		LatticeFermion tmp;  // Use these to compute residua
		LatticeFermion r;    //

		int level = _M_fine.GetLevel();

		Double norm_in, norm_r;


		// Initialize
		out = zero;  // Work with zero intial guess
		r = in;
		norm_r = sqrt(norm2(r));
		norm_in = norm_r;

		Double target = _param.RsdTarget;
		if ( resid_type == RELATIVE ) {
			target *= norm_r;
		}

		// Check if converged already
		if ( toBool ( norm_r <= target ) || _param.MaxIter <= 0  ) {
			res.resid_type = resid_type;
			res.n_count = 0;
			res.resid = toDouble(norm_r);
			if( resid_type == RELATIVE ) {
				res.resid /= toDouble(norm_r);
			}
			return res;
		}

		if( _param.VerboseP ) {
			MasterLog(INFO, "VCYCLE (QDP->COARSE): level=%d Initial || r ||=%16.8e  Target=%16.8e", level, toDouble(norm_r), toDouble(target));
		}
		// At this point we have to do at least one iteration
		int iter = 0;

		bool continueP = true;
		while ( continueP ) {

			++iter;


			LatticeFermion delta = zero;

			// Smoother does not compute a residuum
			_pre_smoother(delta,r);

			// Update solution
			out += delta;

			// Update residuum
			_M_fine(tmp,delta, LINOP_OP);
			r -= tmp;

			if ( _param.VerboseP ) {
				Double norm_pre_presmooth=sqrt(norm2(r));
				MasterLog(INFO, "VCYCLE (QDP->COARSE): level=%d iter=%d After Presmoothing || r ||=%16.8e", level, iter, toDouble(norm_pre_presmooth));
			}

			CoarseSpinor coarse_in(_coarse_info);

			// Coarsen r
			restrictSpinorQDPXXFineToCoarse(_my_blocks, _vecs, r,coarse_in);

			CoarseSpinor coarse_delta(_coarse_info);
			ZeroVec(coarse_delta);
			LinearSolverResults coarse_res =_bottom_solver(coarse_delta,coarse_in);

			// Reuse Smoothed Delta as temporary for prolongating coarse delta back to fine
			prolongateSpinorCoarseToQDPXXFine(_my_blocks, _vecs, coarse_delta, delta);

			// Update solution
			out += delta;

			// Update residuum
			_M_fine(tmp, delta, LINOP_OP);
			r -= tmp;

			if( _param.VerboseP ) {
				Double norm_pre_postsmooth = sqrt(norm2(r));
				MasterLog(INFO, "VCYCLE (QDP->COARSE): level=%d iter=%d Before Post-smoothing || r ||=%16.8e", level, iter, toDouble(norm_pre_postsmooth));
			}

			delta = zero;
			_post_smoother(delta,r);

			// Update full solution
			out += delta;
			_M_fine(tmp,delta,LINOP_OP);
			r -= tmp;
			norm_r = sqrt(norm2(r));

			if( _param.VerboseP ) {
				MasterLog(INFO, "VCYCLE (QDP->COARSE): level=%d iter=%d || r ||=%16.8e target=%16.8e", level, iter, toDouble(norm_r), toDouble(target));
			}

			// Check convergence
			continueP = ( iter < _param.MaxIter ) &&  toBool( norm_r > target );
		}

		res.resid_type = resid_type;
		res.n_count = iter;
		res.resid=toDouble(norm_r);
		if( resid_type == RELATIVE ) {
			res.resid /= toDouble(norm_in);
		}
		return res;
	}

	VCycleQDPCoarse2(const LatticeInfo& coarse_info,
					 const std::vector<Block>& my_blocks,
					 const multi1d<LatticeFermion>& vecs,
					 const LinearOperator<LatticeFermion, multi1d<LatticeColorMatrix>>& M_fine,
					 const MRSmoother& pre_smoother,
					 const MRSmoother& post_smoother,
					 const FGMRESSolverCoarse& bottom_solver,
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
	const multi1d<LatticeFermion>& _vecs;
	const LinearOperator<LatticeFermion, multi1d<LatticeColorMatrix>>& _M_fine;
	const MRSmoother& _pre_smoother;
	const MRSmoother& _post_smoother;
	const FGMRESSolverCoarse& _bottom_solver;
	const LinearSolverParamsBase& _param;
};


}



#endif /* TEST_QDPXX_VCYCLE_QDPXX_COARSE_H_ */
