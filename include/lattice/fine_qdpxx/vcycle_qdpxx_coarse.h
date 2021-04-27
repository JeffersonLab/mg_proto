/*
 * vcycle_qdpxx_coarse.h
 *
 *  Created on: Jan 13, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_FINE_QDPXX_VCYCLE_QDPXX_COARSE_H_
#define INCLUDE_LATTICE_FINE_QDPXX_VCYCLE_QDPXX_COARSE_H_
#include "lattice/coarse/coarse_types.h"
#include "lattice/constants.h"
#include "lattice/fgmres_common.h"
#include "lattice/fine_qdpxx/aggregate_block_qdpxx.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"
#include "qdp.h"
#include "utils/print_utils.h"
#include <lattice/coarse/invfgmres_coarse.h>
#include <lattice/fine_qdpxx/invmr_qdpxx.h>
using namespace QDP;
using namespace MG;

namespace MG {

    class VCycleQDPCoarse2 : public LinearSolver<LatticeFermion> {
    public:
        std::vector<LinearSolverResults>
        operator()(LatticeFermion &out, const LatticeFermion &in,
                   ResiduumType resid_type = RELATIVE,
                   InitialGuess guess = InitialGuessNotGiven) const {
            (void)guess;

            LinearSolverResults res;

            LatticeFermion tmp; // Use these to compute residua
            LatticeFermion r;   //

            int level = _M_fine.GetLevel();

            Double norm_in, norm_r;

            // Initialize
            out = zero; // Work with zero intial guess
            r = in;
            norm_r = sqrt(norm2(r));
            norm_in = norm_r;

            Double target = _param.RsdTarget;
            if (resid_type == RELATIVE) { target *= norm_r; }

            // Check if converged already
            if (toBool(norm_r <= target) || _param.MaxIter <= 0) {
                res.resid_type = resid_type;
                res.n_count = 0;
                res.resid = toDouble(norm_r);
                if (resid_type == RELATIVE) { res.resid /= toDouble(norm_r); }
                return std::vector<LinearSolverResults>(1, res);
            }

            if (_param.VerboseP) {
                if (resid_type == RELATIVE) {
                    MasterLog(INFO,
                              "VCYCLE (QDP->COARSE): level=%d Initial || r ||/|| b ||=%16.8e  "
                              "Target=%16.8e",
                              level, toDouble(norm_r / norm_in), _param.RsdTarget);
                } else {
                    MasterLog(
                        INFO,
                        "VCYCLE (QDP->COARSE): level=%d Initial || r ||=%16.8e  Target=%16.8e",
                        level, toDouble(norm_r), _param.RsdTarget);
                }
            }

            // At this point we have to do at least one iteration
            int iter = 0;

            bool continueP = true;
            while (continueP) {

                ++iter;

                LatticeFermion delta = zero;

                // Smoother does not compute a residuum
                _pre_smoother(delta, r);

                // Update solution
                out += delta;

                // Update residuum
                _M_fine(tmp, delta, LINOP_OP);
                r -= tmp;

                if (_param.VerboseP) {

                    Double norm_pre_presmooth = sqrt(norm2(r));
                    if (resid_type == RELATIVE) {
                        MasterLog(INFO,
                                  "VCYCLE (QDP->COARSE): level=%d iter=%d "
                                  "After Pre-Smoothing || r ||/ || b ||=%16.8e Target=%16.8e",
                                  level, iter, toDouble(norm_pre_presmooth / norm_in),
                                  _param.RsdTarget);

                    } else {
                        MasterLog(INFO,
                                  "VCYCLE (QDP->COARSE): level=%d iter=%d "
                                  "After Pre-Smoothing || r ||=%16.8e Target=%16.8e",
                                  level, iter, toDouble(norm_pre_presmooth), _param.RsdTarget);
                    }
                }

                CoarseSpinor coarse_in(_coarse_info);

                // Coarsen r
                restrictSpinorQDPXXFineToCoarse(_my_blocks, _vecs, r, coarse_in);

                CoarseSpinor coarse_delta(_coarse_info);
                ZeroVec(coarse_delta);
                _bottom_solver(coarse_delta, coarse_in);

                // Reuse Smoothed Delta as temporary for prolongating coarse delta back to fine
                prolongateSpinorCoarseToQDPXXFine(_my_blocks, _vecs, coarse_delta, delta);

                // Update solution
                out += delta;

                // Update residuum
                _M_fine(tmp, delta, LINOP_OP);
                r -= tmp;

                if (_param.VerboseP) {
                    Double norm_pre_postsmooth = sqrt(norm2(r));
                    if (resid_type == RELATIVE) {
                        MasterLog(INFO,
                                  "VCYCLE (QDP->COARSE): level=%d iter=%d "
                                  "After Coarse Solve || r ||/|| b ||=%16.8e Target=%16.8e",
                                  level, iter, toDouble(norm_pre_postsmooth / norm_in),
                                  _param.RsdTarget);
                    } else {
                        MasterLog(INFO,
                                  "VCYCLE (QDP->COARSE): level=%d iter=%d "
                                  "After Coarse Solve || r ||=%16.8e Target=%16.8e",
                                  level, iter, toDouble(norm_pre_postsmooth), _param.RsdTarget);
                    }
                }

                delta = zero;
                _post_smoother(delta, r);

                // Update full solution
                out += delta;
                _M_fine(tmp, delta, LINOP_OP);
                r -= tmp;
                norm_r = sqrt(norm2(r));

                if (_param.VerboseP) {
                    if (resid_type == RELATIVE) {
                        MasterLog(INFO,
                                  "VCYCLE (QDP->COARSE): level=%d iter=%d "
                                  "After Post-Smoothing || r ||/|| b ||=%16.8e Target=%16.8e",
                                  level, iter, toDouble(norm_r / norm_in), _param.RsdTarget);

                    } else {
                        MasterLog(INFO,
                                  "VCYCLE (QDP->COARSE): level=%d iter=%d "
                                  "After Post-Smoothing || r ||=%16.8e Target=%16.8e",
                                  level, iter, toDouble(norm_r), _param.RsdTarget);
                    }
                }

                // Check convergence
                continueP = (iter < _param.MaxIter) && toBool(norm_r > target);
            }

            res.resid_type = resid_type;
            res.n_count = iter;
            res.resid = toDouble(norm_r);
            if (resid_type == RELATIVE) { res.resid /= toDouble(norm_in); }
            return std::vector<LinearSolverResults>(1, res);
        }

        VCycleQDPCoarse2(const LatticeInfo &coarse_info, const std::vector<Block> &my_blocks,
                         const multi1d<LatticeFermion> &vecs,
                         const LinearOperator<LatticeFermion> &M_fine,
                         const LinearSolver<LatticeFermion> &pre_smoother,
                         const LinearSolver<LatticeFermion> &post_smoother,
                         const LinearSolver<CoarseSpinor> &bottom_solver,
                         const LinearSolverParamsBase &param)
            : LinearSolver<LatticeFermion>(M_fine, param),
              _coarse_info(coarse_info),
              _my_blocks(my_blocks),
              _vecs(vecs),
              _M_fine(M_fine),
              _pre_smoother(pre_smoother),
              _post_smoother(post_smoother),
              _bottom_solver(bottom_solver),
              _param(param) {}

        const LatticeInfo &GetInfo() const { return _M_fine.GetInfo(); }
        const CBSubset &GetSubset() const { return _M_fine.GetSubset(); }

    private:
        const LatticeInfo &_coarse_info;
        const std::vector<Block> &_my_blocks;
        const multi1d<LatticeFermion> &_vecs;
        const LinearOperator<LatticeFermion> &_M_fine;
        const LinearSolver<LatticeFermion> &_pre_smoother;
        const LinearSolver<LatticeFermion> &_post_smoother;
        const LinearSolver<CoarseSpinor> &_bottom_solver;
        const LinearSolverParamsBase &_param;
    };
}

#endif /* TEST_QDPXX_VCYCLE_QDPXX_COARSE_H_ */
