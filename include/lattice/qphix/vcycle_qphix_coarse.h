/*
 * vcycle_qphix_coarse.h
 *
 *  Created on: Oct 27, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_QPHIX_VCYCLE_QPHIX_COARSE_H_
#define INCLUDE_LATTICE_QPHIX_VCYCLE_QPHIX_COARSE_H_

#include "MG_config.h"
#include <algorithm>
#include <lattice/solver.h>
#include <lattice/qphix/qphix_types.h>
#include <lattice/qphix/qphix_blas_wrappers.h>
#include <lattice/coarse/coarse_types.h>
#include <lattice/qphix/qphix_aggregate.h>
#include <lattice/qphix/qphix_transfer.h>
#include "utils/auxiliary.h"
#include "utils/timer.h"

#ifdef MG_ENABLE_TIMERS
#include "utils/timer.h"
#endif

namespace MG
{

class VCycleQPhiXCoarse2 :
  public LinearSolver<QPhiXSpinor, QPhiXGauge >,
  public AuxiliarySpinors<QPhiXSpinorF>
{
    using AuxQF = AuxiliarySpinors<QPhiXSpinorF>;
  public:
    std::vector<LinearSolverResults> operator()(QPhiXSpinor& out,
        const QPhiXSpinor& in, ResiduumType resid_type = RELATIVE ) const
    {
      Timer::TimerAPI::startTimer("VCycleQPhiXCoarse2/()/level0");

      assert(out.GetNCol() == in.GetNCol());
      IndexType ncol = out.GetNCol();

      std::vector<LinearSolverResults> res(ncol);

      Timer::TimerAPI::startTimer("VCycleQPhiXCoarse2/convert()/level0");
      std::shared_ptr<QPhiXSpinorF> r = AuxQF::tmp(_fine_info, ncol);
      ConvertSpinor(in,*r);
      Timer::TimerAPI::stopTimer("VCycleQPhiXCoarse2/convert()/level0");

      // May want to do these in double later?
      // But this is just a preconditioner.
      // So try SP for now
      Timer::TimerAPI::startTimer("VCycleQPhiXCoarse2/norm()/level0");
      std::shared_ptr<QPhiXSpinorF> out_f = AuxQF::tmp(_fine_info, ncol);
      Timer::TimerAPI::stopTimer("VCycleQPhiXCoarse2/norm()/level0");

      int level = _M_fine.GetLevel();

      std::vector<double> norm_in(ncol),  norm_r(ncol);
      ZeroVec(*out_f);   // out_f = 0

      norm_r = aux::sqrt(Norm2Vec(*r));
      norm_in = norm_r;

      std::vector<double> target(ncol, _param.RsdTarget);
      bool continueP = false;
      for (int col=0; col < ncol; ++col) {
        if ( resid_type == RELATIVE ) {
          target[col] *= norm_r[col];
        }
        if ( norm_r[col] > target[col]) continueP = true;
      }

      // Check if converged already
      if ( !continueP || _param.MaxIter <= 0  ) {
        for (int col=0; col < ncol; ++col) {
          res[col].resid_type = resid_type;
          res[col].n_count = 0;
          res[col].resid = norm_r[col];
          if( resid_type == RELATIVE ) {
            res[col].resid /= norm_r[col];
          }
        }
        return res;
      }

      if( _param.VerboseP ) {
        for (int col=0; col < ncol; ++col) {
          if( resid_type == RELATIVE) {
            MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d col=%d"
                "Initial || r ||/|| b || =%16.8e  Target=%16.8e",
                level, col, norm_r[col]/norm_in[col], _param.RsdTarget);

          }
          else {
            MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d col=%d"
                "Initial || r ||=%16.8e  Target=%16.8e",
                level, col, norm_r[col], _param.RsdTarget);
          }
        }
      }

      // At this point we have to do at least one iteration
      int iter = 0;

      Timer::TimerAPI::startTimer("VCycleQPhiXCoarse2/norm()/level0");
      std::shared_ptr<QPhiXSpinorF> delta = AuxQF::tmp(_fine_info, ncol);
      std::shared_ptr<QPhiXSpinorF> t = AuxQF::tmp(_fine_info, ncol);
      CoarseSpinor coarse_in(_coarse_info, ncol);
      CoarseSpinor coarse_delta(_coarse_info, ncol);
      Timer::TimerAPI::stopTimer("VCycleQPhiXCoarse2/norm()/level0");

      while ( iter < _param.MaxIter) {
        ++iter;

        ZeroVec(*delta);

        // Smoother does not compute a residuum
        Timer::TimerAPI::startTimer("VCycleQPhiXCoarse2/pre_smoother()/level0");
        _pre_smoother(*delta,*r);
        Timer::TimerAPI::stopTimer("VCycleQPhiXCoarse2/pre_smoother()/level0");

        // Update solution

        YpeqXVec(*delta,*out_f);
        // Update residuum
        Timer::TimerAPI::startTimer("VCycleQPhiXCoarse2/M_fine()/level0");
        _M_fine(*t,*delta, LINOP_OP);
        Timer::TimerAPI::stopTimer("VCycleQPhiXCoarse2/M_fine()/level0");
        YmeqXVec(*t,*r);

        if ( _param.VerboseP ) {
          std::vector<double> norm_pre_presmooth=aux::sqrt(Norm2Vec(*r));
          for (int col=0; col < ncol; ++col) {
            if( resid_type == RELATIVE ) {
              MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d iter=%d col=%d "
                  "After Pre-Smoothing || r ||/||b||=%16.8e Target=%16.8e",
                  level, col, iter, norm_pre_presmooth[col]/norm_in[col], _param.RsdTarget);
            }
            else {
              MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d iter=%d col=%d "
                  "After Pre-Smoothing || r ||=%16.8e Target=%16.8e",
                  level, col, iter, norm_pre_presmooth[col], _param.RsdTarget);
            }
          }
        }


        // Coarsen r
        Timer::TimerAPI::startTimer("VCycleQPhiXCoarse2/R()/level0");
        _Transfer.R(*r,coarse_in);
        Timer::TimerAPI::stopTimer("VCycleQPhiXCoarse2/R()/level0");

        ZeroVec(coarse_delta);
        Timer::TimerAPI::startTimer("VCycleQPhiXCoarse2/bottom_solver()/level0");
        _bottom_solver(coarse_delta,coarse_in);
        Timer::TimerAPI::stopTimer("VCycleQPhiXCoarse2/bottom_solver()/level0");

        // Reuse Smoothed Delta as temporary for prolongating coarse delta back to fine
        Timer::TimerAPI::startTimer("VCycleQPhiXCoarse2/P()/level0");
        _Transfer.P(coarse_delta, *delta);
        Timer::TimerAPI::stopTimer("VCycleQPhiXCoarse2/P()/level0");

        // Update solution
        YpeqXVec(*delta,*out_f);

        // Update residuum
        Timer::TimerAPI::startTimer("VCycleQPhiXCoarse2/M_fine()/level0");
        _M_fine(*t, *delta, LINOP_OP);
        Timer::TimerAPI::stopTimer("VCycleQPhiXCoarse2/M_fine()/level0");
        YmeqXVec(*t,*r);


        if ( _param.VerboseP ) {
          std::vector<double> norm_pre_postsmooth=aux::sqrt(Norm2Vec(*r));
          for (int col=0; col < ncol; ++col) {
            if( resid_type == RELATIVE ) {
              MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d iter=%d col=%d "
                  "After Coarse Solve || r ||/||b||=%16.8e Target=%16.8e",
                  level, col, iter, norm_pre_postsmooth[col]/norm_in[col], _param.RsdTarget);
            }
            else {
              MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d iter=%d col=%d "
                  "After Coarse Solve || r ||=%16.8e Target=%16.8e",
                  level, col, iter, norm_pre_postsmooth[col], _param.RsdTarget);
            }
          }
        }

        ZeroVec(*delta);
        Timer::TimerAPI::startTimer("VCycleQPhiXCoarse2/post_smoother()/level0");
        _post_smoother(*delta,*r);
        Timer::TimerAPI::stopTimer("VCycleQPhiXCoarse2/post_smoother()/level0");

        // Update full solution
        YpeqXVec(*delta,*out_f);

        Timer::TimerAPI::startTimer("VCycleQPhiXCoarse2/M_fine()/level0");
        _M_fine(*t,*delta,LINOP_OP);
        Timer::TimerAPI::stopTimer("VCycleQPhiXCoarse2/M_fine()/level0");
        norm_r = aux::sqrt(XmyNorm2Vec(*r,*t));


        if ( _param.VerboseP ) {
          for (int col=0; col < ncol; ++col) {
            if( resid_type == RELATIVE ) {
              MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d iter=%d col=%d "
                  "After Post-Smoothing || r ||/||b||=%16.8e Target=%16.8e",
                  level, col, iter, norm_r[col]/norm_in[col], _param.RsdTarget);
            }
            else {
              MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d iter=%d col=%d "
                  "After Post-Smoothing || r ||=%16.8e Target=%16.8e",
                  level, col, iter, norm_r[col], _param.RsdTarget);
            }
          }
        }

        // Check convergence
        continueP = false;
        for (int col=0; col < ncol; ++col)
          if ( norm_r[col] >= target[col] ) continueP = true;
        if (!continueP) break;
      }

      // Convert Back Up to DP
      Timer::TimerAPI::startTimer("VCycleQPhiXCoarse2/convert()/level0");
      ConvertSpinor(*out_f,out);
      Timer::TimerAPI::stopTimer("VCycleQPhiXCoarse2/convert()/level0");
      for (int col=0; col < ncol; ++col)  {
        res[col].resid_type = resid_type;
        res[col].n_count = iter;
        res[col].resid= norm_r[col];
        if( resid_type == RELATIVE ) {
          res[col].resid /= norm_in[col];
        }
      }
      Timer::TimerAPI::stopTimer("VCycleQPhiXCoarse2/()/level0");
      return res;
    }

    VCycleQPhiXCoarse2(const LatticeInfo& fine_info,
        const LatticeInfo& coarse_info,
        const std::vector<Block>& my_blocks,
        const std::vector<std::shared_ptr<QPhiXSpinorF>>& vecs,
        LinearOperator<QPhiXSpinorF,QPhiXGaugeF>& M_fine,
        const Smoother<QPhiXSpinorF,QPhiXGaugeF>& pre_smoother,
        const Smoother<QPhiXSpinorF,QPhiXGaugeF>& post_smoother,
        const LinearSolver<CoarseSpinor,CoarseGauge>& bottom_solver,
        const LinearSolverParamsBase& param)
      : _fine_info(fine_info), _coarse_info(coarse_info),
      _my_blocks(my_blocks),
      _vecs(vecs),
      _M_fine(M_fine),
      _pre_smoother(pre_smoother),
      _post_smoother(post_smoother),
      _bottom_solver(bottom_solver),
      _param(param),
      _Transfer(my_blocks,vecs) {

          Timer::TimerAPI::addTimer("VCycleQPhiXCoarse2/M_fine()/level0");
          Timer::TimerAPI::addTimer("VCycleQPhiXCoarse2/pre_smoother()/level0");
          Timer::TimerAPI::addTimer("VCycleQPhiXCoarse2/post_smoother()/level0");
          Timer::TimerAPI::addTimer("VCycleQPhiXCoarse2/bottom_solver()/level0");
          Timer::TimerAPI::addTimer("VCycleQPhiXCoarse2/R()/level0");
          Timer::TimerAPI::addTimer("VCycleQPhiXCoarse2/P()/level0");

    }

    const LatticeInfo& GetInfo() const { return _M_fine.GetInfo(); }
    const CBSubset& GetSubset() const { return _M_fine.GetSubset(); }

  private:
    const LatticeInfo& _fine_info;
    const LatticeInfo& _coarse_info;
    const std::vector<Block>& _my_blocks;
    const std::vector<std::shared_ptr<QPhiXSpinorF>>& _vecs;
    LinearOperator<QPhiXSpinorF, QPhiXGaugeF>& _M_fine;
    const Smoother<QPhiXSpinorF,QPhiXGaugeF>& _pre_smoother;
    const Smoother<QPhiXSpinorF,QPhiXGaugeF>& _post_smoother;
    const LinearSolver<CoarseSpinor,CoarseGauge>& _bottom_solver;
    const LinearSolverParamsBase& _param;
    const QPhiXTransfer<QPhiXSpinorF> _Transfer;

};


// This is essentially the same as a regular VCycle.
// The only main difference is that the operator used is now
// an even odd operator, so we need to call its unprecOp operator.
//

class VCycleQPhiXCoarseEO2 :
  public LinearSolver<QPhiXSpinor, QPhiXGauge >
{
  public:
    std::vector<LinearSolverResults> operator()(QPhiXSpinor& out,
        const QPhiXSpinor& in, ResiduumType resid_type = RELATIVE ) const
    {
      assert(out.GetNCol() == in.GetNCol());
      IndexType ncol = out.GetNCol();

      std::vector<LinearSolverResults> res(ncol);

      QPhiXSpinorF in_f(_fine_info, ncol);
      ZeroVec(in_f, SUBSET_ALL);

      ConvertSpinor(in,in_f,_M_fine.GetSubset());

      // May want to do these in double later?
      // But this is just a preconditioner.
      // So try SP for now
      QPhiXSpinorF r(_fine_info, ncol);
      QPhiXSpinorF out_f(_fine_info, ncol);

      int level = _M_fine.GetLevel();

      std::vector<double> norm_in(ncol),  norm_r(ncol);
      ZeroVec(out_f);   // out_f = 0
      CopyVec(r,in_f);  //  r  <- in_f

      norm_r = aux::sqrt(Norm2Vec(r));
      norm_in = norm_r;

      std::vector<double> target(ncol, _param.RsdTarget);
      bool continueP = false;
      for (int col=0; col < ncol; ++col) {
        if ( resid_type == RELATIVE ) {
          target[col] *= norm_r[col];
        }
        if ( norm_r[col] > target[col]) continueP = true;
      }

      // Check if converged already
      if ( !continueP || _param.MaxIter <= 0  ) {
        for (int col=0; col < ncol; ++col) {
          res[col].resid_type = resid_type;
          res[col].n_count = 0;
          res[col].resid = norm_r[col];
          if( resid_type == RELATIVE ) {
            res[col].resid /= norm_r[col];
          }
        }
        return res;
      }

      if( _param.VerboseP ) {
        for (int col=0; col < ncol; ++col) {
          if( resid_type == RELATIVE) {
            MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d col=%d"
                "Initial || r ||/|| b || =%16.8e  Target=%16.8e",
                level, col, norm_r[col]/norm_in[col], _param.RsdTarget);

          }
          else {
            MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d col=%d"
                "Initial || r ||=%16.8e  Target=%16.8e",
                level, col, norm_r[col], _param.RsdTarget);
          }
        }
      }

      // At this point we have to do at least one iteration
      int iter = 0;

      QPhiXSpinorF delta(_fine_info, ncol);
      QPhiXSpinorF tmp(_fine_info, ncol);
      CoarseSpinor coarse_in(_coarse_info, ncol);
      CoarseSpinor coarse_delta(_coarse_info, ncol);

      while ( iter < _param.MaxIter) {
        ++iter;

        ZeroVec(delta);

        // Smoother does not compute a residuum
        _pre_smoother(delta,r);

        // Update solution

        YpeqXVec(delta,out_f);
        // Update residuum
        _M_fine.unprecOp(tmp,delta, LINOP_OP);
        YmeqXVec(tmp,r);

        if ( _param.VerboseP ) {
          std::vector<double> norm_pre_presmooth=aux::sqrt(Norm2Vec(r));
          for (int col=0; col < ncol; ++col) {
            if( resid_type == RELATIVE ) {
              MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d iter=%d col=%d "
                  "After Pre-Smoothing || r ||/||b||=%16.8e Target=%16.8e",
                  level, col, iter, norm_pre_presmooth[col]/norm_in[col], _param.RsdTarget);
            }
            else {
              MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d iter=%d col=%d "
                  "After Pre-Smoothing || r ||=%16.8e Target=%16.8e",
                  level, col, iter, norm_pre_presmooth[col], _param.RsdTarget);
            }
          }
        }


        // Coarsen r
        _Transfer.R(r,coarse_in);

        ZeroVec(coarse_delta);
        _bottom_solver(coarse_delta,coarse_in);

        // Reuse Smoothed Delta as temporary for prolongating coarse delta back to fine
        _Transfer.P(coarse_delta, delta);

        // Update solution
        YpeqXVec(delta,out_f);

        // Update residuum
        _M_fine.unprecOp(tmp, delta, LINOP_OP);
        YmeqXVec(tmp,r);


        if ( _param.VerboseP ) {
          std::vector<double> norm_pre_postsmooth=aux::sqrt(Norm2Vec(r));
          for (int col=0; col < ncol; ++col) {
            if( resid_type == RELATIVE ) {
              MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d iter=%d col=%d "
                  "After Coarse Solve || r ||/||b||=%16.8e Target=%16.8e",
                  level, col, iter, norm_pre_postsmooth[col]/norm_in[col], _param.RsdTarget);
            }
            else {
              MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d iter=%d col=%d "
                  "After Coarse Solve || r ||=%16.8e Target=%16.8e",
                  level, col, iter, norm_pre_postsmooth[col], _param.RsdTarget);
            }
          }
        }

        ZeroVec(delta);
        _post_smoother(delta,r);

        // Update full solution
        YpeqXVec(delta,out_f);

        _M_fine.unprecOp(tmp,delta,LINOP_OP);
        norm_r = aux::sqrt(XmyNorm2Vec(r,tmp));


        if ( _param.VerboseP ) {
          for (int col=0; col < ncol; ++col) {
            if( resid_type == RELATIVE ) {
              MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d iter=%d col=%d "
                  "After Post-Smoothing || r ||/||b||=%16.8e Target=%16.8e",
                  level, col, iter, norm_r[col]/norm_in[col], _param.RsdTarget);
            }
            else {
              MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d iter=%d col=%d "
                  "After Post-Smoothing || r ||=%16.8e Target=%16.8e",
                  level, col, iter, norm_r[col], _param.RsdTarget);
            }
          }
        }

        // Check convergence
        continueP = false;
        for (int col=0; col < ncol; ++col)
          if ( norm_r[col] >= target[col] ) continueP = true;
        if (!continueP) break;
      }

      // Convert Back Up to DP only on the output subset.
      ZeroVec(out,SUBSET_ALL);
      ConvertSpinor(out_f,out, _M_fine.GetSubset());
      for (int col=0; col < ncol; ++col)  {
        res[col].resid_type = resid_type;
        res[col].n_count = iter;
        res[col].resid= norm_r[col];
        if( resid_type == RELATIVE ) {
          res[col].resid /= norm_in[col];
        }
      }
      return res;
    }

    VCycleQPhiXCoarseEO2(const LatticeInfo& fine_info,
        const LatticeInfo& coarse_info,
        const std::vector<Block>& my_blocks,
        const std::vector<std::shared_ptr<QPhiXSpinorF>>& vecs,
        EOLinearOperator<QPhiXSpinorF,QPhiXGaugeF>& M_fine,
        const Smoother<QPhiXSpinorF,QPhiXGaugeF>& pre_smoother,
        const Smoother<QPhiXSpinorF,QPhiXGaugeF>& post_smoother,
        const LinearSolver<CoarseSpinor,CoarseGauge>& bottom_solver,
        const LinearSolverParamsBase& param)
      : _fine_info(fine_info), _coarse_info(coarse_info),
      _my_blocks(my_blocks),
      _vecs(vecs),
      _M_fine(M_fine),
      _pre_smoother(pre_smoother),
      _post_smoother(post_smoother),
      _bottom_solver(bottom_solver),
      _param(param),
      _Transfer(my_blocks,vecs) {}

    const LatticeInfo& GetInfo() const { return _M_fine.GetInfo(); }
    const CBSubset& GetSubset() const { return _M_fine.GetSubset(); }

  private:
    const LatticeInfo& _fine_info;
    const LatticeInfo& _coarse_info;
    const std::vector<Block>& _my_blocks;
    const std::vector<std::shared_ptr<QPhiXSpinorF>>& _vecs;
    EOLinearOperator<QPhiXSpinorF, QPhiXGaugeF>& _M_fine;
    const Smoother<QPhiXSpinorF,QPhiXGaugeF>& _pre_smoother;
    const Smoother<QPhiXSpinorF,QPhiXGaugeF>& _post_smoother;
    const LinearSolver<CoarseSpinor,CoarseGauge>& _bottom_solver;
    const LinearSolverParamsBase& _param;
    const QPhiXTransfer<QPhiXSpinorF> _Transfer;

};


class VCycleQPhiXCoarseEO3 :
  public LinearSolver<QPhiXSpinor, QPhiXGauge >
{
  public:
    std::vector<LinearSolverResults> operator()(QPhiXSpinor& out,
        const QPhiXSpinor& in, ResiduumType resid_type = RELATIVE ) const
    {
      assert(out.GetNCol() == in.GetNCol());
      IndexType ncol = out.GetNCol();

      int level = _M_fine.GetLevel();
      Timer::TimerAPI::startTimer("VCycleQPhiXCoarseEO3/operator()/level"+std::to_string(level));
      std::vector<LinearSolverResults> res(ncol);
      auto& subset = _M_fine.GetSubset();
      QPhiXSpinorF in_f(_fine_info, ncol);

      ZeroVec(in_f, SUBSET_ALL);

      ConvertSpinor(in,in_f, subset);

      // May want to do these in double later?
      // But this is just a preconditioner.
      // So try SP for now
      QPhiXSpinorF r(_fine_info, ncol);
      QPhiXSpinorF out_f(_fine_info, ncol);


      std::vector<double> norm_in(ncol),  norm_r(ncol);
      ZeroVec(out_f);   // out_f = 0
      CopyVec(r,in_f, subset);  //  r  <- in_f

      norm_r = aux::sqrt(Norm2Vec(r,subset));
      norm_in = norm_r;

      std::vector<double> target(ncol, _param.RsdTarget);
      bool continueP = false;
      for (int col=0; col < ncol; ++col) {
        if ( resid_type == RELATIVE ) {
          target[col] *= norm_r[col];
        }
        if ( norm_r[col] > target[col]) continueP = true;
      }

      // Check if converged already
      if ( !continueP || _param.MaxIter <= 0  ) {
        for (int col=0; col < ncol; ++col) {
          res[col].resid_type = resid_type;
          res[col].n_count = 0;
          res[col].resid = norm_r[col];
          if( resid_type == RELATIVE ) {
            res[col].resid /= norm_r[col];
          }
        }
        Timer::TimerAPI::stopTimer("VCycleQPhiXCoarseEO3/operator()/level"+std::to_string(level));
        return res;
      }

      if( _param.VerboseP ) {
        for (int col=0; col < ncol; ++col) {
          if( resid_type == RELATIVE) {
            MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d col=%d"
                "Initial || r ||/|| b || =%16.8e  Target=%16.8e",
                level, col, norm_r[col]/norm_in[col], _param.RsdTarget);

          }
          else {
            MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d col=%d"
                "Initial || r ||=%16.8e  Target=%16.8e",
                level, col, norm_r[col], _param.RsdTarget);
          }
        }
      }

      // At this point we have to do at least one iteration
      int iter = 0;

      QPhiXSpinorF delta(_fine_info, ncol);
      QPhiXSpinorF tmp(_fine_info, ncol);
      CoarseSpinor coarse_in(_coarse_info, ncol);
      CoarseSpinor coarse_delta(_coarse_info, ncol);

      while ( iter < _param.MaxIter) {
        ++iter;

        Timer::TimerAPI::startTimer("VCycleQPhiXCoarseEO3/presmooth/level"+std::to_string(level));
        ZeroVec(delta,subset);
        // Smoother does not compute a residuum
        _pre_smoother(delta,r);
        Timer::TimerAPI::stopTimer("VCycleQPhiXCoarseEO3/presmooth/level"+std::to_string(level));

        // Update solution
        Timer::TimerAPI::startTimer("VCycleQPhiXCoarseEO3/update/level"+std::to_string(level));
        YpeqXVec(delta,out_f,subset);
        // Update residuum: even odd matrix
        _M_fine(tmp, delta, LINOP_OP);
        YmeqXVec(tmp, r, subset);
        Timer::TimerAPI::stopTimer("VCycleQPhiXCoarseEO3/update/level"+std::to_string(level));

        if ( _param.VerboseP ) {
          std::vector<double> norm_pre_presmooth=aux::sqrt(Norm2Vec(r));
          for (int col=0; col < ncol; ++col) {
            if( resid_type == RELATIVE ) {
              MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d iter=%d col=%d "
                  "After Pre-Smoothing || r ||/||b||=%16.8e Target=%16.8e",
                  level, col, iter, norm_pre_presmooth[col]/norm_in[col], _param.RsdTarget);
            }
            else {
              MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d iter=%d col=%d "
                  "After Pre-Smoothing || r ||=%16.8e Target=%16.8e",
                  level, col, iter, norm_pre_presmooth[col], _param.RsdTarget);
            }
          }
        }

        Timer::TimerAPI::startTimer("VCycleQPhiXCoarseEO3/restrictFrom/level"+std::to_string(level));
        // Coarsen r

#if 1
        _Transfer.R(r,ODD,coarse_in);
#else
        // hit r with clover before coarsening
        _M_fine.M_diag(tmp,r, ODD);
        _Transfer.R(tmp,ODD,coarse_in);
#endif

        Timer::TimerAPI::stopTimer("VCycleQPhiXCoarseEO3/restrictFrom/level"+std::to_string(level));


        Timer::TimerAPI::startTimer("VCycleQPhiXCoarseEO3/bottom_solve/level"+std::to_string(level));
        ZeroVec(coarse_delta);
        _bottom_solver(coarse_delta,coarse_in);
        Timer::TimerAPI::stopTimer("VCycleQPhiXCoarseEO3/bottom_solve/level"+std::to_string(level));


        Timer::TimerAPI::startTimer("VCycleQPhiXCoarseEO3/prolongateTo/level"+std::to_string(level));

        // Reuse Smoothed Delta as temporary for prolongating coarse delta back to fine
        _Transfer.P(coarse_delta, ODD, delta);

        Timer::TimerAPI::stopTimer("VCycleQPhiXCoarseEO3/prolongateTo/level"+std::to_string(level));


        // Update solution
        Timer::TimerAPI::startTimer("VCycleQPhiXCoarseEO3/update/level"+std::to_string(level));
        YpeqXVec(delta,out_f,subset);
        // Update residuum
        _M_fine(tmp, delta, LINOP_OP);
        YmeqXVec(tmp, r,subset);
        Timer::TimerAPI::stopTimer("VCycleQPhiXCoarseEO3/update/level"+std::to_string(level));

        if ( _param.VerboseP ) {
          std::vector<double> norm_pre_postsmooth=aux::sqrt(Norm2Vec(r));
          for (int col=0; col < ncol; ++col) {
            if( resid_type == RELATIVE ) {
              MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d iter=%d col=%d "
                  "After Coarse Solve || r ||/||b||=%16.8e Target=%16.8e",
                  level, col, iter, norm_pre_postsmooth[col]/norm_in[col], _param.RsdTarget);
            }
            else {
              MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d iter=%d col=%d "
                  "After Coarse Solve || r ||=%16.8e Target=%16.8e",
                  level, col, iter, norm_pre_postsmooth[col], _param.RsdTarget);
            }
          }
        }

        //postsmooth
        Timer::TimerAPI::startTimer("VCycleQPhiXCoarseEO3/postsmooth/level"+std::to_string(level));
        ZeroVec(delta,subset);
        _post_smoother(delta,r);
        Timer::TimerAPI::stopTimer("VCycleQPhiXCoarseEO3/postsmooth/level"+std::to_string(level));

        // Update full solution
        Timer::TimerAPI::startTimer("VCycleQPhiXCoarseEO3/update/level"+std::to_string(level));
        YpeqXVec(delta,out_f,subset);
        _M_fine(tmp,delta,LINOP_OP);
        norm_r = aux::sqrt(XmyNorm2Vec(r,tmp,subset));
        Timer::TimerAPI::stopTimer("VCycleQPhiXCoarseEO3/update/level"+std::to_string(level));

        if ( _param.VerboseP ) {
          for (int col=0; col < ncol; ++col) {
            if( resid_type == RELATIVE ) {
              MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d iter=%d col=%d "
                  "After Post-Smoothing || r ||/||b||=%16.8e Target=%16.8e",
                  level, col, iter, norm_r[col]/norm_in[col], _param.RsdTarget);
            }
            else {
              MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d iter=%d col=%d "
                  "After Post-Smoothing || r ||=%16.8e Target=%16.8e",
                  level, col, iter, norm_r[col], _param.RsdTarget);
            }
          }
        }

        // Check convergence
        continueP = false;
        for (int col=0; col < ncol; ++col)
          if ( norm_r[col] >= target[col] ) continueP = true;
        if (!continueP) break;
      }

      // Convert Back Up to DP only on the output subset.
      ZeroVec(out,SUBSET_ALL);
      ConvertSpinor(out_f,out, _M_fine.GetSubset());
      for (int col=0; col < ncol; ++col)  {
        res[col].resid_type = resid_type;
        res[col].n_count = iter;
        res[col].resid= norm_r[col];
        if( resid_type == RELATIVE ) {
          res[col].resid /= norm_in[col];
        }
      }

      Timer::TimerAPI::stopTimer("VCycleQPhiXCoarseEO3/operator()/level"+std::to_string(level));
      return res;
    }

    VCycleQPhiXCoarseEO3(const LatticeInfo& fine_info,
        const LatticeInfo& coarse_info,
        const std::vector<Block>& my_blocks,
        const std::vector<std::shared_ptr<QPhiXSpinorF>>& vecs,
        EOLinearOperator<QPhiXSpinorF,QPhiXGaugeF>& M_fine,
        const Smoother<QPhiXSpinorF,QPhiXGaugeF>& pre_smoother,
        const Smoother<QPhiXSpinorF,QPhiXGaugeF>& post_smoother,
        const LinearSolver<CoarseSpinor,CoarseGauge>& bottom_solver,
        const LinearSolverParamsBase& param)
      : _fine_info(fine_info), _coarse_info(coarse_info),
      _my_blocks(my_blocks),
      _vecs(vecs),
      _M_fine(M_fine),
      _pre_smoother(pre_smoother),
      _post_smoother(post_smoother),
      _bottom_solver(bottom_solver),
      _param(param),
      _Transfer(my_blocks,vecs) {
      int level = _M_fine.GetLevel();
      Timer::TimerAPI::addTimer("VCycleQPhiXCoarseEO3/operator()/level"+std::to_string(level));
      Timer::TimerAPI::addTimer("VCycleQPhiXCoarseEO3/presmooth/level"+std::to_string(level));
      Timer::TimerAPI::addTimer("VCycleQPhiXCoarseEO3/restrictFrom/level"+std::to_string(level));
      Timer::TimerAPI::addTimer("VCycleQPhiXCoarseEO3/prolongateTo/level"+std::to_string(level));
      Timer::TimerAPI::addTimer("VCycleQPhiXCoarseEO3/postsmooth/level"+std::to_string(level));
      Timer::TimerAPI::addTimer("VCycleQPhiXCoarseEO3/bottom_solve/level"+std::to_string(level));
      Timer::TimerAPI::addTimer("VCycleQPhiXCoarseEO3/update/level"+std::to_string(level));
      }

    const LatticeInfo& GetInfo() const { return _M_fine.GetInfo(); }
    const CBSubset& GetSubset() const { return _M_fine.GetSubset(); }

  private:
    const LatticeInfo& _fine_info;
    const LatticeInfo& _coarse_info;
    const std::vector<Block>& _my_blocks;
    const std::vector<std::shared_ptr<QPhiXSpinorF>>& _vecs;
    EOLinearOperator<QPhiXSpinorF, QPhiXGaugeF>& _M_fine;
    const Smoother<QPhiXSpinorF,QPhiXGaugeF>& _pre_smoother;
    const Smoother<QPhiXSpinorF,QPhiXGaugeF>& _post_smoother;
    const LinearSolver<CoarseSpinor,CoarseGauge>& _bottom_solver;
    const LinearSolverParamsBase& _param;
    const QPhiXTransfer<QPhiXSpinorF> _Transfer;

};


}



#endif /* INCLUDE_LATTICE_QPHIX_VCYCLE_QPHIX_COARSE_H_ */
