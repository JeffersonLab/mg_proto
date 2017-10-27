/*
 * vcycle_qphix_coarse.h
 *
 *  Created on: Oct 27, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_QPHIX_VCYCLE_QPHIX_COARSE_H_
#define INCLUDE_LATTICE_QPHIX_VCYCLE_QPHIX_COARSE_H_

#include <lattice/solver.h>
#include <lattice/qphix/qphix_types.h>
#include <lattice/qphix/qphix_blas_wrappers.h>
#include <lattice/coarse/coarse_types.h>
#include <lattice/qphix/qphix_aggregate.h>
namespace MG
{

class VCycleQPhiXCoarse2 :
    public LinearSolver<QPhiXSpinor, QPhiXGauge >
{
public:
  LinearSolverResults operator()(QPhiXSpinor& out,
      const QPhiXSpinor& in, ResiduumType resid_type = RELATIVE ) const
  {
    LinearSolverResults res;

    QPhiXSpinorF in_f(_fine_info);
    ConvertSpinor(in,in_f);

    // May want to do these in double later?
    // But this is just a preconditioner.
    // So try SP for now
    QPhiXSpinorF r(_fine_info);
    QPhiXSpinorF out_f(_fine_info);

    int level = _M_fine.GetLevel();

    double norm_in,  norm_r;
    ZeroVec(out_f);   // out_f = 0
    CopyVec(r,in_f);  //  r  <- in_f

    norm_r = sqrt(Norm2Vec(r));
    norm_in = norm_r;

    double target = _param.RsdTarget;
    if ( resid_type == RELATIVE ) {
      target *= norm_r;
    }

    // Check if converged already
    if ( norm_r <= target  || _param.MaxIter <= 0  ) {
      res.resid_type = resid_type;
      res.n_count = 0;
      res.resid = norm_r;
      if( resid_type == RELATIVE ) {
        res.resid /= norm_r;
      }
      return res;
    }

    if( _param.VerboseP ) {
      if( resid_type == RELATIVE ) {
        MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d Initial || r ||/ || b ||=%16.8e  Target=%16.8e",
            level, norm_r/norm_in, _param.RsdTarget);
      }
      else {
        MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d Initial || r || = %16.8e  Target=%16.8e",
            level, norm_r, _param.RsdTarget);
      }
    }
    // At this point we have to do at least one iteration
    int iter = 0;

    bool continueP = true;
    QPhiXSpinorF delta(_fine_info);
    QPhiXSpinorF tmp(_fine_info);
    CoarseSpinor coarse_in(_coarse_info);
    CoarseSpinor coarse_delta(_coarse_info);
    double done = 1;
    double mone = -1;

    while ( continueP ) {
      ++iter;

      ZeroVec(delta);

      // Smoother does not compute a residuum
      _pre_smoother(delta,r);

      // Update solution

      YpeqXVec(delta,out_f);
      // Update residuum
      _M_fine(tmp,delta, LINOP_OP);
      YmeqXVec(tmp,r);

      if ( _param.VerboseP ) {
        double norm_pre_presmooth=sqrt(Norm2Vec(r));
        if( resid_type == RELATIVE ) {
          MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d iter=%d "
              "After Pre-Smoothing || r ||/||b||=%16.8e Target=%16.8e",
              level, iter, norm_pre_presmooth/norm_in, _param.RsdTarget);
        }
        else {
          MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d iter=%d "
              "After Pre-Smoothing || r ||=%16.8e Target=%16.8e",
              level, iter, norm_pre_presmooth, _param.RsdTarget);
        }
      }


      // Coarsen r
      restrictSpinor(_my_blocks, _vecs, r,coarse_in);

      ZeroVec(coarse_delta);
      LinearSolverResults coarse_res =_bottom_solver(coarse_delta,coarse_in);

      // Reuse Smoothed Delta as temporary for prolongating coarse delta back to fine
      prolongateSpinor(_my_blocks, _vecs, coarse_delta, delta);

      // Update solution
      YpeqXVec(delta,out_f);

      // Update residuum
      _M_fine(tmp, delta, LINOP_OP);
      YmeqXVec(tmp,r);


      if( _param.VerboseP ) {
        double norm_pre_postsmooth = sqrt(Norm2Vec(r));
        if( resid_type == RELATIVE) {
          MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d iter=%d "
              "After Coarse Solve || r ||/|| b ||=%16.8e  Target=%16.8e", level, iter,
              norm_pre_postsmooth/norm_in, _param.RsdTarget);
        }
        else {
          MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d iter=%d "
              "After Coarse Solve || r ||=%16.8e  Target=%16.8e", level, iter,
              norm_pre_postsmooth, _param.RsdTarget);

        }
      }

      ZeroVec(delta);
      _post_smoother(delta,r);

      // Update full solution
      YpeqXVec(delta,out_f);

      _M_fine(tmp,delta,LINOP_OP);
      norm_r = sqrt(XmyNorm2Vec(r,tmp));


      if( _param.VerboseP ) {
        if( resid_type == RELATIVE) {
          MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d iter=%d "
              "After Post-Smoothing || r ||/|| b|| =%16.8e Target=%16.8e",
              level, iter, norm_r/norm_in, _param.RsdTarget);
        }
        else {
          MasterLog(INFO, "VCYCLE (QPhiX->COARSE): level=%d iter=%d "
              "After Post-Smoothing || r || =%16.8e Target=%16.8e",
              level, iter, norm_r, _param.RsdTarget);

        }
      }

      // Check convergence
      continueP = ( iter < _param.MaxIter ) &&  toBool( norm_r > target );
    }

    // Convert Back Up to DP
    ConvertSpinor(out_f,out);
    res.resid_type = resid_type;
    res.n_count = iter;
    res.resid=toDouble(norm_r);
    if( resid_type == RELATIVE ) {
      res.resid /= toDouble(norm_in);
    }
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
    _param(param) {}


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


};



}



#endif /* INCLUDE_LATTICE_QPHIX_VCYCLE_QPHIX_COARSE_H_ */
