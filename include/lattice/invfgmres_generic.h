/*
 * invfgmres.h
 *
 *  Created on: Jan 11, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_INVFGMRES_GENERIC_H_
#define INCLUDE_LATTICE_INVFGMRES_GENERIC_H_

#include "MG_config.h"
#include "lattice/constants.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"
#include <complex>
#include "lattice/array2d.h"
#include <vector>
#include <cmath>
#include <memory>

#ifdef MG_ENABLE_TIMERS
#include "utils/timer.h"
#endif

#undef DEBUG_SOLVER

#include "lattice/fgmres_common.h"
#include "lattice/givens.h"
#include "lattice/coarse/subset.h"

namespace MG {

namespace FGMRESGeneric {

template<typename ST,typename GT>
 void FlexibleArnoldiT(int n_krylov,
     const double& rsd_target,
     const LinearOperator<ST,GT>& A,     // Operator
     const LinearSolver<ST,GT>* M,  // Preconditioner
     std::vector<ST*>& V,                 // Nuisance: Need constructor free way to make these. Init functions?
     std::vector<ST*>& Z,
	 ST& w,
     Array2d<std::complex<double>>& H,
     std::vector< Givens* >& givens_rots,
     std::vector<std::complex<double>>& c,
     int& ndim_cycle,
     ResiduumType resid_type,
     bool VerboseP )

 {
#ifdef MG_ENABLE_TIMERS
	auto  timerAPI = MG::Timer::TimerAPI::getInstance();
#endif
   ndim_cycle = 0;
   int level = A.GetLevel();
   const CBSubset& subset = A.GetSubset();

   if( VerboseP ) {
     MasterLog(INFO,"FLEXIBLE ARNOLDI: level=%d Flexible Arnoldi Cycle: ",level);
   }


   // Work by columns:
   for(int j=0; j < n_krylov; ++j) {

	   // Check convention... Z=solution, V=source
	   // Here we have an opportunity to not precondition...
	   // If M is a nullpointer.
	   if( M != nullptr) {

		   // solve z_j = M^{-1} v_j
		   //
		   // solve into tmpsolve in case the solution process changes the
		   // The Z_j are pre-zeroed
		   // The V_js are only changed on the subset so off-subset should be zeroed
		   // non-subset part.
		   // But I will go through a tmpsolve temporary because
		   // a proper unprec solver may overwrite the off checkerboard parts with a reconstruct etc.
		   //
#ifdef MG_ENABLE_TIMERS
		   timerAPI->startTimer("FGMRESSolverGeneric/preconditioner/level"+std::to_string(level));
#endif

		   (*M)( *Z[j], *(V[j]), resid_type );  // z_j = M^{-1} v_j

#ifdef MG_ENABLE_TIMERS
		   timerAPI->stopTimer("FGMRESSolverGeneric/preconditioner/level"+std::to_string(level));
#endif
	   }
	   else {
		   CopyVec(*(Z[j]), *(V[j]), subset);      // Vector assignment " copy "
	   }

#ifdef DEBUG_SOLVER
     {
       MasterLog(DEBUG, "FLEXIBLE ARNOLDI: level=%d norm of Z_j = %16.8e norm of V_j = %16.8e",level, Norm2Vec(*(Z[j]),subset), Norm2Vec(*(V[j]),subset));
     }
#endif

     A( w, *(Z[j]), LINOP_OP);  // w  = A z_

     // Fill out column j
     for(int i=0; i <= j ;  ++i ) {
       H(j,i) = InnerProductVec(*(V[i]), w,subset);        //  Inner product

       // w[s] -= H(j,i)* V[i];                     // y = y - alpha x = CAXPY
       std::complex<float> minus_Hji = std::complex<float>(-real(H(j,i)),-imag(H(j,i)));
       AxpyVec(minus_Hji, *(V[i]), w, subset);

     }

     double wnorm=sqrt(Norm2Vec(w,subset));               //  NORM
#ifdef DEBUG_SOLVER
     MasterLog(DEBUG, "FLEXIBLE ARNOLDI: level=%d j=%d wnorm=%16.8e\n", level, j, wnorm);
#endif

     H(j,j+1) = std::complex<double>(wnorm,0);

     // In principle I should check w_norm to be 0, and if it is I should
     // terminate: Code Smell -- get rid of 1.0e-14.
     if (  std::fabs( wnorm ) < 1.0e-14 )  {

       // If wnorm = 0 exactly, then we have converged exactly
       // Replay Givens rots here, how to test?
       if( VerboseP ) {
         MasterLog(INFO,"FLEXIBLE ARNOLDI: level=%d Converged at iter = %d ",level, j+1);
       }
       ndim_cycle = j;
       return;
     }

     double invwnorm = (double)1/wnorm;
     // V[j+1] = invwnorm*w;                           // SCAL
     ZeroVec( *(V[j+1]),subset);
     AxpyVec( invwnorm, w, *(V[j+1]),subset);

     // Apply Existing Givens Rotations to this column of H
     for(int i=0;i < j; ++i) {
       (*givens_rots[i])(j,H);
     }

     // Compute next Givens Rot for this column
     givens_rots[j] = new Givens(j,H);

     (*givens_rots[j])(j,H); // Apply it to H
     (*givens_rots[j])(c);   // Apply it to the c vector

     double accum_resid = fabs(real(c[j+1]));

     // j-ndeflate is the 0 based iteration count
     // j-ndeflate+1 is the 1 based human readable iteration count

     if ( VerboseP ) {
       MasterLog(INFO,"FLEXIBLE ARNOLDI: level=%d Iter=%d || r || = %16.8e Target=%16.8e",level,
                                   j+1, accum_resid,rsd_target);

     }
     ndim_cycle = j+1;
     if (  accum_resid <= rsd_target  ) {
       if ( VerboseP ) {
         MasterLog(INFO,"FLEXIBLE ARNOLDI: level=%d Cycle Converged at iter = %d",level, j+1);
       }
       return;
     } // if
   } // while
 } // function



template<typename ST, typename GT>
 class FGMRESSolverGeneric : public LinearSolver<ST,GT>
  {
  public:

    //! Constructor
    /*!
     * \param A_        Linear operator ( Read )
     * \param invParam  inverter parameters ( Read )
     */
  FGMRESSolverGeneric(const LinearOperator<ST,GT>& A,
      const MG::LinearSolverParamsBase& params,
      const LinearSolver<ST,GT>* M_prec=nullptr)  : _A(A), _info(A.GetInfo()),
      _params(static_cast<const FGMRESParams&>(params)), _M_prec(M_prec)
  {

	const CBSubset& subset = A.GetSubset();

    H_.resize(_params.NKrylov, _params.NKrylov+1); // This is odd. Shouldn't it be

    V_.resize(_params.NKrylov+1);
    Z_.resize(_params.NKrylov+1);

    for(int i=0; i < _params.NKrylov+1;++i) {
      V_[i] = new ST(_info);
      Z_[i] = new ST(_info);
    }


    givens_rots_.resize(_params.NKrylov+1);
    c_.resize(_params.NKrylov+1);
    eta_.resize(_params.NKrylov);


    for(int col =0; col < _params.NKrylov; col++) {
      for(int row = 0; row < _params.NKrylov+1; row++) {
        H_(col,row) = std::complex<double>(0,0);       // COMPLEX ZERO
      }
    }

    for(int row = 0; row < _params.NKrylov+1; row++) {
      ZeroVec(*(V_[row]),SUBSET_ALL);                  // BLAS ZERO
      ZeroVec(*(Z_[row]),SUBSET_ALL);                  // BLAS ZERO
      c_[row] = std::complex<double>(0,0);                  // COMPLEX ZERO
      givens_rots_[row] = nullptr;
    }

    for(int row = 0; row < _params.NKrylov; row++) {
      eta_[row] = std::complex<double>(0,0);
    }

#ifdef MG_ENABLE_TIMERS
    int level = _A.GetLevel();
    timerAPI = MG::Timer::TimerAPI::getInstance();
    timerAPI->addTimer("FGMRESSolverGeneric/operator()/level"+std::to_string(level));
    timerAPI->addTimer("FGMRESSolverGeneric/preconditioner/level"+std::to_string(level));
#endif

  }

  FGMRESSolverGeneric(std::shared_ptr<const LinearOperator<ST,GT>> A,
        const MG::LinearSolverParamsBase& params,
        const LinearSolver<ST,GT>* M_prec=nullptr)  : FGMRESSolverGeneric(*A,params,M_prec) {}

    ~FGMRESSolverGeneric()
    {
      for(int i=0; i < _params.NKrylov+1; ++i) {
        delete V_[i];
        delete Z_[i];
        if( givens_rots_[i] != nullptr ) delete givens_rots_[i];
      }
    }

    LinearSolverResults operator()(ST& out, const ST& in, ResiduumType resid_type = RELATIVE) const override

    {
    	int level = _A.GetLevel();
#ifdef MG_ENABLE_TIMERS
        timerAPI->startTimer("FGMRESSolverGeneric/operator()/level"+std::to_string(level));
#endif


      LinearSolverResults res; // Value to return
      const CBSubset& subset = _A.GetSubset();

      res.resid_type = resid_type;

      double norm_rhs = sqrt(Norm2Vec(in,subset));   //  || b ||                      BLAS: NORM2
      double target = _params.RsdTarget;

      if ( resid_type == RELATIVE) {
        target *= norm_rhs; // Target  || r || < || b || RsdTarget
      }

      const LatticeInfo&  in_info = in.GetInfo();
      const LatticeInfo&  out_info = out.GetInfo();
      AssertCompatible( in_info, _A.GetInfo());
      AssertCompatible( out_info, _A.GetInfo());


      // Temporaries - passed into flexible Arnoldi
      ST w( in_info );

      // Compute ||r||
      ST r( in_info ); ZeroVec(r,subset);                                                     // BLAS: ZERO
#ifdef DEBUG_SOLVER
      {
        double tmp_norm_r = sqrt(Norm2Vec(r,subset));
        MasterLog(MG::DEBUG, "FGMRES: level=%d norm_rhs=%16.8e r_norm=%16.8e", level, norm_rhs, tmp_norm_r);
      }
#endif
      ST tmp(in_info ); ZeroVec(tmp,subset);                                                     // BLAS: COPY
      CopyVec( r, in , subset);
#ifdef DEBUG_SOLVER
      {
        double tmp_norm_in = sqrt(Norm2Vec(in,subset));
        double tmp_norm_r = sqrt(Norm2Vec(r,subset));
        MasterLog(MG::DEBUG, "FGMRES: level=%d After copy: in_norm=%16.8e r_norm=%16.8e", level, tmp_norm_in, tmp_norm_r);
      }
#endif

      (_A)(tmp, out, LINOP_OP);

      // r[s] -=tmp;                                                            // BLAS: X=X-Y
      // The current residuum
      //      Double r_norm = sqrt(norm2(r,s));                                      // BLAS: NORM
      double r_norm = sqrt(XmyNorm2Vec(r,tmp,subset));

      // Initialize iterations
      int iters_total = 0;
      if ( _params.VerboseP ) {
        MasterLog(INFO,"FGMRES: level=%d iters=%d || r ||=%16.8e Target || r ||=%16.8e", level, iters_total, r_norm,target);
      }

      if( r_norm < target )  {
        res.n_count = 0;
        res.resid = r_norm ;
        if( resid_type == ABSOLUTE ) {
          if( _params.VerboseP ) {
            MasterLog(INFO,"FGMRES: level=%d  Solve Converged: iters=0  Final Absolute || r ||=%16.8e",level, res.resid);
          }
        }
        else {
          res.resid /= norm_rhs;
          if( _params.VerboseP ) {
            MasterLog(INFO,"FGMRES: level=%d Solve Converged: iters=0  Final Absolute || r ||/|| b ||=%16.8e", level, res.resid);
          }
        }

#ifdef MG_ENABLE_TIMERS
        timerAPI->stopTimer("FGMRESSolverGeneric/operator()/level"+std::to_string(level));
#endif

        return res;
      }

      int n_cycles = 0;

      // We are done if norm is sufficiently accurate,
      bool finished = ( r_norm <= target ) ;

      // We keep executing cycles until we are finished
      while( !finished ) {

        // If not finished, we should do another cycle with RHS='r' to find dx to add to psi.
        ++n_cycles;


        int dim; // dim at the end of cycle (in case we terminate in-cycle
        int n_krylov  = _params.NKrylov;

        // We are either first cycle, or
        // We have no deflation subspace ie we are regular FGMRES
        // and we are just restarting
        //
        // Set up initial vector c = [ beta, 0 ... 0 ]^T
        // In this case beta should be the r_norm, ie || r || (=|| b || for the first cycle)
        //
        // NB: We will have a copy of this called 'g' onto which we will
        // apply Givens rotations to get an inline estimate of the residuum
        for(int j=0; j < c_.size(); ++j) {
          c_[j] = std::complex<double>(0);
        }
        c_[0] = r_norm;

        // Set up initial V[0] = rhs / || r^2 ||
        // and since we are solving for A delta x = r
        // the rhs is 'r'
        //
        double beta_inv = (double)1/r_norm;
        //  V_[0] = beta_inv * r;                       // BLAS: VSCAL
        ZeroVec(*(V_[0]),subset);
        AxpyVec(beta_inv,r,*(V_[0]),subset);



        // Carry out Flexible Arnoldi process for the cycle

        // We are solving for the defect:   A dx = r
        // so the RHS in the Arnoldi process is 'r'
        // NB: We recompute a true 'r' after every cycle
        // So in the cycle we could in principle
        // use reduced precision... TBInvestigated.
        FlexibleArnoldi(n_krylov,
            target,
            V_,
            Z_,
			w,
			H_,
            givens_rots_,
            c_,
            dim,
            resid_type);

        int iters_this_cycle = dim;
        LeastSquaresSolve(H_,c_,eta_, dim); // Solve Least Squares System

        // Compute the correction dx = sum_j  eta_j Z_j
        //LatticeFermion dx = zero;                         // BLAS: ZERO
        for(int j=0; j < dim; ++j) {

          std::complex<float> alpha = std::complex<float>( real(eta_[j]), imag(eta_[j]));
          AxpyVec(alpha,*(Z_[j]),out,subset);                       // Y = Y + AX => BLAS AXPY
        }

        // Recompute r
        CopyVec(r,in,subset);                                        // BLAS: COPY
        (_A)(tmp, out, LINOP_OP);
        r_norm = sqrt(XmyNorm2Vec(r,tmp,subset));

        // Update total iters
        iters_total += iters_this_cycle;
        if ( _params.VerboseP ) {
          MasterLog(INFO, "FGMRES: level=%d iter=%d || r ||=%16.8e target=%16.8e", level, iters_total, r_norm, target);
        }

        // Check if we are done either via convergence, or runnign out of iterations
        finished = ( r_norm <= target ) || (iters_total >= _params.MaxIter);

        // Init matrices should've initialized this but just in case this is e.g. a second call or something.
        for(int j=0; j < _params.NKrylov; ++j) {
          if ( givens_rots_[j] != nullptr ) {
            delete givens_rots_[j];
            givens_rots_[j] = nullptr;
          }
        }
      } // Next Cycle...

      // Either we've exceeded max iters, or we have converged in either case set res:
      res.n_count = iters_total;
      res.resid = r_norm ;
      if( resid_type == ABSOLUTE ) {
        if( _params.VerboseP ) {
          MasterLog(INFO,"FGMRES: level=%d Done. Cycles=%d, Iters=%d || r ||=%16.8e",level,
              n_cycles,iters_total, res.resid, _params.RsdTarget);
        }
      }
      else {
        res.resid /= norm_rhs ;
        if( _params.VerboseP ) {
          MasterLog(INFO,"FGMRES: level=%d Done. Cycles=%d, Iters=%d || r ||/|| b ||=%16.8e",level,
              n_cycles,iters_total, res.resid, _params.RsdTarget);
        }
      }
#ifdef MG_ENABLE_TIMERS
        timerAPI->stopTimer("FGMRESSolverGeneric/operator()/level"+std::to_string(level));

#endif

      return res;
    }

    void FlexibleArnoldi(int n_krylov,
			 const double rsd_target,
			 std::vector<ST*>& V,
			 std::vector<ST*>& Z,
			 ST& w,
			 Array2d<std::complex<double>>& H,
			 std::vector<Givens* >& givens_rots,
			 std::vector<std::complex<double>>& c,
			 int&  ndim_cycle,
			 ResiduumType resid_type) const
    {

      FlexibleArnoldiT<ST,GT>(n_krylov,
            rsd_target,
            _A,
            _M_prec,
            V,Z,w, H,givens_rots,c, ndim_cycle, resid_type, _params.VerboseP);
    }


    void LeastSquaresSolve(const Array2d<std::complex<double>>& H,
    		const std::vector<std::complex<double>>& rhs,
			std::vector<std::complex<double>>& eta,
			int n_cols) const
    {
      /* Assume here we have a square matrix with an extra row.
           Hence the loop counters are the columns not the rows.
           NB: For an augmented system this will change */
      eta[n_cols-1] = rhs[n_cols-1]/H(n_cols-1,n_cols-1);
      for(int row = n_cols-2; row >= 0; --row) {
        eta[row] = rhs[row];
        for(int col=row+1; col <  n_cols; ++col) {
          eta[row] -= H(col,row)*eta[col];
        }
        eta[row] /= H(row,row);
      }
    }

  private:
    const LinearOperator<ST,GT>& _A;
    const LatticeInfo& _info;
    const FGMRESParams _params;
    const LinearSolver<ST,GT>* _M_prec;

    // These can become state variables, as they will need to be
    // handed around
    mutable Array2d<std::complex<double>> H_; // The H matrix
    mutable Array2d<std::complex<double>> R_; // R = H diagonalized with Givens rotations
    mutable std::vector<ST*> V_;  // K(A)
    mutable std::vector<ST*> Z_;  // K(MA)
    mutable std::vector< Givens* > givens_rots_;

    // This is the c = V^H_{k+1} r vector (c is frommers Notation)
    // For regular FGMRES I need to keep only the basis transformed
    // version of it, for rotating by the Q's (I call this 'g')
    // However, for FGMRES-DR I need this because I need
    //  || c - H eta || to extend the G_k matrix to form V_{k+1}
    // it is made of the previous v_{k+1} by explicitly evaluating c
    // except at the end of the first cycle when the V_{k+1} are not
    // yet available

    // Once G_k+1 is available, I can form the QR decomposition
    // and set  my little g = Q^H and then as I do the Arnoldi
    // I can then work it with the Givens rotations...

    mutable std::vector<std::complex<double>> c_;
    mutable std::vector<std::complex<double>> eta_;
    
#ifdef MG_ENABLE_TIMERS
    std::shared_ptr<Timer::TimerAPI> timerAPI;
#endif
  };

} // namespace FGMRESGeneric
} // namespace MG




#endif /* TEST_QDPXX_INVFGMRES_H_ */
