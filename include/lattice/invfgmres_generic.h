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
#include <cassert>
#include <algorithm>

#ifdef MG_ENABLE_TIMERS
#include "utils/timer.h"
#endif

#undef DEBUG_SOLVER

#include "lattice/fgmres_common.h"
#include "lattice/givens.h"
#include "lattice/coarse/subset.h"

namespace MG {

namespace {
	/** castToFloat
	 *
	 * 	Cast all elements to float
	 */
	std::vector<std::complex<float>> castToFloat(const std::vector<std::complex<double>>& x) {
		return std::vector<std::complex<float>>(x.begin(), x.end());
	}

	/** negate
	 *
	 * 	Flip sign on all elements
	 */
	template <typename T>
	std::vector<std::complex<float>> negate(const std::vector<std::complex<T>>& x) {
		std::vector<std::complex<float>> r(x.size());
		std::transform(x.begin(), x.end(), r.begin(), [](std::complex<T>& f) { return -f;});
		return r;
	}
}

namespace FGMRESGeneric {

template<typename ST,typename GT>
 void FlexibleArnoldiT(int n_krylov,
     const double& rsd_target,
     const LinearOperator<ST,GT>& A,     // Operator
     const LinearSolver<ST,GT>* M,  // Preconditioner
     std::vector<ST*>& V,                 // Nuisance: Need constructor free way to make these. Init functions?
     std::vector<ST*>& Z,
     ST& w,
     std::vector<Array2d<std::complex<double>>>& H,
     std::vector<std::vector< Givens* >>& givens_rots,
     std::vector<std::vector<std::complex<double>>>& c,
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
   IndexType ncol = V[0]->GetNCol();    
   assert(ncol == w.GetNCol());
   assert(ncol == H.size());
   assert(ncol == givens_rots.size());
   assert(ncol == c.size());

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
       std::vector<std::complex<double>> hji = InnerProductVec(*(V[i]), w,subset);        //  Inner product
       for (int col=0; col < ncol; ++col) H[col](j,i) = hji[col];        //  Inner product

       // w[s] -= H(j,i)* V[i];                     // y = y - alpha x = CAXPY
       AxpyVec(negate(hji), *(V[i]), w, subset);

     }

     std::vector<double> wnorm2=Norm2Vec(w,subset);               //  NORM
#ifdef DEBUG_SOLVER
     for (int col=0; col < ncol; ++col) MasterLog(DEBUG, "FLEXIBLE ARNOLDI: level=%d j=%d wnorm=%16.8e\n", level, j, sqrt(wnorm2[col]));
#endif

     for (int col=0; col < ncol; ++col) H[col](j,j+1) = std::complex<double>(sqrt(wnorm2[col]),0);

     // In principle I should check w_norm to be 0, and if it is I should
     // terminate: Code Smell -- get rid of 1.0e-14.
     for (int col=0; col < ncol; ++col) {
        if (  sqrt( wnorm2[col] ) < 1.0e-14 )  {

           // If wnorm = 0 exactly, then we have converged exactly
           // Replay Givens rots here, how to test?
           if( VerboseP ) {
              MasterLog(INFO,"FLEXIBLE ARNOLDI: level=%d Converged at iter = %d ",level, j+1);
           }
           ndim_cycle = j;
           return;
        }
     }

     std::vector<double> invwnorm(ncol);
     for (int col=0; col < ncol; ++col) invwnorm[col] = 1.0/sqrt(wnorm2[col]);
     // V[j+1] = invwnorm*w;                           // SCAL
     ZeroVec( *(V[j+1]),subset);
     AxpyVec( invwnorm, w, *(V[j+1]),subset);

     // Apply Existing Givens Rotations to this column of H
     for (int col=0; col < ncol; ++col) {
        for(int i=0;i < j; ++i) {
           (*givens_rots[col][i])(j,H[col]);
        }
     }

     // Compute next Givens Rot for this column
     for (int col=0; col < ncol; ++col) givens_rots[col][j] = new Givens(j,H[col]);

     for (int col=0; col < ncol; ++col) {
        (*givens_rots[col][j])(j,H[col]); // Apply it to H
        (*givens_rots[col][j])(c[col]);   // Apply it to the c vector
     }

     // j-ndeflate is the 0 based iteration count
     // j-ndeflate+1 is the 1 based human readable iteration count

     ndim_cycle = j+1;
     bool all_converged = true;
     for (int col=0; col < ncol; ++col) {
        double accum_resid = fabs(real(c[col][j+1]));

        if ( VerboseP ) {
           MasterLog(INFO,"FLEXIBLE ARNOLDI: level=%d Iter=%d col=%d || r || = %16.8e Target=%16.8e",level,
                 j+1, col, accum_resid,rsd_target);

        }
        if (  accum_resid <= rsd_target  ) {
           if ( VerboseP ) {
              MasterLog(INFO,"FLEXIBLE ARNOLDI: level=%d col=%d Cycle Converged at iter = %d",level, col, j+1);
           }
        } else {
           all_converged = false;
        }
     }
     if (all_converged) return;
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

    initialize(0);

#ifdef MG_ENABLE_TIMERS
    int level = _A.GetLevel();
    timerAPI = MG::Timer::TimerAPI::getInstance();
    timerAPI->addTimer("FGMRESSolverGeneric/operator()/level"+std::to_string(level));
    timerAPI->addTimer("FGMRESSolverGeneric/preconditioner/level"+std::to_string(level));
#endif

  }

private:

  void initialize(IndexType ncol) {  
    if (H_.size() == ncol) return;

    destroy();

    H_.resize(ncol);
    V_.resize(ncol);
    Z_.resize(ncol);
    givens_rots_.resize(ncol);
    c_.resize(ncol);
    eta_.resize(ncol);

    for (int col=0; col < ncol; ++col) {
      H_[col].resize(_params.NKrylov, _params.NKrylov+1); // This is odd. Shouldn't it be

      V_[col].resize(_params.NKrylov+1);
      Z_[col].resize(_params.NKrylov+1);

      for(int i=0; i < _params.NKrylov+1;++i) {
         V_[col][i] = new ST(_info);
         Z_[col][i] = new ST(_info);
      }


      givens_rots_[col].resize(_params.NKrylov+1);
      c_[col].resize(_params.NKrylov+1);
      eta_[col].resize(_params.NKrylov);


      for(int i =0; i < _params.NKrylov; i++) {
         for(int row = 0; row < _params.NKrylov+1; row++) {
            H_[col](i,row) = std::complex<double>(0,0);       // COMPLEX ZERO
         }
      }

      for(int row = 0; row < _params.NKrylov+1; row++) {
         ZeroVec(*(V_[col][row]),SUBSET_ALL);                  // BLAS ZERO
         ZeroVec(*(Z_[col][row]),SUBSET_ALL);                  // BLAS ZERO
         c_[col][row] = std::complex<double>(0,0);                  // COMPLEX ZERO
         givens_rots_[col][row] = nullptr;
      }

      for(int row = 0; row < _params.NKrylov; row++) {
         eta_[col][row] = std::complex<double>(0,0);
      }
    }
  }

  void destroy() {
     for (int col=0; col < H_.size(); ++col)  {
        for(int i=0; i < _params.NKrylov+1; ++i) {
           delete V_[col][i];
           delete Z_[col][i];
           if( givens_rots_[col][i] != nullptr ) delete givens_rots_[col][i];
        }
     }
  }

public:
  FGMRESSolverGeneric(std::shared_ptr<const LinearOperator<ST,GT>> A,
        const MG::LinearSolverParamsBase& params,
        const LinearSolver<ST,GT>* M_prec=nullptr)  : FGMRESSolverGeneric(*A,params,M_prec) {}

  ~FGMRESSolverGeneric()
  {
    destroy();
  }

  std::vector<LinearSolverResults> operator()(ST& out, const ST& in, ResiduumType resid_type = RELATIVE) const override

    {
        int level = _A.GetLevel();
#ifdef MG_ENABLE_TIMERS
        timerAPI->startTimer("FGMRESSolverGeneric/operator()/level"+std::to_string(level));
#endif

      assert(in.GetNCol() == out.GetNCol());

      std::vector<LinearSolverResults> res; // Value to return
      const CBSubset& subset = _A.GetSubset();

      IndexType ncol = in.GetNCol();
      for (int col=0; col < ncol; ++col) res[col].resid_type = resid_type;

      std::vector<double> norm2_rhs = Norm2Vec(in,subset);   //  || b ||                      BLAS: NORM2
      std::vector<double> target(ncol, _params.RsdTarget);

      if ( resid_type == RELATIVE) {
        for (int col=0; col < ncol; ++col) target[col] *= sqrt(norm2_rhs[col]); // Target  || r || < || b || RsdTarget
      }

      const LatticeInfo&  in_info = in.GetInfo();
      const LatticeInfo&  out_info = out.GetInfo();
      AssertCompatible( in_info, _A.GetInfo());
      AssertCompatible( out_info, _A.GetInfo());


      // Temporaries - passed into flexible Arnoldi
      ST w( in_info, ncol );

      // Compute ||r||
      ST r( in_info, ncol ); ZeroVec(r,subset);                                                     // BLAS: ZERO
#ifdef DEBUG_SOLVER
      {
        std::vector<double> tmp_norm2_r = Norm2Vec(r,subset);
        for (int col=0; col < ncol; ++col) {
           MasterLog(MG::DEBUG, "FGMRES: level=%d col=%d norm_rhs=%16.8e r_norm=%16.8e", level, col, sqrt(norm2_rhs[col]), sqrt(tmp_norm2_r[col]));
        }
      }
#endif
      ST tmp(in_info, ncol ); ZeroVec(tmp,subset);                                                     // BLAS: COPY
      CopyVec( r, in , subset);
#ifdef DEBUG_SOLVER
      {
        std::vector<double> tmp_norm2_r_in = Norm2Vec(in,subset);
        std::vector<double> tmp_norm2_r = Norm2Vec(r,subset);
        for (int col=0; col < ncol; ++col) {
           MasterLog(MG::DEBUG, "FGMRES: level=%d col=%d After copy: in_norm=%16.8e r_norm=%16.8e", level, col, sqrt(tmp_norm2_in[col]), sqrt(tmp_norm2_r[col]));
        }
      }
#endif

      (_A)(tmp, out, LINOP_OP);

      // r[s] -=tmp;                                                            // BLAS: X=X-Y
      // The current residuum
      //      Double r_norm = sqrt(norm2(r,s));                                      // BLAS: NORM
      std::vector<double> r_norm2 = XmyNorm2Vec(r,tmp,subset);

      // Initialize iterations
      int iters_total = 0;
      if ( _params.VerboseP ) {
        for (int col=0; col < ncol; ++col) {
           MasterLog(INFO,"FGMRES: level=%d col=%d iters=%d || r ||=%16.8e Target || r ||=%16.8e", level, col, iters_total, sqrt(r_norm2[col]),target);
        }
      }

      bool all_converged = true;
      for (int col=0; col < ncol; ++col) {
         if( sqrt(r_norm2[col]) < target[col] )  {
            res[col].n_count = 0;
            res[col].resid = sqrt(r_norm2[col]) ;
            if( resid_type == ABSOLUTE ) {
               if( _params.VerboseP ) {
                  MasterLog(INFO,"FGMRES: level=%d  col=%d Solve Converged: iters=0  Final Absolute || r ||=%16.8e",level,col, res[col].resid);
               }
            }
            else {
               res[col].resid /= sqrt(norm2_rhs[col]);
               if( _params.VerboseP ) {
                  MasterLog(INFO,"FGMRES: level=%d col=%d Solve Converged: iters=0  Final Absolute || r ||/|| b ||=%16.8e", level,col, res[col].resid);
               }
               all_converged = false;
            }
         }
      }

      if (all_converged) {
#ifdef MG_ENABLE_TIMERS
            timerAPI->stopTimer("FGMRESSolverGeneric/operator()/level"+std::to_string(level));
#endif

            return res;
      }

      int n_cycles = 0;

      // We keep executing cycles until we are finished
      while( iters_total < _params.MaxIter ) {

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
        for (int col=0; col < ncol; ++col) {
           for(int j=0; j < c_.size(); ++j) {
              c_[col][j] = std::complex<double>(0);
           }
           c_[col][0] = sqrt(r_norm2[col]);
        }

        // Set up initial V[0] = rhs / || r^2 ||
        // and since we are solving for A delta x = r
        // the rhs is 'r'
        //
        std::vector<double> beta_inv(ncol);
        for (int col=0; col < ncol; ++col) beta_inv[col] = 1.0/sqrt(r_norm2[col]);
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
        for (int col=0; col < ncol; ++col) LeastSquaresSolve(H_[col],c_[col],eta_[col], dim); // Solve Least Squares System

        // Compute the correction dx = sum_j  eta_j Z_j
        //LatticeFermion dx = zero;                         // BLAS: ZERO
        for(int j=0; j < dim; ++j) {

          std::vector<std::complex<float>> alpha(ncol);
          for (int col=0; col < ncol; ++col) alpha[col] = std::complex<float>( std::real(eta_[col][j]), std::imag(eta_[col][j]));
          AxpyVec(alpha,*(Z_[j]),out,subset);                       // Y = Y + AX => BLAS AXPY
        }

        // Recompute r
        CopyVec(r,in,subset);                                        // BLAS: COPY
        (_A)(tmp, out, LINOP_OP);
        r_norm2 = XmyNorm2Vec(r,tmp,subset);

        // Update total iters
        iters_total += iters_this_cycle;
        if ( _params.VerboseP ) {
          for (int col=0; col < ncol; ++col) {
             MasterLog(INFO, "FGMRES: level=%d iter=%d col=%d || r ||=%16.8e target=%16.8e", level, iters_total, col, sqrt(r_norm2[col]), target[col]);
          }
        }

        // Init matrices should've initialized this but just in case this is e.g. a second call or something.
        for(int j=0; j < _params.NKrylov; ++j) {
          if ( givens_rots_[j] != nullptr ) {
            delete givens_rots_[j];
            givens_rots_[j] = nullptr;
          }
        }

        // Check if all columns are converged
        bool finished = true;
        for (int col=0; col < ncol; ++col)
           if ( sqrt(r_norm2[col]) > target[col] )
            finished = false;
        if (finished) break;

      } // Next Cycle...

      // Either we've exceeded max iters, or we have converged in either case set res:
      for (int col=0; col < ncol; ++col) {
         res[col].n_count = iters_total;
         res[col].resid = sqrt(r_norm2[col]) ;
         if( resid_type == ABSOLUTE ) {
            if( _params.VerboseP ) {
               MasterLog(INFO,"FGMRES: level=%d col=%d Done. Cycles=%d, Iters=%d || r ||=%16.8e",level,col,
                     n_cycles,iters_total, res[col].resid, _params.RsdTarget);
            }
         }
         else {
            res[col].resid /= sqrt(norm2_rhs[col]) ;
            if( _params.VerboseP ) {
               MasterLog(INFO,"FGMRES: level=%d col=%d Done. Cycles=%d, Iters=%d || r ||/|| b ||=%16.8e",level,col,
                     n_cycles,iters_total, res[col].resid, _params.RsdTarget);
            }
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
    mutable std::vector<Array2d<std::complex<double>>> H_; // The H matrix
    mutable std::vector<Array2d<std::complex<double>>> R_; // R = H diagonalized with Givens rotations
    mutable std::vector<std::vector<ST*>> V_;  // K(A)
    mutable std::vector<std::vector<ST*>> Z_;  // K(MA)
    mutable std::vector<std::vector< Givens* >> givens_rots_;

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

    mutable std::vector<std::vector<std::complex<double>>> c_;
    mutable std::vector<std::vector<std::complex<double>>> eta_;
    
#ifdef MG_ENABLE_TIMERS
    std::shared_ptr<Timer::TimerAPI> timerAPI;
#endif
  };

} // namespace FGMRESGeneric
} // namespace MG




#endif /* TEST_QDPXX_INVFGMRES_H_ */
