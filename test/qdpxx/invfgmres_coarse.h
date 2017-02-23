/*
 * invfgmres.h
 *
 *  Created on: Jan 11, 2017
 *      Author: bjoo
 */

#ifndef TEST_QDPXX_INVFGMRES_COARSE_H_
#define TEST_QDPXX_INVFGMRES_COARSE_H_

#include "qdp.h"
#include "lattice/constants.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"
#include <complex>
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_l1_blas.h"
#include "utils/print_utils.h"

#undef DEBUG_SOLVER

#include "fgmres_common.h"

using namespace MG;
using namespace QDP;

namespace MGTesting {



 namespace FGMRESCoarse {

 class Givens {
 public:

	 // Givens rotation.
	 //   There are a variety of ways to choose the rotations
	 //   which can do the job. I employ the method given by Saad
	 //   in Iterative Methods (Sec. 6.5.9 (eq. 6.80)
	 //
	 //  [  conj(c) conj(s)  ] [ h_jj    ] = [ r ]
	 //  [    -s       c     ] [ h_j+1,j ]   [ 0 ]
	 //
	 //  We know that h_j+1,j is a vector norm so imag(h_{j+1,j} = 0)
	 //
	 //  we have: s = h_{j+1,j} / t
	 //           c = h_jj / t
	 //
	 //   t=sqrt( norm2(h_jj) + h_{j+1,j}^2 ) is real and nonnegative
	 //
	 //  so in this case s, is REAL and nonnegative (since t is and h_{j+1,j} is
	 //  but  c is in general complex valued.
	 //
	 //
	 //  using this we find r = conj(c) h_jj + conj(s) h_{j+1,j}
	 //                        = (1/t)* [  conj(h_jj)*h_jj + h_{j+1,j}*h_{j+1,h} ]
	 //                        = (1/t)* [  norm2(h_jj) + h_{j+1,j}^2 ]
	 //                        = (1/t)* [ t^2 ] = t
	 //
	 //  Applying this to a general 2 vector
	 //
	 //   [ conj(c) conj(s) ] [ a ] = [ r_1 ]
	 //   [   -s      c     ] [ b ]   [ r_2 ]
	 //
	 //   we have r_1 = conj(c)*a + conj(s)*b  = conj(c)*a + s*b  since s is real, nonnegative
	 //      and  r_2 = -s*a + c*b
	 //
	 //  NB: In this setup we choose the sine 's' to be real in the rotation.
	 //      This is in contradistinction from LAPACK which typically chooses the cosine 'c' to be real
	 //
	 //
	 // There are some special cases:
	 //   if  h_jj and h_{j+1,j} are both zero we can choose any s and c as long as |c|^2 + |s|^2 =1
	 //   Keeping with the notation that s is real and nonnegative we choose
	 //   if    h_jj != 0 and h_{j+1,h) == 0 => c = sgn(conj(h_jj)), s = 0, r = | h_jj |
	 //   if    h_jj == 0 and h_{j+1,j} == 0 => c = 0, s = 1,  r = 0 = h_{j+1,j}
	 //   if    h_jj == 0 and h_{j+1,j} != 0 => c = 0, s = 1,  r = h_{j+1,j} = h_{j+1,j}
	 //   else the formulae we computed.

	 /*! Given  a marix H, construct the rotator so that H(row,col) = r and H(row+1,col) = 0
	  *
	  *  \param col  the column Input
	  *  \param  H   the Matrix Input
	  */

	 Givens(int col, const multi2d<std::complex<double>>& H) : col_(col)
 {
		 std::complex<double> f = H(col_,col_);
		 std::complex<double> g = H(col_,col_+1);

		 if(  real(f) == 0 && imag(f) == 0  ) {

			 // h_jj is 0
			 c_ = std::complex<double>(0,0);
			 s_ = std::complex<double>(1,0);
			 r_ = g;  // Handles the case when g is also zero
		 }
		 else {
			 if( real(g) == 0 && imag(g) == 0  ) {
				 s_ = std::complex<double>(0,0);

				 // NB: in std::complex norm is what QDP++ calls norm2
				 c_ = conj(f)/sqrt( norm(f) ); //   sgn( conj(f) ) = conj(f) / | conj(f) |  = conj(f) / | f |
				 r_ = std::complex<double>( abs(f), 0 );
			 }
			 else {
				 // Revisit this with
				 double t = sqrt( norm(f) + norm(g) );
				 r_ = std::complex<double>(t,0);
				 c_  = f/t;
				 s_  = g/t;
			 }
		 }
 }

	 /*! Apply the rotation to column col of the matrix H. The
	  *  routine affects col and col+1.
	  *
	  *  \param col  the columm
	  *  \param  H   the matrix
	  */

	 void operator()(int col,  multi2d<std::complex<double>>& H) {
		 if ( col == col_ ) {
			 // We've already done this column and know the answer
			 H(col_,col_) = r_;
			 H(col_,col_+1) = 0;
		 }
		 else {
			 int row = col_; // The row on which the rotation was defined
			 std::complex<double> a = H(col,row);
			 std::complex<double> b = H(col,row+1);
			 H(col,row) = conj(c_)*a + conj(s_)*b;
			 H(col,row+1) = -s_*a + c_*b;
		 }
	 }

	 /*! Apply rotation to Column Vector v */
	 void operator()(multi1d<std::complex<double>>& v) {
		 std::complex<double> a =  v(col_);
		 std::complex<double> b =  v(col_+1);
		 v(col_) = conj(c_)*a + conj(s_)*b;
		 v(col_+1) =  -s_*a + c_*b;

	 }

 private:
	 int col_;
	 std::complex<double> s_;
	 std::complex<double> c_;
	 std::complex<double> r_;
 };



 	 void FlexibleArnoldiT(int n_krylov,
			 const double& rsd_target,
			 const LinearOperator<CoarseSpinor,CoarseGauge>& A,     // Operator
			 const LinearSolver<CoarseSpinor,CoarseGauge>* M,  // Preconditioner
			 multi1d<CoarseSpinor*>& V,                 // Nuisance: Need constructor free way to make these. Init functions?
			 multi1d<CoarseSpinor*>& Z,
			 multi2d<std::complex<double>>& H,
			 multi1d< Givens* >& givens_rots,
			 multi1d<std::complex<double>>& c,
			 int& ndim_cycle,
			 ResiduumType resid_type,
			 bool VerboseP )

 	 {
 		 ndim_cycle = 0;
 		 int level = A.GetLevel();

 		 if( VerboseP ) {
 			 MasterLog(INFO,"FLEXIBLE ARNOLDI: level=%d Flexible Arnoldi Cycle: ",level);
 		 }


 		 // Work by columns:
 		 for(int j=0; j < n_krylov; ++j) {

 			 // Check convention... Z=solution, V=source
 			 // Here we have an opportunity to not precondition...
 			 // If M is a nullpointer.
 			 if( M != nullptr) {
 				 (*M)( *(Z[j]), *(V[j]), resid_type );  // z_j = M^{-1} v_j
 			 }
 			 else {
 				 CopyVec(*(Z[j]), *(V[j]));      // Vector assignment " copy "
 			 }

#ifdef DEBUG_SOLVER
 			 {
 				 MasterLog(DEBUG, "FLEXIBLE ARNOLDI: level=%d norm of Z_j = %16.8e norm of V_j = %16.8e",level, Norm2Vec(*(Z[j])), Norm2Vec(*(V[j])));
 			 }
#endif
 			 CoarseSpinor w( Z[j]->GetInfo() );

 			 A( w, *(Z[j]), LINOP_OP);  // w  = A z_

 			 // Fill out column j
 			 for(int i=0; i <= j ;  ++i ) {
 				 H(j,i) = InnerProductVec(*(V[i]), w);        //  Inner product

 				 // w[s] -= H(j,i)* V[i];                     // y = y - alpha x = CAXPY
 				 std::complex<float> minus_Hji = std::complex<float>(-real(H(j,i)),-imag(H(j,i)));
 				 AxpyVec(minus_Hji, *(V[i]), w);

 			 }

 			 double wnorm=sqrt(Norm2Vec(w));               //  NORM
#ifdef DEBUG_SOLVER
 			 MasterLog(DEBUG, "FLEXIBLE ARNOLDI: level=%d j=%d wnorm=%16.8e\n", level, j, wnorm);
#endif

 			 H(j,j+1) = std::complex<double>(wnorm,0);

 			 // In principle I should check w_norm to be 0, and if it is I should
 			 // terminate: Code Smell -- get rid of 1.0e-14.
 			 if (  fabs( wnorm ) < 1.0e-14 )  {

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
 			 ZeroVec( *(V[j+1]));
 			 AxpyVec( invwnorm, w, *(V[j+1]));

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
 			 if ( toBool( accum_resid <= rsd_target ) ) {
 				 if ( VerboseP ) {
 					 MasterLog(INFO,"FLEXIBLE ARNOLDI: level=%d Cycle Converged at iter = %d",level, j+1);
 				 }
 				 return;
 			 } // if
 		 } // while
 	 } // function
 }; // Namespace FGMRESCoarse


 class FGMRESSolverCoarse : public LinearSolver<MG::CoarseSpinor,MG::CoarseGauge>
  {
  public:

    //! Constructor
    /*!
     * \param A_        Linear operator ( Read )
     * \param invParam  inverter parameters ( Read )
     */
    FGMRESSolverCoarse(const LinearOperator<CoarseSpinor,CoarseGauge>& A,
    			 const MG::LinearSolverParamsBase& params,
    			 const LinearSolver<CoarseSpinor,CoarseGauge>* M_prec=nullptr) : _A(A), _info(A.GetInfo()), _params(static_cast<const FGMRESParams&>(params)), _M_prec(M_prec)
      {


    	H_.resize(_params.NKrylov, _params.NKrylov+1); // This is odd. Shouldn't it be

    	V_.resize(_params.NKrylov+1);
    	Z_.resize(_params.NKrylov+1);

    	for(int i=0; i < _params.NKrylov+1;++i) {
    		V_[i] = new CoarseSpinor(_info);
    		Z_[i] = new CoarseSpinor(_info);
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
    		ZeroVec(*(V_[row]));                  // BLAS ZERO
    		ZeroVec(*(Z_[row]));                  // BLAS ZERO
    		c_[row] = std::complex<double>(0,0);                  // COMPLEX ZERO
    		givens_rots_[row] = nullptr;
    	}

    	for(int row = 0; row < _params.NKrylov; row++) {
    		eta_[row] = std::complex<double>(0,0);
    	}



    }

    ~FGMRESSolverCoarse() {
    	for(int i=0; i < _params.NKrylov+1; ++i) {
    		delete V_[i];
    		delete Z_[i];
    		if( givens_rots_[i] != nullptr ) delete givens_rots_[i];
    	}
    }

    LinearSolverResults operator()(CoarseSpinor& out, const CoarseSpinor& in, ResiduumType resid_type = RELATIVE) const
    {
    	LinearSolverResults res; // Value to return
    	res.resid_type = resid_type;

    	double norm_rhs = sqrt(Norm2Vec(in));   //  || b ||                      BLAS: NORM2
    	double target = _params.RsdTarget;

    	if ( resid_type == RELATIVE) {
    		target *= norm_rhs; // Target  || r || < || b || RsdTarget
    	}

    	const LatticeInfo&  in_info = in.GetInfo();
    	const LatticeInfo&  out_info = out.GetInfo();
    	AssertCompatible( in_info, _A.GetInfo());
    	AssertCompatible( out_info, _A.GetInfo());
    	int level = _A.GetLevel();

    	// Compute ||r||
    	CoarseSpinor r( in_info ); ZeroVec(r);                                                     // BLAS: ZERO
#ifdef DEBUG_SOLVER
    	{
    		double tmp_norm_r = sqrt(Norm2Vec(r));
    		MasterLog(MG::DEBUG, "FGMRES: level=%d norm_rhs=%16.8e r_norm=%16.8e", level, norm_rhs, tmp_norm_r);
    	}
#endif
    	CoarseSpinor tmp(in_info ); ZeroVec(tmp);                                                     // BLAS: COPY
    	CopyVec( r, in );
#ifdef DEBUG_SOLVER
    	{
    		double tmp_norm_in = sqrt(Norm2Vec(in));
    		double tmp_norm_r = sqrt(Norm2Vec(r));
    		MasterLog(MG::DEBUG, "FGMRES: level=%d After copy: in_norm=%16.8e r_norm=%16.8e", level, tmp_norm_in, tmp_norm_r);
    	}
#endif

    	(_A)(tmp, out, LINOP_OP);

    	// r[s] -=tmp;                                                            // BLAS: X=X-Y
    	// The current residuum
    	//    	Double r_norm = sqrt(norm2(r,s));                                      // BLAS: NORM
    	double r_norm = sqrt(XmyNorm2Vec(r,tmp));

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
    		//	V_[0] = beta_inv * r;                       // BLAS: VSCAL
    		ZeroVec(*(V_[0]));
    		AxpyVec(beta_inv,r,*(V_[0]));



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
    				H_,
    				givens_rots_,
    				c_,
    				dim,
					resid_type);

    		int iters_this_cycle = dim;
    		LeastSquaresSolve(H_,c_,eta_, dim); // Solve Least Squares System

    		// Compute the correction dx = sum_j  eta_j Z_j
    		LatticeFermion dx = zero;                         // BLAS: ZERO
    		for(int j=0; j < dim; ++j) {

    			std::complex<float> alpha = std::complex<float>( real(eta_[j]), imag(eta_[j]));
    			AxpyVec(alpha,*(Z_[j]),out);                       // Y = Y + AX => BLAS AXPY
    		}

    		// Recompute r
    		CopyVec(r,in);                                        // BLAS: COPY
    		(_A)(tmp, out, LINOP_OP);
      		r_norm = sqrt(XmyNorm2Vec(r,tmp));

    		// Update total iters
    		iters_total += iters_this_cycle;
    		if ( _params.VerboseP ) {
    			MasterLog(INFO, "FGMRES: level=%d iter=%d || r ||=%16.8e target=%16.8e", level, iters_total, r_norm, target);
    		}

    		// Check if we are done either via convergence, or runnign out of iterations
    		finished = toBool( r_norm <= target ) || (iters_total >= _params.MaxIter);

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

        	MasterLog(INFO,"FGMRES: level=%d Done. Cycles=%d, Iters=%d || r ||=%16.8e",level,
        	n_cycles,iters_total, res.resid, _params.RsdTarget);
    	}
    	else {
    		res.resid /= norm_rhs ;
        	MasterLog(INFO,"FGMRES: level=%d Done. Cycles=%d, Iters=%d || r ||/|| b ||=%16.8e",level,
        	n_cycles,iters_total, res.resid, _params.RsdTarget);
    	}
    	return res;

    }


    void FlexibleArnoldi(int n_krylov,
			 const double rsd_target,
			 multi1d<CoarseSpinor*>& V,
			 multi1d<CoarseSpinor*>& Z,
			 multi2d<std::complex<double>>& H,
			 multi1d< FGMRESCoarse::Givens* >& givens_rots,
			 multi1d<std::complex<double>>& c,
			 int&  ndim_cycle,
			 ResiduumType resid_type) const
    {


    	FGMRESCoarse::FlexibleArnoldiT(n_krylov,
    				rsd_target,
    				_A,
    				_M_prec,
    				V,Z,H,givens_rots,c, ndim_cycle, resid_type, _params.VerboseP);
    }


    void LeastSquaresSolve(const multi2d<std::complex<double>>& H,
    		const multi1d<std::complex<double>>& rhs,
			multi1d<std::complex<double>>& eta,
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
    const LinearOperator<CoarseSpinor,CoarseGauge>& _A;
    const LatticeInfo& _info;
    const FGMRESParams _params;
    const LinearSolver<CoarseSpinor,CoarseGauge>* _M_prec;

    // These can become state variables, as they will need to be
    // handed around
    mutable multi2d<std::complex<double>> H_; // The H matrix
    mutable multi2d<std::complex<double>> R_; // R = H diagonalized with Givens rotations
    mutable multi1d<CoarseSpinor*> V_;  // K(A)
    mutable multi1d<CoarseSpinor*> Z_;  // K(MA)
    mutable multi1d< FGMRESCoarse::Givens* > givens_rots_;

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

    mutable multi1d<std::complex<double>> c_;
    mutable multi1d<std::complex<double>> eta_;

  };

};




#endif /* TEST_QDPXX_INVFGMRES_H_ */
