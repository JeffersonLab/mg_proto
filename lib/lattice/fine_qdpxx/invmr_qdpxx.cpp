/*
 * invmr.cpp
 *
 *  Created on: Mar 21, 2017
 *      Author: bjoo
 */

#include <lattice/fine_qdpxx/invmr_qdpxx.h>

#include "qdp.h"
#include "lattice/constants.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"
#include "lattice/mr_params.h"
using namespace QDP;

namespace MG  {

  //! Minimal-residual (MR) algorithm for a generic Linear Operator
  /*! \ingroup invert
   * This subroutine uses the Minimal Residual (MR) algorithm to determine
   * the solution of the set of linear equations. Here we allow M to be nonhermitian.
   *
   *   	    Chi  =  M . Psi
   *
   * Algorithm:
   *
   *  Psi[0]                                      Argument
   *  r[0]    :=  Chi  -  M . Psi[0] ;            Initial residual
   *  IF |r[0]| <= RsdCG |Chi| THEN RETURN;       Converged?
   *  FOR k FROM 1 TO MaxCG DO                    MR iterations
   *      a[k-1]  := <M.r[k-1],r[k-1]> / <M.r[k-1],M.r[k-1]> ;
   *      ap[k-1] := MRovpar * a[k] ;             Overrelaxtion step
   *      Psi[k]  += ap[k-1] r[k-1] ;   	        New solution std::vector
   *      r[k]    -= ap[k-1] A . r[k-1] ;         New residual
   *      IF |r[k]| <= RsdCG |Chi| THEN RETURN;   Converged?

   * Arguments:

   *  \param M       Linear Operator             (Read)
   *  \param chi     Source                      (Read)
   *  \param psi     Solution                    (Modify)
   *  \param RsdCG   MR residual accuracy        (Read)
   *  \param MRovpar Overrelaxation parameter    (Read)
   *  \param MaxMR   Maximum MR iterations       (Read)

   * Local Variables:

   *  r   	Residual std::vector
   *  cp  	| r[k] |**2
   *  c   	| r[k-1] |**2
   *  k   	MR iteration counter
   *  a   	a[k]
   *  d   	< M.r[k], M.r[k] >
   *  R_Aux     Temporary for  M.Psi
   *  Mr        Temporary for  M.r

   * Global Variables:

   *  MaxMR       Maximum number of MR iterations allowed
   *  RsdCG       Maximum acceptable MR residual (relative to source)
   *
   * Subroutines:
   *
   *  M           Apply matrix to std::vector
   *
   * @{
   */

  template<typename Spinor, typename Gauge>
  LinearSolverResults
  InvMR_a(const LinearOperator<Spinor,Gauge>& M,
	  const Spinor& chi,
	  Spinor& psi,
	  const Real& OmegaRelax,
	  const Real& RsdTarget,
	  int MaxIter,
	  IndexType OpType,
	  ResiduumType resid_type,
	  bool VerboseP,
	  bool TerminateOnResidua)
  {

	  int level = M.GetLevel();
	if( MaxIter < 0 ) {
		MasterLog(ERROR,"MR: level=%d Invalid Value: MaxIter < 0 ", level);
		QDP_abort(1);

	}

    LinearSolverResults res;
    if ( MaxIter == 0 ) {
    	// No work to do -- likely only happens in the case of a smoother
    	res.resid_type=INVALID;
    	res.n_count = 0;
    	res.resid = -1;
    	return res;
    }




    res.resid_type = resid_type;







    Spinor Mr;
    Spinor chi_internal;

    // Hack for here.
    Subset& s = QDP::all;

    Complex a;
    DComplex c;
    Double d;
    int k=0;

    chi_internal[s] = chi;
    /*  r[0]  :=  Chi - M . Psi[0] */
        /*  r  :=  M . Psi  */
        M(Mr, psi, OpType);

        Spinor r;
        r[s]= chi_internal - Mr;


    Double norm_chi_internal;
    Double rsd_sq;
    Double cp;

    if( TerminateOnResidua ) {
    	norm_chi_internal = norm2(chi_internal, s);
    	rsd_sq = Double(RsdTarget)*Double(RsdTarget);

    	if( resid_type == RELATIVE ) {
    		rsd_sq *= norm_chi_internal;
    	}

    	/*  Cp = |r[0]|^2 */
    	Double cp = norm2(r,s);                 /* 2 Nc Ns  flops */

    	if( VerboseP ) {

    		MasterLog(INFO, "MR: level=%d iter=%d || r ||^2 = %16.8e  Target || r ||^2 = %16.8e",level,k,toDouble(cp), toDouble(rsd_sq));

    	}

    	/*  IF |r[0]| <= RsdMR |Chi| THEN RETURN; */
    	if ( toBool(cp  <=  rsd_sq) )
    	{
    		res.n_count = 0;
    		res.resid   = toDouble(sqrt(cp));
    		if( resid_type == ABSOLUTE ) {
    			if( VerboseP ) {
    				MasterLog(INFO, "MR: level=%d Final iters=0 || r ||_accum=16.8e || r ||_actual = %16.8e", level,
    						toDouble(sqrt(cp)), res.resid);

    			}
    		}
    		else {

    			res.resid /= toDouble(sqrt(norm_chi_internal));
    			if( VerboseP ) {
    				MasterLog(INFO, "MR: level=%d Final iters=0 || r ||/|| b ||_accum=16.8e || r ||/|| b ||_actual = %16.8e",level,
    						toDouble(sqrt(cp/norm_chi_internal)), res.resid);
    			}
    		}

    		return res;
    	}
    }

    // TerminateOnResidua==true: if we met the residuum criterion we'd have terminated, safe to say no to terminate
    // TerminateOnResidua==false: We need to do at least 1 iteration (otherwise we'd have exited)
    bool continueP = true;

    /* Main iteration loop */
    while( continueP )
    {
      ++k;

      /*  a[k-1] := < M.r[k-1], r[k-1] >/ < M.r[k-1], M.r[k-1] > ; */
      /*  Mr = M * r  */
      M(Mr, r, OpType);
      /*  c = < M.r, r > */
      c = innerProduct(Mr, r, s);

      /*  d = | M.r | ** 2  */
      d = norm2(Mr, s);

      /*  a = c / d */
      a = c / d;

      /*  a[k-1] *= MRovpar ; */
      a = a * OmegaRelax;

      /*  Psi[k] += a[k-1] r[k-1] ; */
      psi[s] += r * a;

      /*  r[k] -= a[k-1] M . r[k-1] ; */
      r[s] -= Mr * a;


      if( TerminateOnResidua ) {

    	  /*  cp  =  | r[k] |**2 */
    	  cp = norm2(r, s);
    	  if( VerboseP ) {
    		  MasterLog(INFO, "MR: level=% iter=%d || r ||^2 = %16.8e  Target || r^2 || = %16.8e",level,
    				  k, toDouble(cp), toDouble(rsd_sq) );
    	  }
    	  continueP = (k < MaxIter) && (toBool(cp > rsd_sq));
      }
      else {
    	  if( VerboseP ) {
    		  MasterLog(INFO, "MR: level=%d iter=%d",level,k);
    	  }
    	  continueP =  (k < MaxIter);
      }

    }
    res.n_count = k;
    res.resid = 0;

    if( TerminateOnResidua) {
    	// Compute the actual residual


    	M(Mr, psi, OpType);
    	Double actual_res = norm2(chi_internal - Mr,s);
    	res.resid = toDouble(sqrt(actual_res));
		if( resid_type == ABSOLUTE ) {
			if( VerboseP ) {
				MasterLog(INFO, "MR: level=%d Final iters=%d || r ||_accum=%16.8e || r ||_actual = %16.8e",level,
				    						res.n_count, toDouble(sqrt(cp)), res.resid);
			}
		}
		else {

			res.resid /= toDouble(sqrt(norm_chi_internal));
			if( VerboseP ) {
				MasterLog(INFO, "MR: level=%d Final iters=%d || r ||_accum=%16.8e || r ||_actual = %16.8e", level,
				    						res.n_count, toDouble(sqrt(cp/norm_chi_internal)), res.resid);
			}
		}
    }
    return res;
  }


	 MRSolverQDPXX::MRSolverQDPXX(const LinearOperator<QDP::LatticeFermion,
			                        QDP::multi1d<QDP::LatticeColorMatrix> >& M,
									const MG::LinearSolverParamsBase& params) : _M(M),
	  _params(params){}

	  std::vector<LinearSolverResults>
	  MRSolverQDPXX::operator()(QDP::LatticeFermion& out, const QDP::LatticeFermion& in, ResiduumType resid_type) const {
		  return  std::vector<LinearSolverResults>(1, InvMR_a(_M, in, out, Real(_params.Omega), Real(_params.RsdTarget),
				  _params.MaxIter, LINOP_OP, resid_type, _params.VerboseP , true));

	  }



	  MRSmootherQDPXX::MRSmootherQDPXX(const LinearOperator<QDP::LatticeFermion,QDP::multi1d<QDP::LatticeColorMatrix> > & M, const MG::LinearSolverParamsBase& params) : _M(M),
	  _params(params){}

	  void
	  MRSmootherQDPXX::operator()(QDP::LatticeFermion& out, const QDP::LatticeFermion& in) const {
		  InvMR_a(_M, in, out, Real(_params.Omega), Real(_params.RsdTarget),
				  _params.MaxIter, LINOP_OP,  ABSOLUTE, _params.VerboseP , false );

	  }


}



