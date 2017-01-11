/*! \file
 *  \brief Minimal-Residual (MR) for a generic fermion Linear Operator
 */

#include "qdp.h"
#include "lattice/constants.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"

using namespace MG;
using namespace QDP;

namespace MGTesting  {

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

	if( MaxIter <= 0 ) {
		QDPIO::cerr << "MR: Invalid Value: MaxIter <= 0 " << std::endl;
		QDP_abort(1);

	}
    LinearSolverResults res; res.resid_type = resid_type;
    Spinor Mr;
    Spinor chi_internal;

    // Hack for here.
    Subset& s = all;

    Complex a;
    DComplex c;
    Double d;
    int k=0;

    if( VerboseP ) {
    	QDPIO::cout << "MR Solver Starting: " << std::endl;
    }

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
    		QDPIO::cout << "MR: iter=" << k << " || r ||^2 = " << cp << " Target || r ||^2 = " << rsd_sq << std::endl;


    	}

    	/*  IF |r[0]| <= RsdMR |Chi| THEN RETURN; */
    	if ( toBool(cp  <=  rsd_sq) )
    	{
    		res.n_count = 0;
    		res.resid   = toDouble(sqrt(cp));
    		if( resid_type == ABSOLUTE ) {
    			if( VerboseP ) {
    				QDPIO::cout << "MR: Final Absolute Residua: || r ||_accum = " << sqrt(cp) << " || r ||_actual = "
    			    		    				<< res.resid << std::endl;
    			}
    		}
    		else {

    			res.resid /= toDouble(sqrt(norm_chi_internal));
    			if( VerboseP ) {
    			QDPIO::cout << "MR: Final Residua: || r ||/|| b ||_accum = " << sqrt(cp/norm_chi_internal) << " || r || / || b ||_actual = "
    		    				<< res.resid << std::endl;
    			}
    		}

    		return res;
    	}
    }

    // TerminateOnResidua==true: if we met the residuum criterion we'd have terminated, safe to say no to terminate
    // TerminateOnResidua==false: We need to do at least 1 iteration.
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
    		  QDPIO::cout << "MR: iter=" << k << " || r ||^2 =" << cp << " Target || r ||^2 = " << rsd_sq << std::endl;
    	  }
    	  continueP = (k < MaxIter) && (toBool(cp > rsd_sq));
      }
      else {
    	  if( VerboseP ) {
    		  QDPIO::cout << "MR: iter=" << k << std::endl;

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
				QDPIO::cout << "MR: Final Absolute Residua: || r ||_accum = " << sqrt(cp) << " || r ||_actual = "
			    		    				<< res.resid << std::endl;
			}
		}
		else {

			res.resid /= toDouble(sqrt(norm_chi_internal));
			if( VerboseP ) {
			QDPIO::cout << "MR: Final Relative Residua: || r ||/|| b ||_accum = " << sqrt(cp/norm_chi_internal) << " || r || / || b ||_actual = "
		    				<< res.resid << std::endl;
			}
		}
    }
    return res;
  }

 class MRSolverParams : public MG::LinearSolverParamsBase {
 public:
	  double Omega; // OverRelaxation

  };

  template<typename Spinor, typename Gauge>
  class MRSolver : LinearSolver<Spinor,Gauge> {
  public:
	  MRSolver(const LinearOperator<Spinor,Gauge>& M, const MG::LinearSolverParamsBase& params) : _M(M),
	  _params(static_cast<const MRSolverParams&>(params)){}

	  LinearSolverResults operator()(Spinor& out, const Spinor& in, ResiduumType resid_type = RELATIVE) const {
		  return  InvMR_a(_M, in, out, Real(_params.Omega), Real(_params.RsdTarget),
				  _params.MaxIter, LINOP_OP, resid_type, _params.VerboseP , true);

	  }

  private:
	  const LinearOperator<Spinor,Gauge>& _M;
	  const MRSolverParams& _params;

  };

  template<typename Spinor, typename Gauge>
  class MRSmoother : Smoother<Spinor,Gauge> {
  public:
	  MRSmoother(const LinearOperator<Spinor,Gauge>& M, const MG::LinearSolverParamsBase& params) : _M(M),
	  _params(static_cast<const MRSolverParams&>(params)){}

	  void operator()(Spinor& out, const Spinor& in, ResiduumType resid_type = RELATIVE) const {
		  InvMR_a(_M, in, out, Real(_params.Omega), Real(_params.RsdTarget),
				  _params.MaxIter, LINOP_OP, resid_type, _params.VerboseP , false );

	  }

  private:
	  const LinearOperator<Spinor,Gauge>& _M;
	  const MRSolverParams& _params;

  };
}
