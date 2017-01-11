/*! \file
 *  \brief Conjugate-Gradient algorithm for a generic Linear Operator
 */

#include "qdp.h"
#include "lattice/constants.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"

using namespace MG;
using namespace QDP;

namespace MGTesting {

template<typename Spinor, typename Gauge>
LinearSolverResults
InvBiCGStab_a(const LinearOperator<Spinor,Gauge>& A,
	      const Spinor& chi,
	      Spinor& psi,
	      const Real& RsdTarget,
	      int MaxIter,
	      IndexType OpType,
		  ResiduumType resid_type,
		  bool VerboseP)

{
  LinearSolverResults ret;
  ret.resid_type = resid_type;
  const Subset& s = all;

  bool convP = false;	

  if( MaxIter <= 0 ) {
  		QDPIO::cerr << "BiCGStab: Invalid Value: MaxIter <= 0 " << std::endl;
  		QDP_abort(1);

  }

  ret.n_count = MaxIter;
	
  if( VerboseP ) {
	  QDPIO::cout << "BiCGStab Solver Staring: " << std::endl;
  }
  Double chi_sq =  norm2(chi,s);
  Double rsd_sq = RsdTarget*RsdTarget;

  if ( resid_type == RELATIVE ) {
	  rsd_sq *= chi_sq;
  }
  // First get r = r0 = chi - A psi
  Spinor r;
  Spinor r0;

  // Get A psi, use r0 as a temporary
  A(r0, psi, OpType);

  // now work out r= chi - Apsi = chi - r0
  r[s] = chi - r0;

    // Also copy back to r0. We are no longer in need of the
  // nth component
  r0[s] = r;
  

  // Now we have r = r0 = chi - Mpsi
  // Check if solution is already good enough
  Double r_norm = norm2(r,s);
  if( VerboseP ) {
	  QDPIO::cout << "BiCGStab: iter=0" << " || r ||^2=" << r_norm
			  << " Target || r ||^2=" << rsd_sq << std::endl;

  }

  if ( toBool(r_norm  <=  rsd_sq) )
  {
	  ret.n_count = 0;
	  ret.resid   = toDouble(sqrt(r_norm));
	  if ( resid_type == ABSOLUTE ) {
		  if( VerboseP ) {
			  QDPIO::cout << "BiCGStab: Final Absolute Residua: || r ||_accum = " << sqrt(r_norm)
  	  		  										<< " || r ||_actual = "
													<< ret.resid << std::endl;
		  }
	  }
	  else {
		  ret.resid /= toDouble(sqrt(chi_sq));
		  if( VerboseP ) {
			  QDPIO::cout << "BiCGStab: Final Relative Residua: || r ||/|| b ||_accum = " << sqrt(r_norm/chi_sq)
  		  								<< " || r ||/|| b ||_actual = "
										<< ret.resid << std::endl;
		  }
	  }
	  return ret;
  }



  // Now initialise v = p = 0
  Spinor p;
  Spinor v;

  p[s] = zero;
  v[s] = zero;

  Spinor tmp;
  Spinor t;

  ComplexD rho, rho_prev, alpha, omega;

  // rho_0 := alpha := omega = 1
  // Iterations start at k=1, so rho_0 is in rho_prev
  rho_prev = Double(1);
  alpha = Double(1);
  omega = Double(1);

  // The iterations 
  for(int k = 1; k <= MaxIter && !convP ; k++) {
    
    // rho_{k+1} = < r_0 | r >
    rho = innerProduct(r0,r,s);


    if( toBool( real(rho) == 0 ) && toBool( imag(rho) == 0 ) ) {
      QDPIO::cout << "BiCGStab breakdown: rho = 0" << std::endl;
      QDP_abort(1);
    }

    // beta = ( rho_{k+1}/rho_{k})(alpha/omega)
    ComplexD beta;
    beta = ( rho / rho_prev ) * (alpha/omega);
    
    // p = r + beta(p - omega v)

    // first work out p - omega v 
    // into tmp
    // then do p = r + beta tmp
    Complex omega_r = omega;
    Complex beta_r = beta;
    tmp[s] = p - omega_r*v;
    p[s] = r + beta_r*tmp;


    // v = Ap
    A(v,p,OpType);


    // alpha = rho_{k+1} / < r_0 | v >
    // put <r_0 | v > into tmp
    DComplex ctmp = innerProduct(r0,v,s);


    if( toBool( real(ctmp) == 0 ) && toBool( imag(ctmp) == 0 ) ) {
      QDPIO::cout << "BiCGStab breakdown: <r_0|v> = 0" << std::endl;
      QDP_abort(1);
    }

    alpha = rho / ctmp;

    // Done with rho now, so save it into rho_prev
    rho_prev = rho;

    // s = r - alpha v
    // I can overlap s with r, because I recompute it at the end.
    Complex alpha_r = alpha;
    r[s]  -=  alpha_r*v;


    // t = As  = Ar 
    A(t,r,OpType);
    // omega = < t | s > / < t | t > = < t | r > / norm2(t);

    // This does the full 5D norm
    Double t_norm = norm2(t,s);


    if( toBool(t_norm == 0) ) { 
      QDPIO::cerr << "Breakdown || Ms || = || t || = 0 " << std::endl;
      QDP_abort(1);
    }

    // accumulate <t | s > = <t | r> into omega
    omega = innerProduct(t,r,s);
    omega /= t_norm;

    // psi = psi + omega s + alpha p 
    //     = psi + omega r + alpha p
    //
    // use tmp to compute psi + omega r
    // then add in the alpha p
    omega_r = omega;
    alpha_r = alpha;
    tmp[s] = psi + omega_r*r;   
    psi[s] = tmp + alpha_r*p;



    // r = s - omega t = r - omega t1G

    
    r[s] -= omega_r*t;


    r_norm = norm2(r,s);
    if( VerboseP ) {
     	  QDPIO::cout << "BiCGStab: iter=" << k << " || r ||^2=" << r_norm
     			  	  << " Target || r ||^2=" << rsd_sq<< std::endl;

       }


    //    QDPIO::cout << "Iteration " << k << " : r = " << r_norm << std::endl;
    if( toBool(r_norm < rsd_sq ) ) {
      convP = true;
      ret.resid = toDouble(sqrt(r_norm/chi_sq));
      ret.n_count = k;

    }
    else { 
      convP = false;
    }


  }
  
	// Compute the actual residual


  	A(r, psi, OpType);
  	Double actual_res = norm2(chi - r,s);
  	ret.resid = toDouble(sqrt(actual_res));
  	if ( resid_type == ABSOLUTE ) {
  		if( VerboseP ) {
  		QDPIO::cout << "BiCGStab: Final Absolute Residua: || r ||_accum = " << sqrt(r_norm)
  		  						<< " || r ||_actual = "
  								<< ret.resid << std::endl;
  		}
  	}
  	else {

  		ret.resid /= toDouble(sqrt(chi_sq));
  		if( VerboseP ) {
  			QDPIO::cout << "BiCGStab: Final Relative Residua: || r ||/|| b ||_accum = " << sqrt(r_norm/chi_sq)
  						<< " || r ||/|| b ||_actual = "
						<< ret.resid << std::endl;
  		}
  	}

  return ret;
}

template<typename Spinor, typename Gauge>
 class BiCGStabSolver : public LinearSolver<Spinor,Gauge> {
 public:
	 	 BiCGStabSolver(const LinearOperator<Spinor,Gauge>& M, const LinearSolverParamsBase& params) : _M(M),
	  _params(params){}

	  LinearSolverResults operator()(Spinor& out, const Spinor& in, ResiduumType resid_type = RELATIVE ) const {
		  return  InvBiCGStab_a(_M, in, out, _params.RsdTarget, _params.MaxIter, LINOP_OP, resid_type, _params.VerboseP ) ;

	  }

 private:
	  const LinearOperator<Spinor,Gauge>& _M;
	  const LinearSolverParamsBase& _params;

 };

}  // end namespace MGTEsting
