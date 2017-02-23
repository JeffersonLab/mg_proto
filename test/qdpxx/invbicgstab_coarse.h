/*! \file
 *  \brief Conjugate-Gradient algorithm for a generic Linear Operator
 */

#ifndef TEST_QDPXX_INVBICGSTAB_COARSE_H_
#define TEST_QDPXX_INVBICGSTAB_COARSE_H_

#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_l1_blas.h"
#include "utils/print_utils.h"

using namespace MG;


namespace MGTesting {

LinearSolverResults
InvBiCGStabCoarse_a(const LinearOperator<CoarseSpinor,CoarseGauge>& A,
	      const CoarseSpinor& chi,
	      CoarseSpinor& psi,
	      const double& RsdTarget,
	      int MaxIter,
	      IndexType OpType,
		  ResiduumType resid_type,
		  bool VerboseP)

{

  using DComplex = std::complex<double>;
  using FComplex = std::complex<float>;
  const int level = A.GetLevel();
  const LatticeInfo& info = A.GetInfo();

  {
  const LatticeInfo& chi_info = chi.GetInfo();
  const LatticeInfo& psi_info = psi.GetInfo();
  AssertCompatible( info, chi_info);
  AssertCompatible( info, psi_info);
  }

  LinearSolverResults ret;
  ret.resid_type = resid_type;

  bool convP = false;	

  if( MaxIter <= 0 ) {
  		MasterLog(ERROR,"BiCGStab: level=%d Invalid Value: MaxIter <= 0 ",level);
  }

  ret.n_count = MaxIter;
	
  if( VerboseP ) {
	 MasterLog(INFO, "BiCGStab: level=%d Solver Staring: ", level);
  }

  double chi_sq =  Norm2Vec(chi);
  double rsd_sq = RsdTarget*RsdTarget;

  if ( resid_type == RELATIVE ) {
	  rsd_sq *= chi_sq;
  }
  // First get r = r0 = chi - A psi
  CoarseSpinor r(info);
  CoarseSpinor r0(info);

  // Get A psi, use r0 as a temporary
  A(r0, psi, OpType);

  // now work out r= chi - Apsi = chi - r0
  //r[s] = chi - r0;
  XmyzVec(chi,r0,r);

    // Also copy back to r0. We are no longer in need of the
  // nth component
  //r0[s] = r;
  CopyVec(r0,r);

  // Now we have r = r0 = chi - Mpsi
  // Check if solution is already good enough
  // Double r_norm = norm2(r,s);
  double r_norm = Norm2Vec(r);

  if( VerboseP ) {
	  MasterLog(INFO,"BiCGStab: level=%d iter=0 || r ||^2=%16.8e Target || r ||^2=%16.8e",level,r_norm,rsd_sq);
  }

  if ( r_norm  <=  rsd_sq )
  {
	  ret.n_count = 0;
	  ret.resid   = sqrt(r_norm);
	  if ( resid_type == ABSOLUTE ) {
		  if( VerboseP ) {
			  MasterLog(INFO,"BiCGStab: level=%d Final Absolute Residua: || r ||_accum = %16.8e  || r ||_actual = %16.8e",
													level,sqrt(r_norm), ret.resid);
		  }
	  }
	  else {
		  ret.resid /= sqrt(chi_sq);
		  if( VerboseP ) {
			  MasterLog(INFO, "BiCGStab: level=%d Final Relative Residua: || r ||/|| b ||_accum = %16.8e || r ||/|| b ||_actual = %16.8e ",
										 level,sqrt(r_norm/chi_sq),ret.resid);
		  }
	  }
	  return ret;
  }



  // Now initialise v = p = 0
  CoarseSpinor p(info);  ZeroVec(p);
  CoarseSpinor v(info);	 ZeroVec(v);

  CoarseSpinor tmp(info);
  CoarseSpinor t(info);

  DComplex rho, rho_prev, alpha, omega;

  // rho_0 := alpha := omega = 1
  // Iterations start at k=1, so rho_0 is in rho_prev
  rho_prev = DComplex((double)1,(double)0);
  alpha = DComplex((double)1,(double)0);
  omega = DComplex((double)1,(double)0);

  // The iterations 
  for(int k = 1; k <= MaxIter && !convP ; k++) {
    
    // rho_{k+1} = < r_0 | r >
    rho = InnerProductVec(r0,r);


    if( real(rho) == 0  &&  imag(rho) == 0  ) {
      MasterLog(ERROR, "BiCGStab: level=%d breakdown: rho = 0", level);
    }

    // beta = ( rho_{k+1}/rho_{k})(alpha/omega)
    DComplex beta;
    beta = ( rho / rho_prev ) * (alpha/omega);
    
    // p = r + beta(p - omega v)

    // first work out p - omega v 
    // into tmp
    // then do p = r + beta tmp
    FComplex omega_r( (float)omega.real(), (float)omega.imag());
    FComplex beta_r( (float)beta.real(), (float)beta.imag());
  //  tmp[s] = p - omega_r*v;
  //  p[s] = r + beta_r*tmp;
   BiCGStabPUpdate(beta_r,r,omega_r, v, p);



    // v = Ap
    A(v,p,OpType);


    // alpha = rho_{k+1} / < r_0 | v >
    // put <r_0 | v > into tmp
    DComplex ctmp = InnerProductVec(r0,v);


    if( real(ctmp) == 0  &&  imag(ctmp) == 0  ) {
      MasterLog(ERROR,"BiCGStab: level=%d breakdown: <r_0|v> = 0",level);
    }

    alpha = rho / ctmp;

    // Done with rho now, so save it into rho_prev
    rho_prev = rho;

    // s = r - alpha v
    // I can overlap s with r, because I recompute it at the end.
    FComplex malpha_r(-(float)alpha.real(), -(float)alpha.imag());

    //r[s]  -=  alpha_r*v;
    AxpyVec(malpha_r,v,r);

    // t = As  = Ar 
    A(t,r,OpType);
    // omega = < t | s > / < t | t > = < t | r > / norm2(t);

    // This does the full 5D norm
    double t_norm = Norm2Vec(t);


    if( t_norm == 0 ) {
      MasterLog(ERROR, "BiCGStab: level=%d Breakdown || Ms || = || t || = 0 ",level);
    }

    // accumulate <t | s > = <t | r> into omega
    omega = InnerProductVec(t,r);
    omega /= t_norm;

    // psi = psi + omega s + alpha p 
    //     = psi + omega r + alpha p
    //
    // use tmp to compute psi + omega r
    // then add in the alpha p
    omega_r = omega;
    FComplex alpha_r((float)alpha.real(),(float)alpha.imag());

    //tmp[s] = psi + omega_r*r;
    //psi[s] = tmp + alpha_r*p;
    //psi[s] = psi + omega_r*r + alpha_r*p
    BiCGStabXUpdate(omega_r, r, alpha_r, p, psi);

    // r = s - omega t = r - omega t1G

    FComplex momega_r(-(float)omega.real(), -(float)omega.imag());
    AxpyVec( momega_r, t, r);

    r_norm = Norm2Vec(r);
    if( VerboseP ) {
     	  MasterLog(INFO,"BiCGStab: level=%d iter=%d || r ||^2=%16.8e  Target || r ||^2=%16.8e",level, k, r_norm, rsd_sq);

       }


    //    QDPIO::cout << "Iteration " << k << " : r = " << r_norm << std::endl;
    if( r_norm < rsd_sq  ) {
      convP = true;
      ret.resid = sqrt(r_norm/chi_sq);
      ret.n_count = k;

    }
    else { 
      convP = false;
    }


  }
  
	// Compute the actual residual
  	A(r, psi, OpType);
  	double actual_res = XmyNorm2Vec(r,chi);
  	ret.resid = sqrt(actual_res);
  	if ( resid_type == ABSOLUTE ) {
  		if( VerboseP ) {
  			MasterLog(INFO,"BiCGStab: level=%d Final Absolute Residua: || r ||_accum = %16.8e || r ||_actual = %16.8e",level,
  					sqrt(r_norm), ret.resid);
  		}
  	}
  	else {

  		ret.resid /= sqrt(chi_sq);
  		if( VerboseP ) {
  			MasterLog(INFO,"BiCGStab: level=%d Final Relative Residua: || r ||/|| b ||_accum = %16.8e || r ||/|| b ||_actual = %16.8e", level, sqrt(r_norm/chi_sq),ret.resid);
  		}
  	}

  return ret;
}


class BiCGStabSolverCoarse : public LinearSolver<CoarseSpinor,CoarseGauge> {
public:

	BiCGStabSolverCoarse(const LinearOperator<CoarseSpinor,CoarseGauge>& M, const LinearSolverParamsBase& params) : _M(M),
	  _params(params){}

	  LinearSolverResults operator()(CoarseSpinor& out, const CoarseSpinor& in, ResiduumType resid_type = RELATIVE ) const {
		  return  InvBiCGStabCoarse_a(_M, in, out, _params.RsdTarget, _params.MaxIter, LINOP_OP, resid_type, _params.VerboseP ) ;

	  }

 private:
	  const LinearOperator<CoarseSpinor,CoarseGauge>& _M;
	  const LinearSolverParamsBase& _params;

 };

}  // end namespace MGTEsting

#endif /* TEST_QDPXX_INVBICGSTAB_H_ */

