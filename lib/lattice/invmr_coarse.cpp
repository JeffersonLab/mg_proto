/*
 * * invmr_coarse.cpp
 *
 *  Created on: Mar 21, 2017
 *      Author: bjoo
 */

#include <lattice/coarse/invmr_coarse.h>
#include "lattice/constants.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"
#include "lattice/mr_params.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_l1_blas.h"

#include "utils/print_utils.h"

#include <complex>

namespace MG
{


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


LinearSolverResults
InvMR_T(const LinearOperator<CoarseSpinor,CoarseGauge>& M,
		const CoarseSpinor& chi,
		CoarseSpinor& psi,
		const double& OmegaRelax,
		const double& RsdTarget,
		int MaxIter,
		IndexType OpType,
		ResiduumType resid_type,
		bool VerboseP,
		bool TerminateOnResidua)
{
	const int level = M.GetLevel();

	const LatticeInfo& info = chi.GetInfo();
	{
		const LatticeInfo& M_info = M.GetInfo();
		AssertCompatible( M_info, info );
		const LatticeInfo& psi_info = psi.GetInfo();
		AssertCompatible( psi_info, info );
	}

	if( MaxIter < 0 ) {
		MasterLog(ERROR,"MR: level=%d Invalid Value: MaxIter < 0 ",level);
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

	CoarseSpinor Mr(info);
	CoarseSpinor chi_internal(info);

	std::complex<double> a;
	std::complex<double> c;
	double d;
	int k=0;


	// chi_internal[s] = chi;
	CopyVec(chi_internal, chi);

	/*  r[0]  :=  Chi - M . Psi[0] */
	/*  r  :=  M . Psi  */
	M(Mr, psi, OpType);

	CoarseSpinor r(info);
	// r[s]= chi_internal - Mr;
	XmyzVec(chi_internal,Mr,r);


	double norm_chi_internal;
	double rsd_sq;
	double cp;

	if( TerminateOnResidua ) {
		norm_chi_internal = Norm2Vec(chi_internal);
		rsd_sq = RsdTarget*RsdTarget;

		if( resid_type == RELATIVE ) {
			rsd_sq *= norm_chi_internal;
		}

		/*  Cp = |r[0]|^2 */
		double cp = Norm2Vec(r);                 /* 2 Nc Ns  flops */

		if( VerboseP ) {

			MasterLog(INFO, "MR: level=%d iter=%d || r ||^2 = %16.8e  Target || r ||^2 = %16.8e",level,k,cp, rsd_sq);

		}

		/*  IF |r[0]| <= RsdMR |Chi| THEN RETURN; */
		if ( cp  <=  rsd_sq )
		{
			res.n_count = 0;
			res.resid   = sqrt(cp);
			if( resid_type == ABSOLUTE ) {
				if( VerboseP ) {
					MasterLog(INFO, "MR Solver: level=%d Final iters=0 || r ||_accum=16.8e || r ||_actual = %16.8e",level,
							sqrt(cp), res.resid);

				}
			}
			else {

				res.resid /= sqrt(norm_chi_internal);
				if( VerboseP ) {
					MasterLog(INFO, "MR: level=%d Final iters=0 || r ||/|| b ||_accum=16.8e || r ||/|| b ||_actual = %16.8e",level,
							sqrt(cp/norm_chi_internal), res.resid);
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
		c = InnerProductVec(Mr, r);

		/*  d = | M.r | ** 2  */
		d = Norm2Vec(Mr);

		/*  a = c / d */
		a = c / d;

		/*  a[k-1] *= MRovpar ; */
		a = a * OmegaRelax;

		/*  Psi[k] += a[k-1] r[k-1] ; */
		//psi[s] += r * a;
		std::complex<float> af( (float)a.real(), (float)a.imag() );
		AxpyVec(af,r,psi);

		/*  r[k] -= a[k-1] M . r[k-1] ; */
		// r[s] -= Mr * a;
		std::complex<float> maf(-af.real(), -af.imag());
		AxpyVec(maf,Mr,r);


		if( TerminateOnResidua ) {

			/*  cp  =  | r[k] |**2 */
			cp = Norm2Vec(r);
			if( VerboseP ) {
				MasterLog(INFO, "MR: level=%d iter=%d || r ||^2 = %16.8e  Target || r^2 || = %16.8e", level,
						k, cp, rsd_sq );
			}
			continueP = (k < MaxIter) && (cp > rsd_sq);
		}
		else {
			if( VerboseP ) {
				MasterLog(INFO, "MR: level=%d iter=%d",level, k);
			}
			continueP =  (k < MaxIter);
		}

	}
	res.n_count = k;
	res.resid = 0;

	if( TerminateOnResidua) {
		// Compute the actual residual


		M(Mr, psi, OpType);
		//Double actual_res = norm2(chi_internal - Mr,s);
		double actual_res = XmyNorm2Vec(chi_internal,Mr);
		res.resid = sqrt(actual_res);
		if( resid_type == ABSOLUTE ) {
			if( VerboseP ) {
				MasterLog(INFO, "MR: level=%d Final iters=%d || r ||_accum=%16.8e || r ||_actual=%16.8e", level,
						res.n_count, sqrt(cp), res.resid);
			}
		}
		else {

			res.resid /= sqrt(norm_chi_internal);
			if( VerboseP ) {
				MasterLog(INFO, "MR: level=%d Final iters=%d || r ||_accum=%16.8e || r ||_actual=%16.8e", level,
						res.n_count, sqrt(cp/norm_chi_internal), res.resid);
			}
		}
	}
	return res;
}




MRSolverCoarse::MRSolverCoarse(const LinearOperator<CoarseSpinor,CoarseGauge>& M,
		const MG::LinearSolverParamsBase& params) : _M(M),
				_params(static_cast<const MRSolverParams&>(params)){}

LinearSolverResults
MRSolverCoarse::operator()(CoarseSpinor& out,
		const CoarseSpinor& in,
		ResiduumType resid_type) const {
	return  InvMR_T(_M, in, out, _params.Omega, _params.RsdTarget,
			_params.MaxIter, LINOP_OP, resid_type, _params.VerboseP , true);

}


MRSmootherCoarse::MRSmootherCoarse(const LinearOperator<CoarseSpinor,CoarseGauge>& M,
		const MG::LinearSolverParamsBase& params) : _M(M),
				_params(static_cast<const MRSolverParams&>(params)){}

void
MRSmootherCoarse::operator()(CoarseSpinor& out, const CoarseSpinor& in) const {
	InvMR_T(_M, in, out, _params.Omega, _params.RsdTarget,
			_params.MaxIter, LINOP_OP,  ABSOLUTE, _params.VerboseP , false );
}

}; // Namespace




