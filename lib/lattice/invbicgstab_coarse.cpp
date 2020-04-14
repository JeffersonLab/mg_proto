/*
 * invbicgstab_coarse.cpp
 *
 *  Created on: Mar 21, 2017
 *      Author: bjoo
 */

#include <complex>
#include <algorithm>
#include <lattice/coarse/invbicgstab_coarse.h>
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_l1_blas.h"
#include "utils/print_utils.h"



namespace MG {

namespace {
	/** all
	 *
	 * 	Returns true if all elements are true
	 */
	bool all(const std::vector<bool>& x) {
		return std::all_of(x.begin(), x.end(), [](bool b){return b;});
	}

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
		std::transform(x.begin(), x.end(), r.begin(), [](const std::complex<T>& f) -> std::complex<T> { return -f;});
		return r;
	}

}

std::vector<LinearSolverResults>
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
	const CBSubset& subset = A.GetSubset();
	IndexType ncol = psi.GetNCol();

	{
		const LatticeInfo& chi_info = chi.GetInfo();
		const LatticeInfo& psi_info = psi.GetInfo();
		AssertCompatible( info, chi_info);
		AssertCompatible( info, psi_info);
	}

	std::vector<LinearSolverResults> ret(ncol);
	for (int col=0; col < ncol; ++col) ret[col].resid_type = resid_type;

	if( MaxIter <= 0 ) {
		MasterLog(ERROR,"BiCGStab: level=%d Invalid Value: MaxIter <= 0 ",level);
	}

	for (int col=0; col < ncol; ++col) ret[col].n_count = MaxIter;

	if( VerboseP ) {
		MasterLog(INFO, "BiCGStab: level=%d Solver Staring: ", level);
	}

	std::vector<double> chi_sq =  Norm2Vec(chi,subset);
	std::vector<double> rsd_sq(ncol, RsdTarget*RsdTarget);

	if ( resid_type == RELATIVE ) {
		for (int col=0; col < ncol; ++col) rsd_sq[col] *= chi_sq[col];
	}
	// First get r = r0 = chi - A psi
	CoarseSpinor r(info, ncol);
	CoarseSpinor r0(info, ncol);

	// Get A psi, use r0 as a temporary
	A(r0, psi, OpType);

	// now work out r= chi - Apsi = chi - r0
	//r[s] = chi - r0;
	XmyzVec(chi,r0,r,subset);

	// Also copy back to r0. We are no longer in need of the
	// nth component
	//r0[s] = r;
	CopyVec(r0,r,subset);

	// Now we have r = r0 = chi - Mpsi
	// Check if solution is already good enough
	// Double r_norm = norm2(r,s);
	std::vector<double> r_norm = Norm2Vec(r,subset);

	// Flags the converged columns
	std::vector<bool> convP(ncol, false);

	for (int col=0; col < ncol; ++col) {
		if( VerboseP ) {
			MasterLog(INFO,"BiCGStab: col=%d level=%d iter=0 || r ||^2=%16.8e Target || r ||^2=%16.8e",col,level,r_norm[col],rsd_sq[col]);
		}

		if ( r_norm[col]  <=  rsd_sq[col] )
		{
			ret[col].n_count = 0;
			ret[col].resid   = std::sqrt(r_norm[col]);
			if ( resid_type == ABSOLUTE ) {
				if( VerboseP ) {
					MasterLog(INFO,"BiCGStab: col=%d level=%d Final Absolute Residua: || r ||_accum = %16.8e  || r ||_actual = %16.8e",
							col,level,sqrt(r_norm[col]), ret[col].resid);
				}
			}
			else {
				ret[col].resid /= sqrt(chi_sq[col]);
				if( VerboseP ) {
					MasterLog(INFO, "BiCGStab: col=%d level=%d Final Relative Residua: || r ||/|| b ||_accum = %16.8e || r ||/|| b ||_actual = %16.8e ",
							col,level,sqrt(r_norm[col]/chi_sq[col]),ret[col].resid);
				}
			}
			convP[col] = true;
		}
	}

	if (all(convP)) {
		return ret;
	}

	// Now initialise v = p = 0
	CoarseSpinor p(info, ncol);  ZeroVec(p,subset);
	CoarseSpinor v(info, ncol);  ZeroVec(v,subset);

	CoarseSpinor tmp(info, ncol);
	CoarseSpinor t(info, ncol);

	// rho_0 := alpha := omega = 1
	// Iterations start at k=1, so rho_0 is in rho_prev
	std::vector<DComplex> rho(ncol), rho_prev(ncol, 1.0), alpha(ncol, 1.0), omega(ncol, 1.0);

	// The iterations
	for(int k = 1; k <= MaxIter && !all(convP) ; k++) {

		// rho_{k+1} = < r_0 | r >
		rho = InnerProductVec(r0,r,subset);

		for (int col=0; col < ncol; ++col) {
			if( std::abs(rho[col]) == 0) {
				MasterLog(ERROR, "BiCGStab: col=%d level=%d breakdown: rho = 0", col, level);
				convP[col] = true;
			}
		}

		// beta = ( rho_{k+1}/rho_{k})(alpha/omega)
		std::vector<FComplex> beta(ncol);
		for (int col=0; col < ncol; ++col) beta[col] = ( rho[col] / rho_prev[col] ) * (alpha[col]/omega[col]);

		// So beta in initial iter is 1:
		// p = r + beta(p - omega v)

		// so in iter 0, p should equal r.


		// first work out p - omega v (initial iter v=0 & p = 0, so p-omega v = 0
		// into tmp
		// then do p = r + beta tmp
		//  tmp[s] = p - omega*v;
		//  p[s] = r + beta*tmp;
		BiCGStabPUpdate(beta,r,castToFloat(omega), v, p, subset);



		// v = Ap
		A(v,p,OpType);



		// alpha = rho_{k+1} / < r_0 | v >
		// put <r_0 | v > into tmp
		std::vector<DComplex> ctmp = InnerProductVec(r0,v,subset);

		for (int col=0; col < ncol; ++col) {
			if( std::abs(ctmp[col]) == 0) {
				MasterLog(ERROR, "BiCGStab: col=%d level=%d breakdown: <r_0|v> = 0", col, level);
				convP[col] = true;
			}
		}


		for (int col=0; col < ncol; ++col) alpha[col] = rho[col] / ctmp[col];

		// Done with rho now, so save it into rho_prev
		rho_prev = rho;

		// s = r - alpha v
		// I can overlap s with r, because I recompute it at the end.

		//r[s]  -=  alpha_r*v;
		AxpyVec(negate(alpha),v,r,subset);



		// t = As  = Ar
		A(t,r,OpType);

		// omega = < t | s > / < t | t > = < t | r > / norm2(t);

		// This does the full 5D norm
		std::vector<double> t_norm = Norm2Vec(t,subset);

		for (int col=0; col < ncol; ++col) {
			if( std::abs(t_norm[col]) == 0) {
				MasterLog(ERROR, "BiCGStab: col=%d level=%d Breakdown || Ms || = || t || = 0", col, level);
				convP[col] = true;
			}
		}

		// accumulate <t | s > = <t | r> into omega
		omega = InnerProductVec(t,r,subset);
		for (int col=0; col < ncol; ++col) omega[col] /= t_norm[col];

		// psi = psi + omega s + alpha p
		//     = psi + omega r + alpha p
		//
		// use tmp to compute psi + omega r
		// then add in the alpha p

		//tmp[s] = psi + omega*r;
		//psi[s] = tmp + alpha*p;
		//psi[s] = psi + omega*r + alpha*p
		BiCGStabXUpdate(castToFloat(omega), r, castToFloat(alpha), p, psi,subset);

		// r = s - omega t = r - omega t1G

		AxpyVec( negate(omega), t, r,subset);

		r_norm = Norm2Vec(r,subset);
		if( VerboseP ) {
			for (int col=0; col < ncol; ++col) {
				 MasterLog(INFO,"BiCGStab: col=%d level=%d iter=%d || r ||^2=%16.8e  Target || r ||^2=%16.8e", col, level, k, r_norm[col], rsd_sq[col]);
			}
		}


		//    QDPIO::cout << "Iteration " << k << " : r = " << r_norm << std::endl;
		for (int col=0; col < ncol; ++col) {
			if( !convP[col] && r_norm[col] < rsd_sq[col]  ) {
				convP[col] = true;
				ret[col].resid = sqrt(r_norm[col]/chi_sq[col]);
				ret[col].n_count = k;
			}
		}

	}

	// Compute the actual residual
	A(r, psi, OpType);
	std::vector<double> actual_res = XmyNorm2Vec(r,chi,subset);
	for (int col=0; col < ncol; ++col) {
		ret[col].resid = sqrt(actual_res[col]);
		if ( resid_type == ABSOLUTE ) {
			if( VerboseP ) {
				MasterLog(INFO,"BiCGStab: col=%d level=%d Final Absolute Residua: || r ||_accum = %16.8e || r ||_actual = %16.8e",col,level,
						sqrt(r_norm[col]), ret[col].resid);
			}
		}
		else {

			ret[col].resid /= sqrt(chi_sq[col]);
			if( VerboseP ) {
				MasterLog(INFO,"BiCGStab: col=%d level=%d Final Relative Residua: || r ||/|| b ||_accum = %16.8e || r ||/|| b ||_actual = %16.8e", col, level, sqrt(r_norm[col]/chi_sq[col]),ret[col].resid);
			}
		}
	}

	return ret;
}


BiCGStabSolverCoarse::BiCGStabSolverCoarse(const LinearOperator<CoarseSpinor,CoarseGauge>& M, const LinearSolverParamsBase& params) : _M(M),
		_params(params) {}
BiCGStabSolverCoarse::BiCGStabSolverCoarse( const std::shared_ptr<const LinearOperator<CoarseSpinor,CoarseGauge>> M,
		const LinearSolverParamsBase& params) : _M(*M),	_params(params) {}
std::vector<LinearSolverResults>
BiCGStabSolverCoarse::operator()(CoarseSpinor& out, const CoarseSpinor& in, ResiduumType resid_type  ) const {
	return  InvBiCGStabCoarse_a(_M, in, out, _params.RsdTarget, _params.MaxIter, LINOP_OP, resid_type, _params.VerboseP ) ;

}


}  // end namespace MGTEsting




