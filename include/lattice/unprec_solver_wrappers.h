/*
 * unprec_solver_wrapper.h
 *
 *  Created on: Aug 13, 2018
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_UNPREC_WRAPPER_H_
#define INCLUDE_LATTICE_UNPREC_WRAPPER_H_

#include "lattice/solver.h"
#include "lattice/linear_operator.h"
#include <memory>

namespace MG
{
	template<typename Spinor, typename Gauge, typename EOSolver>
	class UnprecLinearSolverWrapper: public  UnprecLinearSolver<Spinor,Gauge,EOSolver> {
	public:
		UnprecLinearSolverWrapper(const std::shared_ptr<const EOSolver>& eo_solver,
					  const std::shared_ptr<const EOLinearOperator<Spinor,Gauge>>& M)
		: _EOSolver(eo_solver), _EOOperator(M) {}

		UnprecLinearSolverWrapper( const std::shared_ptr<const EOLinearOperator<Spinor,Gauge>> M,
				const LinearSolverParamsBase& params) :
			_EOSolver(std::make_shared<EOSolver>(*M,params)), _EOOperator(M) {}

		UnprecLinearSolverWrapper( const std::shared_ptr<const EOLinearOperator<Spinor,Gauge>> M,
				const LinearSolverParamsBase& params,
				const LinearSolver<Spinor,Gauge>* M_prec) :
					_EOSolver(std::make_shared<EOSolver>(*M,params,M_prec)), _EOOperator(M) {}

		void SourcePrepare(Spinor& new_source, const Spinor& original_source) const override
		{
			_EOOperator->leftInvOp(new_source,original_source);
		}

		void ResultReconstruct(Spinor& new_result, const Spinor& original_result) const override
		{
			_EOOperator->rightInvOp(new_result,original_result);

		}
		void InitGuessPrepare(Spinor& new_guess, const Spinor& original_guess) const override
		{
			_EOOperator->rightOp(new_guess,original_guess);
		}
		void OtherSubsetSolve(Spinor& new_guess, const Spinor& original_guess) const override
		{
			_EOOperator->M_ee_inv(new_guess, original_guess, LINOP_OP);
		}

		const EOSolver& GetEOSolver() const override {
			return *_EOSolver;
		}
		
#if 0
		LinearSolverResults operator()(Spinor& out, const Spinor& in, ResiduumType resid_type = RELATIVE) const override {
				LinearSolverResults ret_val;

			// Prepare the source: L^{-1} in
			// In principle, this may change both even and odd parts,
			// Depending on the preconditioning style.
			// So worth preserving the prepped source.
			SourcePrepare(tmp_src,in);

			// Solve odd part with Krylov solver
			// Zero out the Even part of tmp_src for this
			// It is assumed that the solver will not touch the EVEN part.
			InitGuessPrepare(tmp_out,out);

			ret_val = (*_EOSolver)(tmp_out, tmp_src, resid_type);

			OtherSubsetSolve(tmp_out,tmp_src);

			// Reconstruct the result
			ResultReconstruct(out,tmp_out);
			return ret_val;

		}
#endif
	private:

		const std::shared_ptr<const EOSolver> _EOSolver;
		const std::shared_ptr<const EOLinearOperator<Spinor,Gauge>> _EOOperator;
	};

	template<typename Spinor, typename Gauge,  typename EOSmoother>
	class UnprecSmootherWrapper: public  UnprecSmoother<Spinor,Gauge, EOSmoother> {
	public:
		UnprecSmootherWrapper(const std::shared_ptr<const EOSmoother>& eo_smoother,
				const std::shared_ptr<const EOLinearOperator<Spinor,Gauge>>& linop) :
					_EOSmoother(eo_smoother), _EOOperator(linop) {}

		UnprecSmootherWrapper( const std::shared_ptr<const EOLinearOperator<Spinor,Gauge>> M,
				const LinearSolverParamsBase& params) :
					_EOSmoother(std::make_shared<EOSmoother>(*M,params)), _EOOperator(M) {}

		UnprecSmootherWrapper( const std::shared_ptr<const EOLinearOperator<Spinor,Gauge>> M,
				const LinearSolverParamsBase& params,
				const LinearSolver<Spinor,Gauge>* M_prec) :
					_EOSmoother(std::make_shared<EOSmoother>(*M,params,M_prec)), _EOOperator(M) {}

		void SourcePrepare(Spinor& new_source, const Spinor& original_source) const override
		{
			_EOOperator->leftInvOp(new_source,original_source);
		}

		void ResultReconstruct(Spinor& new_result, const Spinor& original_result) const override
		{
			_EOOperator->rightInvOp(new_result,original_result);

		}

		void InitGuessPrepare(Spinor& new_guess, const Spinor& original_guess) const override
		{
			_EOOperator->rightOp(new_guess,original_guess);
		}
		void OtherSubsetSolve(Spinor& new_guess, const Spinor& original_guess) const override
		{
			_EOOperator->M_ee_inv(new_guess, original_guess, LINOP_OP);
		}

		const EOSmoother& GetEOSmoother() const override {
			return *_EOSmoother;
		}
#if 0
		void operator()(Spinor& out, const Spinor& in) const override {

			MasterLog(INFO, "Before source prepare: Spinor Out has norm =%lf Spinor In has norm %lf", Norm2Vec(out), Norm2Vec(in));

			// Prepare the source: L^{-1} in
			// In principle, this may change both even and odd parts,
			// Depending on the preconditioning style.
			// So worth preserving the prepped source.
			SourcePrepare(tmp_src,in);

			MasterLog(INFO, "After source prepare: tmp_src has norm =%16.8e  In has norm %16.8e", Norm2Vec(tmp_src), Norm2Vec(in));

			// Solve odd part with Krylov solver
			// Use odd part of initial guess (or later use R_op to convert existing initial guess)
			CopyVec(tmp_out,out,SUBSET_ODD);

			(*_EOSmoother)(tmp_out, tmp_src);
			MasterLog(INFO, "After smoother: tmp_out has norm =%lf  tmp_src has norm %lf", Norm2Vec(tmp_out), Norm2Vec(tmp_src));

			// Solve even part directly
			_EOOperator->M_ee_inv(tmp_out, tmp_src, LINOP_OP);
			MasterLog(INFO, "After M_ee_inv tmp_out has norm =%16.8e  tmp_src has norm %16.8e", Norm2Vec(tmp_out, SUBSET_EVEN), Norm2Vec(tmp_src, SUBSET_ODD));

			// Reconstruct the result
			ResultReconstruct(out,tmp_out);
			MasterLog(INFO, "After reconstruct: out has norm =%lf  tmp_out has norm %lf", Norm2Vec(out), Norm2Vec(tmp_out));
		}
#endif

	private:

		const std::shared_ptr<const EOSmoother> _EOSmoother;
		const std::shared_ptr<const EOLinearOperator<Spinor,Gauge>> _EOOperator;
	};
}


#endif /* INCLUDE_LATTICE_UNPREC_SOLVER_WRAPPER_H_ */
