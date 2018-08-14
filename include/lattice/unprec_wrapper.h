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
	template<typename Spinor, typename Gauge,  class EOBase, template <typename,typename> class Base>
	class UnprecWrapper: public  Base<Spinor,Gauge> {
	public:
		UnprecWrapper(const std::shared_ptr<const EOBase> eo_solver,
					  const std::shared_ptr<const EOLinearOperator<Spinor,Gauge>>& linop)
		: _EOSolver(eo_solver), _EOOperator(linop), tmp_src(linop->GetInfo()), tmp_out(linop->GetInfo()) {}

		void SourcePrepare(Spinor& new_source, const Spinor& original_source) const override
		{
			_EOOperator->leftInvOp(new_source,original_source);
		}

		void ResultReconstruct(Spinor& new_result, const Spinor& original_result) const override
		{
			_EOOperator->rightInvOp(new_result,original_result);

		}

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

			// Solve even part directly
			_EOOperator->M_ee_inv(tmp_out, tmp_src, LINOP_OP);

			// Solve the odd part: Assume we don't touch even part.
			ret_val = (*_EOSolver)(tmp_out, tmp_src, resid_type);


			// Reconstruct the result
			ResultReconstruct(out,tmp_out);
			return ret_val;

		}

	private:

		const std::shared_ptr<const EOBase> _EOSolver;
		const std::shared_ptr<const EOLinearOperator<Spinor,Gauge>> _EOOperator;
		mutable Spinor tmp_src;
		mutable Spinor tmp_out;
	};

}


#endif /* INCLUDE_LATTICE_UNPREC_SOLVER_WRAPPER_H_ */
