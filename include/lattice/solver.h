/*
 * solver.h
 *
 *  Created on: Jan 10, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_SOLVER_H_
#define INCLUDE_LATTICE_SOLVER_H_

namespace MG {
	enum ResiduumType { ABSOLUTE, RELATIVE, INVALID};

	struct LinearSolverResults {
		ResiduumType resid_type;
		int n_count;
		double resid;
	};

	template<typename Spinor, typename Gauge>
	class LinearSolver {
	public:
		virtual LinearSolverResults operator()(Spinor& out, const Spinor& in, ResiduumType resid_type = RELATIVE ) const=0;
		virtual ~LinearSolver(){}
	};

	/** A solver that solves the unpreconditioned systems */
	template<typename Spinor, typename Gauge, typename EOSolver>
	class UnprecLinearSolver : public LinearSolver<Spinor,Gauge> {
	public:
		virtual void SourcePrepare(Spinor& new_source, const Spinor& original_source) const = 0;
		virtual void InitGuessPrepare(Spinor& new_guess, const Spinor& original_guess) const = 0;
		virtual void OtherSubsetSolve(Spinor& new_guess, const Spinor& original_guess) const = 0;
		virtual void ResultReconstruct(Spinor& new_result, const Spinor& original_result) const = 0;
		virtual ~UnprecLinearSolver() {}
		virtual const EOSolver& GetEOSolver() const = 0;
		virtual Spinor& GetTmpSpinorIn() const = 0;
		virtual Spinor& GetTmpSpinorOut() const = 0;

		LinearSolverResults operator()(Spinor& out, const Spinor& in, ResiduumType resid_type = RELATIVE) const override {
			LinearSolverResults ret_val;
			Spinor& tmp_src = GetTmpSpinorIn();
			Spinor& tmp_out = GetTmpSpinorOut();

			// Prepare the source: L^{-1} in
			// In principle, this may change both even and odd parts,
			// Depending on the preconditioning style.
			// So worth preserving the prepped source.
			SourcePrepare(tmp_src,in);

			// Solve odd part with Krylov solver
			// Zero out the Even part of tmp_src for this
			// It is assumed that the solver will not touch the EVEN part.
			InitGuessPrepare(tmp_out,out);

			ret_val = (GetEOSolver())(tmp_out, tmp_src, resid_type);

			OtherSubsetSolve(tmp_out,tmp_src);

			// Reconstruct the result
			ResultReconstruct(out,tmp_out);
			return ret_val;

		}


	};


	// Base Parameter Struct
	class LinearSolverParamsBase {
	public:
		double RsdTarget;
		int MaxIter;
		bool VerboseP;
		LinearSolverParamsBase() {
			MaxIter=-1;
			VerboseP=false;

		}
	};

	// A Smoother Is much like a solver, but there are some 'don't care'-s
	// E.g. I may not care about the residua, and the iteration count may
	// be fixed.
	template<typename Spinor, typename Gauge>
	class Smoother {
	public:
		virtual void operator()(Spinor& out, const Spinor& in) const = 0;
		virtual ~Smoother(){}
	};

	// Base Parameter Struct
		class SmootherParamsBase {
		public:
			int MaxIter;
			bool VerboseP;
			SmootherParamsBase() {
				MaxIter = -1;
				VerboseP=false;

			}
		};

		/** A solver that solves the unpreconditioned systems */
		template<typename Spinor, typename Gauge, typename EOSmoother>
		class UnprecSmoother : public Smoother<Spinor,Gauge> {
		public:
			virtual void SourcePrepare(Spinor& new_source, const Spinor& original_source) const = 0;
			virtual void ResultReconstruct(Spinor& new_result, const Spinor& original_result) const = 0;
			virtual void InitGuessPrepare(Spinor& new_guess, const Spinor& original_guess) const = 0;
			virtual void OtherSubsetSolve(Spinor& new_guess, const Spinor& original_guess) const = 0;
			virtual ~UnprecSmoother() {}
			virtual const EOSmoother& GetEOSmoother() const = 0;
			virtual Spinor& GetTmpSpinorIn() const = 0;
			virtual Spinor& GetTmpSpinorOut() const = 0;
			void operator()(Spinor& out, const Spinor& in) const override {

				Spinor& tmp_src = GetTmpSpinorIn();
				Spinor& tmp_out = GetTmpSpinorOut();

				SourcePrepare(tmp_src,in);
				InitGuessPrepare(tmp_out,out);

				// Solve on odd part
				(GetEOSmoother())(tmp_out, tmp_src);

				// Solve even part directly
				OtherSubsetSolve(tmp_out, tmp_src);

				// Reconstruct the result
				ResultReconstruct(out,tmp_out);
			}
		};
};




#endif /* INCLUDE_LATTICE_SOLVER_H_ */
