/*
 * mg_level.h
 *
 *  Created on: Feb 11, 2016
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_MG_LEVEL_H_
#define INCLUDE_LATTICE_MG_LEVEL_H_

#include <vector>
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "utils/print_utils.h"
namespace MG {


class Spinor {
private:
	void *data;	// Amorphous Spinor
};


/** Linear Operator concept. Can apply itself to a vector (Spinor).
 *  Hermitian Conjugation must be implemented, tho this may be refactored
 *  if it is not easy to do for coarse Ops.
 *  Caveats:  LatticeInfo needs to be bound in here, and care should be taken that
 *    Spinor out and Spinor in have the same lattice info as the Operator, if this
 *    can't be type-checked at compile time.
 */
class LinearOperator {
public:
	LinearOperator(int level) : _level(level) {
		MasterLog(INFO, "Creating LinearOperator for level %d", _level);
	}

	void operator()(Spinor& out, const Spinor& in, LinOpType type = LINOP) {
		if (type == LINOP) {
			MasterLog(INFO, "Applying LinOp");
		} else {
			MasterLog(INFO, "Applying Daggered Op");
		}
	}

	int GetLevel(void) const {
		return _level;
	}

private:
	int _level;
};

/** Solver concept. Can apply itself to a vector (Spinor)
 *  Uses LinearOperator
 *  Care must be taken that the LinearOperator, The Input/Output and temporary spinors work
 *  on the same lattice if this can't be checked at compile time.
 *  In principle, a Solver is just a kind of LinOp so perhaps it should inherit the interface
 *  For now I will not do this and keep them as distinct. E.g. a Solver won't be able to handle
 *  Solving with a dagger...
 */

struct SolverParams {
	SolverParams() {
		overrelax_omega = 1.0;
		gmres_n_krylov = 10;
	}
	double overrelax_omega; 	// Overrelaxation parameter
	int max_iter;				// Maximum Iterations
	double rsd_target; 			// Target Residuum
	int gmres_n_krylov;			// Size of Krylov Subspace, for GMRES type algorithms								// This is 1 for MR & BiCGStab and ignored
};


class Solver {
public:
	virtual void operator()(Spinor& out, const Spinor& in, const SolverParams& params) const = 0;
	virtual ~Solver() {
	}

	/* Do I need an interface function to return the linop ? */
};

enum SolverType {
	MR, GCR, BICGSTAB
};


class MRSolver: public Solver {
public:
	MRSolver(const LinearOperator& M) : _M(M), _level(M.GetLevel())
	{
		MasterLog(INFO, "Creating MR Solver for Level %d", _level);
	}

	void operator()(Spinor& out, const Spinor& in,const SolverParams& param) const {
		MasterLog(INFO, "Applying MR solver on Level %d", _level);
		MasterLog(INFO, " \t RsdTarget=%16.8e MaxIter=%d Omega=%16.8e\n",
								param.rsd_target, param.max_iter, param.overrelax_omega );
	}

	~MRSolver() {
		MasterLog(INFO, "Destroying MR Solver for Level %d", _level);
	}
private:
	const LinearOperator& _M;
	int _level;
};

/** BiCGStab Solver */

class BiCGStabSolver: public Solver {
public:
	BiCGStabSolver(const LinearOperator& M) : _M(M),_level(M.GetLevel()) {
		MasterLog(INFO, "Creating BICGStab Solver for Level %d", _level);
	}

	void operator()(Spinor& out, const Spinor& in, const SolverParams& param) const {
		MasterLog(INFO, "Applying BICGStab solver on Level %d. MaxIter=%d RsdTarget=%16.8e", _level, param.max_iter, param.rsd_target);
	}

	~BiCGStabSolver() {
		MasterLog(INFO, "Destroying BiCGStab Solver for Level %d", _level);
	}
private:
	const LinearOperator& _M;
	int _level;
};

/** GCR Solver */

class GCRSolver: public Solver {
public:



	GCRSolver(const LinearOperator& M, const Solver* M_prec) : _M(M), _M_prec(M_prec),
				_level(M.GetLevel()) {
			if( M_prec == nullptr ) {
				MasterLog(INFO, "Creating Un-preconditioned GCR Solver for Level %d", _level);
			}
			else {
				MasterLog(INFO, "Creating Preconditioned GCR Solver for Level %d", _level);
			}
	}

	void operator()(Spinor& out, const Spinor& in, const SolverParams& param) const {
		if( _M_prec == nullptr ) {
			MasterLog(INFO, "Applying GCR solver on Level %d", _level);
		}
		else {
			MasterLog(INFO, "Applying Preconditioned GCR Solver on level %d", _level);
			(*_M_prec)(out, in, param);  // Fixme: PARAM Shouldn't be here
			                             // Maybe M_prec should be a linop...
		}
		MasterLog(INFO, " \t RsdTarget=%16.8e MaxIter=%d N_krylov=%d Omega=%16.8e\n",
						param.rsd_target, param.max_iter, param.gmres_n_krylov, param.overrelax_omega );
	}

	~GCRSolver() {
		MasterLog(INFO, "Destroying MR Solver for Level %d", _level);
	}

private:
	const LinearOperator& _M;
	const Solver* _M_prec;
	int _level;
};


inline
LinearOperator* createLinearOperator(int level) {

	return new LinearOperator(level);

}

/** This is essentially a solver factory..
 * we will want to redo this properly, to also deal with parameters
 */
inline
Solver *createSolver(SolverType type, const LinearOperator& M, const Solver* M_prec=nullptr) {
	Solver *ret_val = nullptr;
	switch (type) {
	case MR:
		ret_val = new MRSolver(M);
		break;
	case BICGSTAB:
		ret_val = new BiCGStabSolver(M);
		break;
	case GCR:
		ret_val = new GCRSolver(M, M_prec);
		break;
	default:
		MasterLog(ERROR, "Unknown solver type requested", M.GetLevel());
		break;
	}

	return ret_val;
}
/** Restrictor Concept
 *  I am not 100% sure of the interface to this yet.
 *  For now I will let it have two methods which act on Spinors
 */

class Restrictor {
public:
	Restrictor(int level, std::vector<Spinor*> vecs) : _level(level) {
		MasterLog(INFO, "Constructing restrictor from level %d to %d with %d vecs",
				level, level+1, vecs.size());
	}

	void operator()(Spinor& out, const Spinor& in) {
		MasterLog(INFO, "Restricting from Level %d to Level %d\n", _level, _level+1);
	}
private:
	int _level;
};

/** Prolongator Idea
 *
 */

class Prolongator {
public:
	Prolongator(int level, std::vector<Spinor*> vecs) : _level(level) {
			MasterLog(INFO, "Constructing Prolongator from level %d to %d with %d vecs",
					level+1, level, vecs.size());
		}

	void operator()(Spinor& out, const Spinor& in) {
		MasterLog(INFO, "Restricting from Level %d to Level %d\n", _level+1, _level);
	}
private:
	int _level;
};



struct MGLevel {
	std::vector<Spinor*> null_vecs; // NULL Vectors
	LatticeInfo* info;       // Info about the current lattice level
	Restrictor* R;                 // Restrict to next level
	Prolongator* P;                // Prolongate to previous level
	Solver* null_solver;           // Solver for NULL on this level
	Solver* smoother;          // PreSmoother on this level
	LinearOperator* M; // Linear Operator for this level (needed to construct pre smoother, post smoother, level_solver)

	~MGLevel() {
		if( M != nullptr) delete M;
		if( smoother != nullptr) delete smoother;
		if( null_solver != nullptr ) delete null_solver;
		if( P != nullptr ) delete P;
		if( R != nullptr ) delete R;
		if( info != nullptr ) delete info;
		for(int vec=0; vec < null_vecs.size(); ++vec) {
			if ( null_vecs[vec] != nullptr ) delete null_vecs[vec];
		}
 	}
};


inline
void gaussian(Spinor& s)
{
	MasterLog(INFO, "Filling spinor with gaussian noise");
}


inline
void zero(Spinor& s)
{
	MasterLog(INFO, "Filling spinor with zero");
}


inline
void blockOrthogonalize( std::vector< Spinor* >& vecs,
						 const IndexArray& block_dims)
{
	MasterLog(INFO, "Block Orthogonalizing Vectors with block size (%d %d %d %d)",
			block_dims[0],block_dims[1],block_dims[2],block_dims[3]);

}

struct SetupParams {
	int n_levels;
	std::vector<int> n_vecs;
	IndexArray local_lattice_size;
	std::vector< std::array<int,n_dim> > block_sizes;
	std::vector< int > null_solver_max_iter;
	std::vector< double > null_solver_rsd_target;
};

struct VCycleParams {
	// Pre Smoother Params
	int pre_hits;
	double pre_omega;
	double pre_rsd_target;

	// Lower Solve params
	double coarse_rsd_target;
	int    max_coarse_iter;
	int    N_krylov_coarse;
	double coarse_omega;

	// Post Smoother Params
	int post_hits;
	double post_omega;
	double post_rsd_target;
};

inline
Spinor *allocateSpinor(const LatticeInfo& info, int level)
{
	return new Spinor();
}

inline
void freeSpinor(Spinor* spinor)
{
	delete spinor;
}

class VCyclePreconditioner : public Solver {
public:
	VCyclePreconditioner(int level,
			std::vector<VCycleParams>& p,
			std::vector<MGLevel>& mg_levels) :
				_level(level),
				_n_levels(mg_levels.size()), _p(p), _mg_levels(mg_levels)
	{
		MasterLog(INFO, "VCycle(%d): Initializing", _level);

		if (_level == _n_levels - 2) {
			MasterLog(INFO, "VCycle(%d): Recursing to top level %d", _level,
					_level + 1);
			SolverCoarse = createSolver(GCR, *(_mg_levels[_level + 1].M),
					nullptr);
		} else {
			MasterLog(INFO, "VCycle(%d): Recursing to level %d", _level,
					_level + 1);

			PrecondCoarse = new VCyclePreconditioner(_level + 1, _p, _mg_levels);
			SolverCoarse = createSolver(GCR, *(_mg_levels[_level + 1].M),
					PrecondCoarse);
		}
		R = _mg_levels[_level].R;
		P = _mg_levels[_level + 1].P;
		Smoother = _mg_levels[_level].smoother;

		pre_smoother_params.max_iter = _p[_level].pre_hits;
		pre_smoother_params.rsd_target = _p[_level].pre_rsd_target;
		pre_smoother_params.overrelax_omega = _p[_level].pre_omega;

		post_smoother_params.max_iter = _p[_level].post_hits;
		post_smoother_params.rsd_target = _p[_level].post_rsd_target;
		post_smoother_params.overrelax_omega = _p[_level].post_omega;

		coarse_gcr_params.max_iter = _p[_level].max_coarse_iter;
		coarse_gcr_params.rsd_target = _p[_level].coarse_rsd_target;
		coarse_gcr_params.gmres_n_krylov = _p[_level].N_krylov_coarse;
		coarse_gcr_params.overrelax_omega = _p[_level].coarse_omega;

	}

	~VCyclePreconditioner()
	{
		if( _level != _n_levels - 2) {
			delete SolverCoarse;
		}
		else {
			delete SolverCoarse;
			delete PrecondCoarse;
		}
	}

	void operator()(Spinor& out, const Spinor& in, const SolverParams& param) const {
		if( _level < _n_levels - 1) {
			Spinor* smoothed_tmp = allocateSpinor(*(_mg_levels[_level].info), _level);
			Spinor* coarse_in = allocateSpinor(*(_mg_levels[_level-1].info), _level-1);
			Spinor* coarse_out = allocateSpinor(*(_mg_levels[_level-1].info), _level-1);
			MasterLog(INFO, " VCycle(%d): PreSmoothing", _level);
			(*Smoother)(*smoothed_tmp, in, pre_smoother_params);

			MasterLog(INFO, " VCycle(%d): Restricting to %d", _level, _level+1);
			(*R)(*smoothed_tmp, *coarse_in);

			MasterLog(INFO, " VCycle(%d): Coarse Solving", _level);
			(*SolverCoarse)(*coarse_out, *coarse_in, coarse_gcr_params);

			MasterLog(INFO, " VCycle(%d): Prolongating from %d", _level, _level+1);
			(*P)(*coarse_out, *smoothed_tmp);

			MasterLog(INFO, " VCycle(%d): PostSmoothing", _level);
			(*Smoother)(out, *smoothed_tmp, post_smoother_params);

			freeSpinor(coarse_out);
			freeSpinor(coarse_in);
			freeSpinor(smoothed_tmp);
		}
	}
public:
	int _level;
	int _n_levels;
	std::vector<VCycleParams>& _p;
	std::vector<MGLevel>& _mg_levels;
	Restrictor *R;
	Prolongator *P;
	Solver *Smoother;
	Solver *PrecondCoarse;
	Solver *SolverCoarse;
	SolverParams pre_smoother_params;
	SolverParams post_smoother_params;
	SolverParams coarse_gcr_params;

};


void mgSetup(const SetupParams& p,std::vector< MGLevel >& mg_levels);




}
#endif /* INCLUDE_LATTICE_MG_LEVEL_H_ */
