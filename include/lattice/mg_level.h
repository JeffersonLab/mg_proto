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
		MasterLog(INFO, "Applying BICGStab solver on Level %d", _level);
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
		}
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
		MasterLog(ERROR, "Unknown solver type requested", level);
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
	Solver* pre_smoother;          // PreSmoother on this level
	Solver* post_smoother;         // Post Smoother on this level
	LinearOperator* M; // Linear Operator for this level (needed to construct pre smoother, post smoother, level_solver)
	Solver* level_solver; // The level solver on this level (Does this make sense? This could be part of the cycle)

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

void mgSetup(const SetupParams& p,std::vector< MGLevel >& mg_levels);




}
#endif /* INCLUDE_LATTICE_MG_LEVEL_H_ */
