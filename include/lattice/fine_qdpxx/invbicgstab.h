/*! \file
 *  \brief Conjugate-Gradient algorithm for a generic Linear Operator
 */

#ifndef INCLUDE_LATTICE_FINE_QDPXX_INVBICGSTAB_H_
#define INCLUDE_LATTICE_FINE_QDPXX_INVBICGSTAB_H_

#include "qdp.h"
#include "lattice/constants.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"


using namespace QDP;

namespace MG {



class BiCGStabSolver : public LinearSolver<QDP::LatticeFermion,QDP::multi1d<QDP::LatticeColorMatrix> > {
public:
	using Spinor = QDP::LatticeFermion;
	using Gauge = QDP::multi1d< QDP::LatticeColorMatrix> ;
	BiCGStabSolver(const LinearOperator<Spinor,Gauge>& M, const LinearSolverParamsBase& params) : _M(M),
	  _params(params){}

	  LinearSolverResults operator()(Spinor& out, const Spinor& in, ResiduumType resid_type = RELATIVE ) const;
 private:
	  const LinearOperator<Spinor,Gauge>& _M;
	  const LinearSolverParamsBase& _params;

 };

}  // end namespace MGTEsting

#endif /* TEST_QDPXX_INVBICGSTAB_H_ */

