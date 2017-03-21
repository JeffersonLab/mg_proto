/*! \file
 *  \brief Minimal-Residual (MR) for a generic fermion Linear Operator
 */

#ifndef INCLUDE_LATTICE_FINE_QDPXX_INVMR_H_
#define INCLUDE_LATTICE_FINE_QDPXX_INVMR_H_

#include "qdp.h"
#include "lattice/constants.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"
#include "lattice/mr_params.h"

using namespace QDP;

namespace MG  {

  class MRSolver : public LinearSolver< QDP::LatticeFermion,QDP::multi1d<QDP::LatticeColorMatrix> > {
  public:
	  MRSolver(const LinearOperator<QDP::LatticeFermion,
			                        QDP::multi1d<QDP::LatticeColorMatrix> >& M,
									const MG::LinearSolverParamsBase& params);

	  LinearSolverResults operator()(QDP::LatticeFermion& out, const QDP::LatticeFermion& in, ResiduumType resid_type = RELATIVE) const;

  private:
	  const LinearOperator<QDP::LatticeFermion,QDP::multi1d<QDP::LatticeColorMatrix> >& _M;
	  const MRSolverParams& _params;

  };


  class MRSmoother : public Smoother<QDP::LatticeFermion,QDP::multi1d<QDP::LatticeColorMatrix> > {
  public:
	  MRSmoother(const LinearOperator<QDP::LatticeFermion,QDP::multi1d<QDP::LatticeColorMatrix> > & M, const MG::LinearSolverParamsBase& params);
	  void operator()(QDP::LatticeFermion& out, const QDP::LatticeFermion& in) const;

  private:
	  const LinearOperator<QDP::LatticeFermion,QDP::multi1d<QDP::LatticeColorMatrix> >& _M;
	  const MRSolverParams& _params;

  };
}

#endif /* TEST_QDPXX_INVMR_H_ */
