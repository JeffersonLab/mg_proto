/*! \file
 *  \brief Conjugate-Gradient algorithm for a generic Linear Operator
 */

#ifndef INCLUDE_LATTICE_COARSE_INVBICGSTAB_COARSE_H_
#define INCLUDE_LATTICE_COARSE_INVBICGSTAB_COARSE_H_

#include "lattice/constants.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/unprec_wrapper.h"


namespace MG {



class BiCGStabSolverCoarse : public LinearSolver<CoarseSpinor,CoarseGauge> {
public:

	BiCGStabSolverCoarse(const LinearOperator<CoarseSpinor,CoarseGauge>& M, const LinearSolverParamsBase& params);

	  LinearSolverResults operator()(CoarseSpinor& out, const CoarseSpinor& in, ResiduumType resid_type = RELATIVE ) const;

 private:
	  const LinearOperator<CoarseSpinor,CoarseGauge>& _M;
	  const LinearSolverParamsBase& _params;

 };

using UnprecBiCGStabSolverCoarseWrapper = UnprecWrapper<CoarseSpinor,CoarseGauge, BiCGStabSolverCoarse, UnprecLinearSolver>;

}  // end namespace MGTEsting

#endif /* TEST_QDPXX_INVBICGSTAB_H_ */

