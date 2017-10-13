/*
 * invfgmres.h
 *
 *  Created on: Jan 11, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_INVFGMRES_COARSE_H_
#define INCLUDE_LATTICE_INVFGMRES_COARSE_H_

#include "lattice/constants.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"
#include <complex>
#include "lattice/array2d.h"
#include <vector>
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_l1_blas.h"


#undef DEBUG_SOLVER

#include "lattice/fgmres_common.h"
#include "lattice/givens.h"

namespace MG {





 class FGMRESSolverCoarse : public LinearSolver<MG::CoarseSpinor,MG::CoarseGauge>
  {
  public:

    //! Constructor
    /*!
     * \param A_        Linear operator ( Read )
     * \param invParam  inverter parameters ( Read )
     */
    FGMRESSolverCoarse(const LinearOperator<CoarseSpinor,CoarseGauge>& A,
    			 const MG::LinearSolverParamsBase& params,
    			 const LinearSolver<CoarseSpinor,CoarseGauge>* M_prec=nullptr);

    ~FGMRESSolverCoarse();

    LinearSolverResults operator()(CoarseSpinor& out, const CoarseSpinor& in, ResiduumType resid_type = RELATIVE) const;


    void FlexibleArnoldi(int n_krylov,
			 const double rsd_target,
			 std::vector<CoarseSpinor*>& V,
			 std::vector<CoarseSpinor*>& Z,
			 Array2d<std::complex<double>>& H,
			 std::vector< FGMRES::Givens* >& givens_rots,
			 std::vector<std::complex<double>>& c,
			 int&  ndim_cycle,
			 ResiduumType resid_type) const;

    void LeastSquaresSolve(const Array2d<std::complex<double>>& H,
    		const std::vector<std::complex<double>>& rhs,
			std::vector<std::complex<double>>& eta,
			int n_cols) const;

  private:
    const LinearOperator<CoarseSpinor,CoarseGauge>& _A;
    const LatticeInfo& _info;
    const FGMRESParams _params;
    const LinearSolver<CoarseSpinor,CoarseGauge>* _M_prec;

    // These can become state variables, as they will need to be
    // handed around
    mutable Array2d<std::complex<double>> H_; // The H matrix
    mutable Array2d<std::complex<double>> R_; // R = H diagonalized with Givens rotations
    mutable std::vector<CoarseSpinor*> V_;  // K(A)
    mutable std::vector<CoarseSpinor*> Z_;  // K(MA)
    mutable std::vector< FGMRES::Givens* > givens_rots_;

    // This is the c = V^H_{k+1} r vector (c is frommers Notation)
    // For regular FGMRES I need to keep only the basis transformed
    // version of it, for rotating by the Q's (I call this 'g')
    // However, for FGMRES-DR I need this because I need
    //  || c - H eta || to extend the G_k matrix to form V_{k+1}
    // it is made of the previous v_{k+1} by explicitly evaluating c
    // except at the end of the first cycle when the V_{k+1} are not
    // yet available

    // Once G_k+1 is available, I can form the QR decomposition
    // and set  my little g = Q^H and then as I do the Arnoldi
    // I can then work it with the Givens rotations...

    mutable std::vector<std::complex<double>> c_;
    mutable std::vector<std::complex<double>> eta_;

  };

};




#endif /* TEST_QDPXX_INVFGMRES_H_ */
