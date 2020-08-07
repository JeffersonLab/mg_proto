/*
 * invfgmres.h
 *
 *  Created on: Jan 11, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_FINE_QDPXX_INVFGMRES_H_
#define INCLUDE_LATTICE_FINE_QDPXX_INVFGMRES_H_

#include "lattice/constants.h"
#include "lattice/fgmres_common.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"
#include "qdp.h"

using namespace QDP;

namespace MG {

    namespace QDPFGMRES {

        class Givens {
        public:
            /*! Given  a marix H, construct the rotator so that H(row,col) = r and H(row+1,col) = 0
 	  *
 	  *  \param col  the column Input
 	  *  \param  H   the Matrix Input
 	  */

            Givens(int col, const multi2d<DComplex> &H);

            /*! Apply the rotation to column col of the matrix H. The
 	  *  routine affects col and col+1.
 	  *
 	  *  \param col  the columm
 	  *  \param  H   the matrix
 	  */

            void operator()(int col, multi2d<DComplex> &H);

            /*! Apply rotation to Column Vector v */
            void operator()(multi1d<DComplex> &v);

        private:
            int col_;
            DComplex s_;
            DComplex c_;
            DComplex r_;
        };

        void FlexibleArnoldiT(
            int n_krylov, const Real &rsd_target,
            const LinearOperator<LatticeFermion, multi1d<LatticeColorMatrix>> &A, // Operator
            const LinearSolver<LatticeFermion, multi1d<LatticeColorMatrix>> *M,   // Preconditioner
            multi1d<LatticeFermion> &V, multi1d<LatticeFermion> &Z, multi2d<DComplex> &H,
            multi1d<Givens *> &givens_rots, multi1d<DComplex> &c, int &ndim_cycle,
            ResiduumType resid_type, bool VerboseP);

    } // Namespace FGMRES

    class FGMRESSolverQDPXX : public LinearSolver<LatticeFermion, multi1d<LatticeColorMatrix>> {
    public:
        //! Constructor
        /*!
     * \param A_        Linear operator ( Read )
     * \param invParam  inverter parameters ( Read )
     */
        FGMRESSolverQDPXX(
            const LinearOperator<LatticeFermion, multi1d<LatticeColorMatrix>> &A,
            const MG::LinearSolverParamsBase &params,
            const LinearSolver<LatticeFermion, multi1d<LatticeColorMatrix>> *M_prec = nullptr);

        //! Initialize the internal matrices
        void InitMatrices();

        std::vector<LinearSolverResults> operator()(LatticeFermion &out, const LatticeFermion &in,
                                                    ResiduumType resid_type = RELATIVE) const;

        void FlexibleArnoldi(int n_krylov, const Real &rsd_target, multi1d<LatticeFermion> &V,
                             multi1d<LatticeFermion> &Z, multi2d<DComplex> &H,
                             multi1d<QDPFGMRES::Givens *> &givens_rots, multi1d<DComplex> &c,
                             int &ndim_cycle, ResiduumType resid_type) const;

        void LeastSquaresSolve(const multi2d<DComplex> &H, const multi1d<DComplex> &rhs,
                               multi1d<DComplex> &eta, int n_cols) const;

        const LatticeInfo &GetInfo() const { return _A.GetInfo(); }
        const CBSubset &GetSubset() const { return _A.GetSubset(); }

    private:
        const LinearOperator<LatticeFermion, multi1d<LatticeColorMatrix>> &_A;
        const LinearSolverParamsBase _params;
        const LinearSolver<LatticeFermion, multi1d<LatticeColorMatrix>> *_M_prec;

        // These can become state variables, as they will need to be
        // handed around
        mutable multi2d<DComplex> H_;       // The H matrix
        mutable multi2d<DComplex> R_;       // R = H diagonalized with Givens rotations
        mutable multi1d<LatticeFermion> V_; // K(A)
        mutable multi1d<LatticeFermion> Z_; // K(MA)
        mutable multi1d<QDPFGMRES::Givens *> givens_rots_;

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

        mutable multi1d<DComplex> c_;
        mutable multi1d<DComplex> eta_;
    };
}

#endif /* TEST_QDPXX_INVFGMRES_H_ */
