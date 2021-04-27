/*
 * eigs_common.h
 *
 *  Created on: June 4, 2020
 *      Author: eloy
 */

#ifndef INCLUDE_LATTICE_EIGS_COMMON_H_
#define INCLUDE_LATTICE_EIGS_COMMON_H_

#include "lattice/constants.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"

namespace MG {

    //! Params for eigensolvers
    /*! \ingroup eigensolver */
    struct EigsParams : public MG::LinearSolverParamsBase {
    public:
        int MaxRestartSize; // Maximum rank of the search subspace
        int MaxNumEvals;    // Maximum number of eigenvalues to find
        EigsParams() : MaxRestartSize(0), MaxNumEvals(0) {}
    };
}

#endif
