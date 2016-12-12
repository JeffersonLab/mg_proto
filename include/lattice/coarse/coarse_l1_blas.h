/*
 * coarse_l1_blas.h
 *
 *  Created on: Dec 12, 2016
 *      Author: bjoo
 */

#ifndef TEST_QDPXX_COARSE_L1_BLAS_H_
#define TEST_QDPXX_COARSE_L1_BLAS_H_

#include "lattice/coarse/coarse_types.h"
using namespace MG;

namespace MG {

/** returns || x - y ||^2
 * @param x  - CoarseSpinor ref
 * @param y  - CoarseSpinor ref
 * @return   double containing the norm of the difference
 */
double xmyNorm2Coarse(const CoarseSpinor& x, const CoarseSpinor& y);


}


#endif /* TEST_QDPXX_COARSE_L1_BLAS_H_ */
