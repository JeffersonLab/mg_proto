/*
 * qphix_linear_operator.h
 *
 *  Created on: Aug 14, 2018
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_QPHIX_QPHIX_LINEAR_OPERATOR_H_
#define INCLUDE_LATTICE_QPHIX_QPHIX_LINEAR_OPERATOR_H_

#include "lattice/qphix/qphix_types.h"
#include "lattice/qphix/qphix_qdp_utils.h"
#include "lattice/coarse/block.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_l1_blas.h"
#include <qphix/clover.h>
#include "lattice/linear_operator.h"

namespace MG {

template<typename FT>
class QPhiXLinearOperator : public LinearOperator<QPhiXSpinorT<FT>,QPhiXGaugeT<FT> >
{
public:

	 virtual ~QPhiXLinearOperator() {};

	 virtual QPhiXClovOpT<FT>& getQPhiXOp() =0;

	 virtual void generateCoarse(const std::vector<Block>& blocklist,
	       const std::vector<std::shared_ptr<Spinor>>& in_vecs,
	       CoarseGauge& u_coarse) const = 0;

	 virtual void DslashDir(Spinor& spinor_out,
		        const Spinor& spinor_in,
		        const IndexType dir) const = 0;
};

template<typename FT>
class QPhiXLinearOperator : public LinearOperator<QPhiXSpinorT<FT>,QPhiXGaugeT<FT> >
{
public:

	 virtual ~QPhiXLinearOperator() {};

	 virtual QPhiXClovOpT<FT>& getQPhiXOp() =0;

	 virtual void generateCoarse(const std::vector<Block>& blocklist,
	       const std::vector<std::shared_ptr<Spinor>>& in_vecs,
	       CoarseGauge& u_coarse) const = 0;

	 virtual void DslashDir(Spinor& spinor_out,
		        const Spinor& spinor_in,
		        const IndexType dir) const = 0;
};

}



#endif /* INCLUDE_LATTICE_QPHIX_QPHIX_LINEAR_OPERATOR_H_ */
