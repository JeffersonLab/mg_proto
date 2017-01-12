/*
 * coarse_wilson_clover_linear_operator.h
 *
 *  Created on: Jan 12, 2017
 *      Author: bjoo
 */

#ifndef TEST_QDPXX_COARSE_WILSON_CLOVER_LINEAR_OPERATOR_H_
#define TEST_QDPXX_COARSE_WILSON_CLOVER_LINEAR_OPERATOR_H_

#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_op.h"
#include "lattice/linear_operator.h"
#include <omp.h>
using namespace MG;

namespace MGTesting {

class CoarseWilsonCloverLinearOperator : public LinearOperator<CoarseSpinor,CoarseGauge > {
public:
	// Hardwire n_smt=1 for now.
	CoarseWilsonCloverLinearOperator(const Gauge* gauge_in, const CoarseClover* clover_in, int level) : _u(gauge_in),
	_clov(clover_in), _the_op( gauge_in->GetInfo(), 1), _level(level)
	{
		AssertCompatible( gauge_in->GetInfo(), clover_in->GetInfo());


	}

	~CoarseWilsonCloverLinearOperator(){}

	void operator()(Spinor& out, const Spinor& in, IndexType type = LINOP_OP) const {


#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			for(int cb=0; cb < n_checkerboard; ++cb) {
				_the_op(out,      // Output Spinor
						(*_u),    // Gauge Field
						(*_clov), // The Coarse Clover Field
						in,
						cb,
						type,
						tid);
			}
		}
	}
	int GetLevel(void) const {
		return _level;
	}

	const LatticeInfo& GetInfo(void) const {
		return _u->GetInfo();
	}
private:
	const Gauge* _u;
	const CoarseClover* _clov;
	const CoarseDiracOp _the_op;
	const int _level;

};

};




#endif /* TEST_QDPXX_COARSE_WILSON_CLOVER_LINEAR_OPERATOR_H_ */
