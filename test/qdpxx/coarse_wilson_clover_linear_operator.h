/*
 * coarse_wilson_clover_linear_operator.h
 *
 *  Created on: Jan 12, 2017
 *      Author: bjoo
 */

#ifndef TEST_QDPXX_COARSE_WILSON_CLOVER_LINEAR_OPERATOR_H_
#define TEST_QDPXX_COARSE_WILSON_CLOVER_LINEAR_OPERATOR_H_

#include <vector>
#include <memory>
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_op.h"
#include "lattice/linear_operator.h"
#include "aggregate_block_coarse.h"
#include <omp.h>
using namespace MG;

namespace MGTesting {

class CoarseWilsonCloverLinearOperator : public LinearOperator<CoarseSpinor,CoarseGauge > {
public:
	// Hardwire n_smt=1 for now.
	CoarseWilsonCloverLinearOperator(const Gauge* gauge_in, int level) : _u(gauge_in),
	 _the_op( gauge_in->GetInfo(), 1), _level(level)
	{

	}

	~CoarseWilsonCloverLinearOperator(){}

	void operator()(Spinor& out, const Spinor& in, IndexType type = LINOP_OP) const {


			for(int cb=0; cb < n_checkerboard; ++cb) {
				_the_op(out,      // Output Spinor
						(*_u),    // Gauge Field
						in,
						cb,
						type,
						0);
			}

	}

	void generateCoarse(const std::vector<Block>& blocklist, const std::vector< std::shared_ptr<CoarseSpinor> > in_vecs, CoarseGauge& u_coarse) const
	{
		const LatticeInfo& info = u_coarse.GetInfo();
		int num_colorspin = info.GetNumColorSpins();

		// Generate the triple products directly into the u_coarse
		ZeroGauge(u_coarse);
		for(int mu=0; mu < 8; ++mu) {
			QDPIO::cout << "QDPWilsonCloverLinearOperator: Dslash Triple Product in direction: " << mu << std::endl;
			dslashTripleProductDir(_the_op,blocklist, mu, (*_u), in_vecs, u_coarse);
		}


		QDPIO::cout << "QDPWilsonCloverLinearOperator: Clover Triple Product" << std::endl;
		clovTripleProduct(_the_op, blocklist, (*_u), in_vecs, u_coarse);
	}

	int GetLevel(void) const {
		return _level;
	}

	const LatticeInfo& GetInfo(void) const {
		return _u->GetInfo();
	}
private:
	const Gauge* _u;
	const CoarseDiracOp _the_op;
	const int _level;

};

};




#endif /* TEST_QDPXX_COARSE_WILSON_CLOVER_LINEAR_OPERATOR_H_ */
