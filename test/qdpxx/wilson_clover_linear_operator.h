/*
 * wilson_clover_linear_operator.h
 *
 *  Created on: Jan 9, 2017
 *      Author: bjoo
 */

#ifndef TEST_QDPXX_WILSON_CLOVER_LINEAR_OPERATOR_H_
#define TEST_QDPXX_WILSON_CLOVER_LINEAR_OPERATOR_H_

/*!
 * Wrap a QDP++ linear operator, so as to provide the interface
 * with new types
 */
#include "lattice/linear_operator.h"
#include "clover_fermact_params_w.h"
#include "clover_term_qdp_w.h"
#include "qdpxx_helpers.h"
#include "dslashm_w.h"
#include "lattice/coarse/coarse_types.h"
#include "aggregate_block_qdpxx.h"

using namespace QDP;
using namespace MGTesting;

namespace MG {


class QDPWilsonCloverLinearOperator : public LinearOperator<LatticeFermion,multi1d<LatticeColorMatrix> > {
public:
	QDPWilsonCloverLinearOperator(float m_q, float c_sw, int t_bc, const Gauge& gauge_in ) : _t_bc(t_bc)
	{
		_u.resize(Nd);
		for(int mu=0; mu < Nd; ++mu) _u[mu] = gauge_in[mu];

		// Apply boundary
		_u[Nd-1] *= where(Layout::latticeCoordinate(Nd-1) == (Layout::lattSize()[Nd-1]-1),
			                        Integer(_t_bc), Integer(1));
		_params.Mass =  Real(m_q);
		_params.clovCoeffR = Real(c_sw);
		_params.clovCoeffT = Real(c_sw);
		_clov.create(_u,_params);  // Make the clover term
	}

	void operator()(Spinor& out, const Spinor& in, IndexType type = LINOP_OP) const {

		// Cut this out of Chroma, modified for explicit checkerrboarding

		LatticeFermion tmp=zero;
		Real mhalf=-0.5;
		const int isign = (type == LINOP_OP ) ? 1: -1;

		for(int cb=0; cb < n_checkerboard;++cb)  {
			_clov.apply(out, in, isign, cb );       // CB -> CB
			MGTesting::dslash(tmp, _u, in, isign, cb);    // (1-CB) -> C
			out[rb[cb]] += mhalf*tmp;
		}
	}

	int GetLevel(void) const {
		return 0;
	}

	void generateCoarseGauge(const std::vector<Block>& blocklist, const multi1d<LatticeFermion>& in_vecs, CoarseGauge& u_coarse)
	{
		// Generate the triple products directly into the u_coarse
		for(int mu=0; mu < 8; ++mu) {
			QDPIO::cout << "QDPWilsonCloverLinearOperator: Dslash Triple Product in direction: " << mu << std::endl;
			dslashTripleProductDirQDPXX(blocklist, mu, _u, in_vecs, u_coarse);
		}
	}

	void generateCoarseClover(const std::vector<Block>& blocklist, const multi1d<LatticeFermion>& in_vecs, CoarseClover& coarse_clov)
	{
		QDPIO::cout << "QDPWilsonCloverLinearOperator: Clover Triple Product" << std::endl;
		clovTripleProductQDPXX(blocklist, _clov, in_vecs, coarse_clov);
	}
private:
	const int _t_bc;
	Gauge _u;
	MGTesting::CloverFermActParams _params;
	MGTesting::QDPCloverTerm _clov;
	LatticeFermion from;
	LatticeFermion to;
};


}



#endif /* TEST_QDPXX_WILSON_CLOVER_LINEAR_OPERATOR_H_ */
