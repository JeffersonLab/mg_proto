/*
 * coarse_wilson_clover_linear_operator.h
 *
 *  Created on: Jan 12, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_COARSE_COARSE_EO_WILSON_CLOVER_LINEAR_OPERATOR_H_
#define INCLUDE_LATTICE_COARSE_COARSE_EO_WILSON_CLOVER_LINEAR_OPERATOR_H_

#include <vector>
#include <memory>
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_op.h"
#include "lattice/coarse/coarse_l1_blas.h"
#include "lattice/linear_operator.h"
#include "lattice/coarse/aggregate_block_coarse.h"
#include "utils/print_utils.h"
#include <omp.h>


namespace MG {

class CoarseEOWilsonCloverLinearOperator : public LinearOperator<CoarseSpinor,CoarseGauge > {
public:
	// Hardwire n_smt=1 for now.
	CoarseEOWilsonCloverLinearOperator(const std::shared_ptr<Gauge>& gauge_in, int level) : _u(gauge_in),
	 _the_op( gauge_in->GetInfo(), 1), _level(level)
	{

	}

	~CoarseEOWilsonCloverLinearOperator(){}

	const CBSubset& GetSubset() const override
	{
		return SUBSET_ODD;
	}

	void operator()(Spinor& out, const Spinor& in, IndexType type = LINOP_OP) const override {

			_the_op.EOPrecOp(out,      // Output Spinor
								(*_u),    // Gauge Field
								in,
								ODD,
								type);

	}

	void unprecOp(Spinor& out, const Spinor& in, IndexType type = LINOP_OP) const override {
#pragma omp parallel
		{
			int tid=omp_get_thread_num();
			_the_op.unprecOp(out,(*_u), in, EVEN, type,tid);
			_the_op.unprecOp(out,(*_u), in, ODD, type, tid);
		}
	}

	void leftOp(Spinor& out, const Spinor& in, IndexType type = LINOP_OP ) const override {
			_the_op.L_matrix(out, (*_u),in, type);
	}

	void leftInvOp(Spinor& out, const Spinor& in, IndexType type = LINOP_OP ) const override {
			_the_op.L_inv_matrix(out, (*_u),in, type);
	}

	void rightOp(Spinor& out, const Spinor& in, IndexType type = LINOP_OP ) const override {
			_the_op.R_matrix(out, (*_u),in, type);
	}
	void rigthInvOp(Spinor& out, const Spinor& in, IndexType type = LINOP_OP ) const override {
			_the_op.R_inv_matrix(out, (*_u),in, type);
	}

	void generateCoarse(const std::vector<Block>& blocklist, const std::vector< std::shared_ptr<CoarseSpinor> > in_vecs, CoarseGauge& u_coarse) const
	{
		const LatticeInfo& info = u_coarse.GetInfo();
		int num_colorspin = info.GetNumColorSpins();

		// Generate the triple products directly into the u_coarse
		ZeroGauge(u_coarse);
		for(int mu=0; mu < 8; ++mu) {
			MasterLog(INFO,"CoarseCloverLinearOperator: Dslash Triple Product in direction:%d",mu);
			dslashTripleProductDir(_the_op,blocklist, mu, (*_u), in_vecs, u_coarse);
		}


		MasterLog(INFO,"CoarseCloverLinearOperator: Clover Triple Product");
		clovTripleProduct(_the_op, blocklist, (*_u), in_vecs, u_coarse);
	}

	int GetLevel(void) const override {
		return _level;
	}

	const LatticeInfo& GetInfo(void) const override{
		return _u->GetInfo();
	}
private:
	const std::shared_ptr<Gauge> _u;
	const std::shared_ptr<Gauge> _clovInvU;
	const CoarseDiracOp _the_op;
	const int _level;

};

};




#endif /* TEST_QDPXX_COARSE_WILSON_CLOVER_LINEAR_OPERATOR_H_ */
