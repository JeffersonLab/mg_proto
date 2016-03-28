/*
 * coarse_op.h
 *
 *  Created on: Jan 21, 2016
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_COARSE_COARSE_OP_H_
#define INCLUDE_LATTICE_COARSE_COARSE_OP_H_

#include "MG_config.h"
#include "lattice/constants.h"
#include "lattice/cmat_mult.h"
#include "lattice/coarse/coarse_types.h"

namespace MG {

struct ThreadLimits {


	IndexType min_vrow;
	IndexType max_vrow;
/*
 *  Was:
 *    min_row,max_vrow
 */
	IndexType min_site;
	IndexType max_site;
	unsigned char pad[MG_DEFAULT_CACHE_LINE_SIZE-4*sizeof(IndexType)]; // Cache line pad
};

class CoarseDiracOp {
public:
	CoarseDiracOp(const LatticeInfo& l_info, IndexType n_smt = 1);


	~CoarseDiracOp() {}

	/** The main user callable operator()
	 * Evaluate spinor_in [ 1 + \sum Y_{mu} delta_x+mu ] spinor_in
	 */

	void operator()(CoarseSpinor& spinor_out,
			const CoarseGauge& gauge_in,
			const CoarseClover& clover_in,
			const CoarseSpinor& spinor_in,
			const IndexType target_cb,
			const IndexType tid) const;

	void CloverApply(CoarseSpinor& spinor_out,
				const CoarseClover& clov_in,
				const CoarseSpinor& spinor_in,
				const IndexType target_cb,
				const IndexType tid) const;

	void Dslash(CoarseSpinor& spinor_out,
				const CoarseGauge& gauge_in,
				const CoarseSpinor& spinor_in,
				const IndexType target_cb,
				const IndexType tid) const;


	void siteApplyDslash( float *output,
			  	  	  	  	 	 const float* gauge_links[8],
								 const float* spinor_cb,
								 const float* neigh_spinors[8]) const;

	void siteApply( float *output,
			  	  	  	  	 	 const float* gauge_links[8],
								 const float* clover_cb_0,
								 const float* clover_cb_1,
								 const float* spinor_cb,
								 const float* neigh_spinors[8]) const ;

	void siteApplyClover( float *output,
						  const float* clover_chiral_0,
						  const float* clover_chiral_1,
						  const float *input) const ;


	inline
	IndexType GetNumColorSpin() const {
		return _n_colorspin;

	}

	inline
	IndexType GetNumColor() const {
		return _n_color;
	}

	inline
	IndexType GetNumSpin() const {
		return _n_spin;
	}

private:
	const LatticeInfo& _lattice_info;
	const IndexType _n_color;
	const IndexType _n_spin;
	const IndexType _n_colorspin;
	const IndexType _n_smt;
	const IndexType _n_vrows;

	int _n_threads;
	ThreadLimits* _thread_limits;

	// These are handy to have around
	// Scoped to the class. They can be computed on instantiation
	// as they are essentially in the lattice info.
	const IndexType _n_xh;
	const IndexType _n_x;
	const IndexType _n_y;
	const IndexType _n_z;
	const IndexType _n_t;



};



}

#endif /* INCLUDE_LATTICE_COARSE_COARSE_OP_H_ */
