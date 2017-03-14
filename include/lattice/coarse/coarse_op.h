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
#include "lattice/coarse/thread_limits.h"
#include "lattice/spinor_halo.h"

namespace MG {



class CoarseDiracOp {
public:
	CoarseDiracOp(const LatticeInfo& l_info, IndexType n_smt = 1);


	~CoarseDiracOp() {}

	/** The main user callable operator()
	 * Evaluate spinor_in [ 1 + \sum Y_{mu} delta_x+mu ] spinor_in
	 */

	void operator()(CoarseSpinor& spinor_out,
			const CoarseGauge& gauge_clov_in,
			const CoarseSpinor& spinor_in,
			const IndexType target_cb,
			const IndexType dagger,
			const IndexType tid) const ;

	void CloverApply(CoarseSpinor& spinor_out,
				const CoarseGauge& gauge_clov_in,
				const CoarseSpinor& spinor_in,
				const IndexType target_cb,
				const IndexType dagger,
				const IndexType tid) const;


	void DslashDir(CoarseSpinor& spinor_out,
					const CoarseGauge& gauge_in,
					const CoarseSpinor& spinor_in,
					const IndexType target_cb,
					const IndexType dir,
					const IndexType tid) const;

	void siteApplyDslash( float* output,
			  	  	  	  	 	 const float* gauge_links[9],
								 const float* spinor_cb,
								 const float* neigh_spinors[8],
								 const IndexType dagger) const;





	void siteApplyClover( float* output,
						  const float* clover,
						  const float* input,
						  const IndexType dagger) const ;


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



	inline
		MGTesting::SpinorHaloCB& GetSpinorHalo() {
			return _halo;
		}
	inline
		const MGTesting::SpinorHaloCB& GetSpinorHalo() const {
			return _halo;
		}


	void packFace( const CoarseSpinor& spinor, IndexType cb, IndexType mu, IndexType fb) const;

	const float*
	GetNeighborXPlus(int x, int y, int z, int t, int source_cb, const CoarseSpinor& spinor_in) const;

	const float*
	GetNeighborXMinus(int x, int y, int z, int t, int source_cb, const CoarseSpinor& spinor_in) const;

	const float*
	GetNeighborYPlus(int xcb, int y, int z, int t, int source_cb, const CoarseSpinor& spinor_in) const;

	const float*
	GetNeighborYMinus(int xcb, int y, int z, int t, int source_cb, const CoarseSpinor& spinor_in) const ;

	const float*
	GetNeighborZPlus(int xcb, int y, int z, int t, int source_cb, const CoarseSpinor& spinor_in) const;

	const float*
	GetNeighborZMinus(int xcb, int y, int z, int t, int source_cb, const CoarseSpinor& spinor_in) const;

	const float*
	GetNeighborTPlus(int xcb, int y, int z, int t, int source_cb, const CoarseSpinor& spinor_in) const ;

	const float*
	GetNeighborTMinus(int xcb, int y, int z, int t, int source_cb, const CoarseSpinor& spinor_in) const;

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

	mutable  MGTesting::SpinorHaloCB _halo;


};



}

#endif /* INCLUDE_LATTICE_COARSE_COARSE_OP_H_ */
