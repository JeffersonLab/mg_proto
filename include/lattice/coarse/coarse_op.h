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
#include "coarse_l1_blas.h"
#include <omp.h>
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
				const IndexType tid) const
	{
		unprecOp(spinor_out,
					gauge_clov_in,
					spinor_in,
					target_cb,
					dagger,
					tid);
	}
    // Applies M on target checkerboard
	// Full Op is:
	//
	// [ y_e ] = [ Mee  M_eo ] [ x_e ]
	// [ y_o ]   [ M_oe M_oo ] [ x_o ]
	//
	//   so  e.g.  y_cb = M_(cb,cb) x_(cb) + M_(cb,1-cb) x_(1-cb)
	void unprecOp(CoarseSpinor& spinor_out,
			const CoarseGauge& gauge_clov_in,
			const CoarseSpinor& spinor_in,
			const IndexType target_cb,
			const IndexType dagger,
			const IndexType tid) const ;

	// Apply Diagonal part with U, or apply M_ee/oo
	void M_diag(CoarseSpinor& spinor_out,
			const CoarseGauge& gauge_clov_in,
			const CoarseSpinor& spinor_in,
			const IndexType target_cb,
			const IndexType dagger,
			const IndexType tid) const;

	// Apply Inverse of Diagonal Part or apply M^{-1}_ee/oo
	void M_diagInv(CoarseSpinor& spinor_out,
				const CoarseGauge& gauge_clov_in,
				const CoarseSpinor& spinor_in,
				const IndexType target_cb,
				const IndexType dagger,
				const IndexType tid) const;

	// Apply OffDoagonal and add-to/subtract from spinor_out
	//  spinor_out_o/e + alpha* M_oe/eo spinor_in_e/o
	void M_offDiag_xpay(CoarseSpinor& spinor_out,
				       const float alpha,
				       const CoarseGauge& gauge_in,
					   const CoarseSpinor& spinor_in,
					   const IndexType target_cb,
					   const IndexType dagger,
					   const IndexType tid) const;


	// Apply M_diag^{-1}M_off_diag
	void M_invOffDiag_xpay(CoarseSpinor& spinor_out,
				           const float alpha,
						   const CoarseGauge& gauge_in,
						   const CoarseSpinor& spinor_in,
						   const IndexType target_cb,
						   const IndexType dagger,
						   const IndexType tid) const;

	// Apply M_diag^{-1}M_off_diag
	void M_invOffDiag_xpayz(CoarseSpinor& spinor_out,
				           const float alpha,
						   const CoarseGauge& gauge_in,
						   const CoarseSpinor& spinor_cb,
						   const CoarseSpinor& spinor_od,
						   const IndexType target_cb,
						   const IndexType dagger,
						   const IndexType tid) const;

	// Apply M_diag^{-1}M_off_diag
	void M_invOffDiag(CoarseSpinor& spinor_out,
				const CoarseGauge& gauge_in,
				const CoarseSpinor& spinor_in,
				const IndexType target_cb,
				const IndexType dagger,
				const IndexType tid) const;

	// [  M_ee      0   ] [ spinor_in_e ] = [ M_ee spinor_in_e                    ]
	// [  M_oe     M_oo ] [ spinor_in_o ]   [ M_oo spinor_in_o + M_oe spinor_in_o ]
		void L_matrix(CoarseSpinor& spinor_out,
				  const CoarseGauge& gauge_in,
				  const CoarseSpinor& spinor_in,
				  const IndexType dagger) const {


			  if( dagger == LINOP_OP) {
#pragma omp parallel
				{
					int tid = omp_get_thread_num();

					M_diag(spinor_out, gauge_in, spinor_in, 0, LINOP_OP, tid );
					unprecOp(spinor_out, gauge_in, spinor_in, 1, LINOP_OP, tid);

				} // omp parallel
			  }
			  else {
				  // Dagger not yet implemented.

			  }
		}


	void	L_inv_matrix(CoarseSpinor& spinor_out,
					const CoarseGauge& gauge_clov_in,
					const CoarseSpinor& spinor_in,
					const IndexType dagger) const;



	// R = [  1    A^{-1} M_eo ]
	//     [  0         1      ]
	void R_matrix(CoarseSpinor& spinor_out,
				  const CoarseGauge& gauge_in,
				  const CoarseSpinor& spinor_in,
				  const IndexType dagger) const {
		if( dagger == LINOP_OP )  {
			CopyVec( spinor_out, spinor_in);
#pragma omp parallel
			{
				int tid=omp_get_thread_num();
				M_invOffDiag_xpay(spinor_out, 1, gauge_in, spinor_in, 0, dagger, tid);
			}
		}
		else {
			// Not yet imlemented
		}
	}

	// R^{inv} = [  1    -A^{-1} M_eo ]
	//           [  0         1      ]
	void R_inv_matrix(CoarseSpinor& spinor_out,
					const CoarseGauge& gauge_in,
					const CoarseSpinor& spinor_in,
					const IndexType dagger) const {
		if( dagger == LINOP_OP ) {
			CopyVec( spinor_out, spinor_in  );
#pragma omp parallel
			{
				int tid = omp_get_thread_num();
				M_invOffDiag_xpay(spinor_out, -1, gauge_in, spinor_in, 0, dagger, tid);
			}
		}
		else {
			// Dagger Not yet Implemented
		}
	}


	void EOPrecOp(CoarseSpinor& spinor_out,
			const CoarseGauge& gauge_in,
			const CoarseSpinor& spinor_in,
			const int target_cb,
			const IndexType dagger) const {

#pragma omp parallel
		{
			int tid = omp_get_thread_num();

			M_invOffDiag(_tmpvec,
					gauge_in,
					spinor_in,
					1-target_cb,
					dagger,
					tid);
#pragma omp barrier
			M_invOffDiag_xpayz(spinor_out,
					-1.0,
					gauge_in,
					spinor_in,
					_tmpvec,
					target_cb,
					dagger,
					tid);
		}

	}

	void Schur_matrix(CoarseSpinor& spinor_out,
			const CoarseGauge& gauge_in,
			const CoarseSpinor& spinor_in,
			const IndexType dagger) const {


			CopyVec(spinor_out,spinor_in, SUBSET_EVEN);
			EOPrecOp(spinor_out,gauge_in,spinor_in,ODD,dagger);

	}

	void CloverApply(CoarseSpinor& spinor_out,
				const CoarseGauge& gauge_clov_in,
				const CoarseSpinor& spinor_in,
				const IndexType target_cb,
				const IndexType dagger,
				const IndexType tid) const
	{
		M_diag(spinor_out,
				gauge_clov_in,
				spinor_in,
				target_cb,
				dagger,
				tid);
	}






	// output = sum_0..7 U_mu neigh_mu
	void siteApplyDslash( float* output,
			  	  	  	  	 	 const float* gauge_links[9],
								 const float* neigh_spinors[8],
								 const IndexType dagger) const;



    // output =  in_1 +/- sum_0..7 U_mu neigh_mu
	void siteApplyDslash_xpmy( float* output,
								 const float coeff,
			  	  	  	  	 	 const float* gauge_links[9],
								 const float* neigh_spinors[8],
								 const IndexType dagger) const;


	void siteApplyDslash_xpayz( float *output,
								 const float coeff,
			  	  	  	  	 	 const float* gauge_links[9],
								 const float* in_spinor_cb,
								 const float* neigh_spinors[8],
								 const IndexType dagger) const;

	// output = A_ee input
	void siteApplyClover( float* output,
						  const float* clover,
						  const float* input,
						  const IndexType dagger) const ;

	void DslashDir(CoarseSpinor& spinor_out,
						const CoarseGauge& gauge_in,
						const CoarseSpinor& spinor_in,
						const IndexType target_cb,
						const IndexType dir,
						const IndexType tid) const;



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
	mutable CoarseSpinor _tmpvec;

};



}

#endif /* INCLUDE_LATTICE_COARSE_COARSE_OP_H_ */
