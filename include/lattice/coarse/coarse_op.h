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
#include "lattice/halo.h"
#include "utils/auxiliary.h"
#include "coarse_l1_blas.h"
#include <memory>

#include <omp.h>
namespace MG {



class CoarseDiracOp : public AuxiliarySpinors<CoarseSpinor> {
public:
	CoarseDiracOp(const LatticeInfo& l_info, IndexType n_smt = 1);


	~CoarseDiracOp() {}




	/** The main user callable operator()
	 * Evaluate spinor_in [ 1 + \sum Y_{mu} delta_x+mu ] spinor_in
	 */
#if 0
	void operator()(CoarseSpinor& spinor_out,
				const CoarseGauge& gauge_clov_in,
				const CoarseSpinor& spinor_in,
				const IndexType target_cb,
				const IndexType dagger,
				const IndexType tid) const
	{


	}
#endif

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
	// Use with UnprecOp?
	void M_D_xpay(CoarseSpinor& spinor_out,
			const float alpha,
			const CoarseGauge& gauge_in,
			const CoarseSpinor& spinor_in,
			const IndexType target_cb,
			const IndexType dagger,
			const IndexType tid) const;







	// spinor_out = spinor_cb + alpha A^{-1} D spinor_od
	void M_AD_xpayz(CoarseSpinor& spinor_out,
				           const float alpha,
						   const CoarseGauge& gauge_in,
						   const CoarseSpinor& spinor_cb,
						   const CoarseSpinor& spinor_od,
						   const IndexType target_cb,
						   const IndexType dagger,
						   const IndexType tid) const;

	// spinor_out = spinor_cb + alpha D A^{-1} spinor_od
	void M_DA_xpayz(CoarseSpinor& spinor_out,
				           const float alpha,
						   const CoarseGauge& gauge_in,
						   const CoarseSpinor& spinor_cb,
						   const CoarseSpinor& spinor_od,
						   const IndexType target_cb,
						   const IndexType dagger,
						   const IndexType tid) const;


	// spinor_out =  A^{-1} D spinor_od
	void M_AD(CoarseSpinor& spinor_out,
				const CoarseGauge& gauge_in,
				const CoarseSpinor& spinor_in,
				const IndexType target_cb,
				const IndexType dagger,
				const IndexType tid) const;

	// spinor_out = D A^{-1} spinor_od
	void M_DA(CoarseSpinor& spinor_out,
				const CoarseGauge& gauge_in,
				const CoarseSpinor& spinor_in,
				const IndexType target_cb,
				const IndexType dagger,
				const IndexType tid) const;



	// [  1                 0 ] [ spinor_in_e ] = [ spinor_in_e                              ]
	// [ A_oo^{-1} D_oe     1 ] [ spinor_in_o ]   [ spinor_in_o + A_oo^{-1} D_oe spinor_in_e ]
	void L_matrix(CoarseSpinor& spinor_out,
			const CoarseGauge& gauge_in,
			const CoarseSpinor& spinor_in) const {

		CopyVec(spinor_out,spinor_in, SUBSET_ALL);

#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			M_AD_xpayz(spinor_out, 1.0, gauge_in, spinor_out, spinor_in, ODD, LINOP_OP, tid );
		} // omp parallel

	}

	// [  1                 0 ] [ spinor_in_e ] = [ spinor_in_e                              ]
	// [ -A_oo^{-1} D_oe     1 ] [ spinor_in_o ]   [ spinor_in_o - A_oo^{-1} D_oe spinor_in_e ]
	void	L_inv_matrix(CoarseSpinor& spinor_out,
			const CoarseGauge& gauge_clov_in,
			const CoarseSpinor& spinor_in) const {

		CopyVec(spinor_out,spinor_in, SUBSET_ALL);


#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			M_AD_xpayz(spinor_out, -1.0, gauge_clov_in, spinor_out, spinor_in, ODD, LINOP_OP, tid );
		} // omp parallel

	}



	// R = [  1    A^{-1} M_eo ]
	//     [  0         1      ]
	void R_matrix(CoarseSpinor& spinor_out,
			const CoarseGauge& gauge_in,
			const CoarseSpinor& spinor_in) const {

		CopyVec( spinor_out, spinor_in, SUBSET_ALL  );


#pragma omp parallel
		{
			int tid=omp_get_thread_num();
			M_AD_xpayz(spinor_out, 1, gauge_in, spinor_out, spinor_in, EVEN, LINOP_OP, tid);
		}
	}

	// R^{inv} = [  1    -A^{-1} M_eo ]
	//           [  0         1      ]
	void R_inv_matrix(CoarseSpinor& spinor_out,
			const CoarseGauge& gauge_in,
			const CoarseSpinor& spinor_in) const {

		CopyVec( spinor_out, spinor_in, SUBSET_ALL  );

#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			M_AD_xpayz(spinor_out, -1, gauge_in, spinor_out, spinor_in, EVEN, LINOP_OP, tid);
		}

	}


	void EOPrecOp(CoarseSpinor& spinor_out,
			const CoarseGauge& gauge_in,
			const CoarseSpinor& spinor_in,
			const int target_cb,
			const IndexType dagger) const {
		std::shared_ptr<CoarseSpinor> t = tmp(spinor_in);

#pragma omp parallel
		{
			int tid = omp_get_thread_num();

			// dagger = LINOP_OP => tmp = A^{-1} D spinor_in
			// dagger = LINOP_DAGGER => tmp = Gamma_c D A^{-1} Gamma_c spinor in
			M_AD(*t,
					gauge_in,
					spinor_in,
					1-target_cb,
					dagger,
					tid);
#pragma omp barrier

			// dagger = LINOP_OP => out = spinor_in - A^{-1} D tmpvec = spinor_in - A^{-1} D A^{-1} D spinor_in
			// dagger = LINOP_DAGGER => out = spinor_in - Gamma_c D A^{-1} Gamma_c tmpvec
			//                              = spinor_in - Gamma_c D A^{-1} Gamma_c Gamma_c D A^{-1} Gamma_c spinor in
			//                              = spinor_in - Gamma_c D A^{1} D A^{-1} Gamma_c spinor_in
			//
			M_AD_xpayz(spinor_out,
					-1.0,
					gauge_in,
					spinor_in,
					*t,
					target_cb,
					dagger,
					tid);
		} // Parallel
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
		SpinorHaloCB& GetSpinorHalo() {
			return _halo;
		}
	inline
		const SpinorHaloCB& GetSpinorHalo() const {
			return _halo;
		}

	static void write(const CoarseGauge& gauge_clov_in, std::string& filename);
	
private:
	const LatticeInfo _lattice_info;
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

	mutable  SpinorHaloCB _halo;

};



}

#endif /* INCLUDE_LATTICE_COARSE_COARSE_OP_H_ */
