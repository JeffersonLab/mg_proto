/*
 * invclov_coarse.cpp
 *
 *  Created on: Jun 7, 2018
 *      Author: bjoo
 */
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/aggregate_block_coarse.h"
#include "utils/print_utils.h"

#include <Eigen/Dense>



using Eigen::MatrixXcf;
using Eigen::Matrix;

using Eigen::Map;
using Eigen::RowMajor;
using Eigen::Dynamic;
using Eigen::Aligned64;

// It would be better if this was
#define RowOrder RowMajor

namespace MG {

// Invert the diagonal part of u, into eo_clov
void invertCloverDiag(CoarseGauge& u)
{
	const LatticeInfo& info = u.GetInfo();
	const int num_cbsites = info.GetNumCBSites();
	const int num_colorspin = info.GetNumColorSpins();

	MasterLog(INFO, "Inverting Diagonal Part of Gauge Field");

#pragma omp parallel for collapse(2)
	for(IndexType cb=0; cb < n_checkerboard; ++cb) {
		for(IndexType cbsite=0; cbsite < num_cbsites; ++cbsite ) {
			float* diag_data_in = u.GetSiteDirDataPtr(cb,cbsite,8);
			float* diag_data_out = u.GetSiteDirADDataPtr(cb,cbsite,8);

			Map< Matrix< std::complex<float>, Dynamic, Dynamic, RowOrder > > e_mat_in( reinterpret_cast<std::complex<float>*>(diag_data_in), num_colorspin, num_colorspin );
			Map< Matrix< std::complex<float>, Dynamic, Dynamic, RowOrder > > e_mat_out( reinterpret_cast<std::complex<float>*>(diag_data_out), num_colorspin, num_colorspin );

			e_mat_out = e_mat_in.inverse();
		}
	}
}

// Multiply the inverse part of the clover into eo_clov
void multInvClovOffDiaLeft(CoarseGauge& u)
{
	const LatticeInfo& info = u.GetInfo();
	const int num_cbsites = info.GetNumCBSites();
	const int num_colorspin = info.GetNumColorSpins();

	MasterLog(INFO, "Computing A^{-1} D");
	for(IndexType cb=0; cb < n_checkerboard; ++cb) {
		for(IndexType cbsite=0; cbsite < num_cbsites; ++cbsite ) {

			// This is
			float* diag_inv = u.GetSiteDirADDataPtr(cb,cbsite,8);
			Map< Matrix< std::complex<float>, Dynamic, Dynamic, RowOrder > > e_diag_inv( reinterpret_cast<std::complex<float>*>(diag_inv), num_colorspin, num_colorspin );

			// Go through the off diagonal links:

			for(int mu=0; mu < 8; ++mu) {

				// Original off diag link
				float* off_diag = u.GetSiteDirDataPtr(cb,cbsite,mu);

				// Result A^{-1} D link
				float* inv_off_diag = u.GetSiteDirADDataPtr(cb,cbsite,mu);

				// Wrap the pointers as Eigen Maps
				Map< Matrix< std::complex<float>, Dynamic, Dynamic, RowOrder > > e_off_diag( reinterpret_cast<std::complex<float>*>(off_diag), num_colorspin, num_colorspin );
				Map< Matrix< std::complex<float>, Dynamic, Dynamic, RowOrder > > e_inv_off_diag( reinterpret_cast<std::complex<float>*>(inv_off_diag), num_colorspin, num_colorspin );

				// Multiply
				e_inv_off_diag = e_diag_inv * e_off_diag;
			}
		}
	}
}

};



