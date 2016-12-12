/*
 * coarse_l1_blas.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: bjoo
 */

#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/coarse/coarse_l1_blas.h"


namespace MG
{

namespace GlobalComm {
	void GlobalSum( double& my_summand )
	{
		return; // Return Summand Unchanged -- MPI version should use an MPI_ALLREDUCE

	}
}


/** returns || x - y ||^2
 * @param x  - CoarseSpinor ref
 * @param y  - CoarseSpinor ref
 * @return   double containing the norm of the difference
 */
double xmyNorm2Coarse(const CoarseSpinor& x, const CoarseSpinor& y)
{
	double norm_diff = (double)0;

	const LatticeInfo& x_info = x.GetInfo();


	const LatticeInfo& y_info = y.GetInfo();
	AssertCompatible(x_info, y_info);


	IndexType num_cbsites = x_info.GetNumCBSites();
	IndexType num_colorspin = x.GetNumColorSpin();

	// Loop over the sites and sum up the norm
#pragma omp parallel for collapse(2) reduction(+:norm_diff)
	for(int cb=0; cb < 2; ++cb ) {
		for(int cbsite = 0; cbsite < num_cbsites; ++cbsite ) {


			// Identify the site
			const float* x_site_data = x.GetSiteDataPtr(cb,cbsite);
			const float* y_site_data = y.GetSiteDataPtr(cb,cbsite);

			// Sum difference over the colorspins
			for(int cspin=0; cspin < num_colorspin; ++cspin) {
				double diff_re = x_site_data[ RE + n_complex*cspin ] - y_site_data[ RE + n_complex*cspin ];
				double diff_im = x_site_data[ IM + n_complex*cspin ] - y_site_data[ IM + n_complex*cspin ];

				norm_diff += diff_re*diff_re + diff_im*diff_im;

			}

		}
	} // End of Parallel for reduction

	// I would probably need some kind of global reduction here  over the nodes which for now I will ignore.
	MG::GlobalComm::GlobalSum(norm_diff);

	return norm_diff;
}

};


