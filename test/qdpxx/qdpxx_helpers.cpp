/*
 * qdpxx_helpers.cpp
 *
 *  Created on: Mar 17, 2016
 *      Author: bjoo
 */

#include "qdpxx_helpers.h"

#include <qdp.h>
using namespace QDP;


using namespace MG;

namespace MGTesting {
	void initQDPXXLattice(const IndexArray& latdims )
	{
		multi1d<int> nrow(n_dim);
		for(int i=0; i < n_dim; ++i) nrow[i] = latdims[i];

		 Layout::setLattSize(nrow);
		 Layout::create();
	}


    void
	QDPSpinorToCoarseSpinor(const LatticeFermion& qdpxx_in,
    			CoarseSpinor& coarse_out)
    {
    	IndexType num_colorspin = coarse_out.GetNumColorSpin();
    	IndexType num_cb_sites = coarse_out.GetInfo().GetNumCBSites();


    	// Assert site tables are equal
    	if ( num_cb_sites != rb[0].numSiteTable() ) {
    		QDPIO::cerr << "Num cb sites in coarse = " << num_cb_sites
    					<< " But QDP++ checkerboard size is" << rb[0].numSiteTable() << std::endl;
    		QDP_abort(1);
    	}

    	// Assert num colorspin is 12 otherwise these are not compatible
    	if ( num_colorspin != 12 ) {
    		QDPIO::cerr << "Num colorspin ( " << num_colorspin<<" != 12 " << std::endl;
    		QDP_abort(1);
    	}

#pragma omp paralel for collapse(2)
    	for(int cb=0; cb < 2; ++cb) {
    		for(int cbsite=0; cbsite < num_cb_sites; ++cbsite) {
    			float *spinor_data=coarse_out.GetSiteDataPtr(cb,cbsite);

    			for(int colorspin=0; colorspin < num_colorspin; ++colorspin) {
    				int spin=colorspin/3;
    				int color=colorspin%3;

    				// real_part
    				int qdpxx_site = (rb[cb].siteTable())[cbsite];
    				spinor_data[RE+n_complex*colorspin] = qdpxx_in.elem(qdpxx_site).elem(spin).elem(color).real();
    				spinor_data[IM+n_complex*colorspin] = qdpxx_in.elem(qdpxx_site).elem(spin).elem(color).imag();
    			}
    		}
    	}
    }

    void
 	CoarseSpinorToQDPSpinor(const CoarseSpinor& coarse_in,
     			LatticeFermion& qdpxx_out)
     {
     	IndexType num_colorspin = coarse_in.GetNumColorSpin();
     	IndexType num_cb_sites = coarse_in.GetInfo().GetNumCBSites();

     	// Assert site tables are equal
     	if ( num_cb_sites != rb[0].numSiteTable() ) {
     		QDPIO::cerr << "Num cb sites in coarse = " << num_cb_sites
     					<< " But QDP++ checkerboard size is" << rb[0].numSiteTable() << std::endl;
     		QDP_abort(1);
     	}

     	// Assert num colorspin is 12 otherwise these are not compatible
     	if ( num_colorspin != 12 ) {
     		QDPIO::cerr << "Num colorspin ( " << num_colorspin<<" != 12 " << std::endl;
     		QDP_abort(1);
     	}

#pragma omp parallel for collapse(2)
     	for(int cb=0; cb < 2; ++cb) {
     		for(int cbsite=0; cbsite < num_cb_sites; ++cbsite) {

     			const float *spinor_data=coarse_in.GetSiteDataPtr(cb,cbsite);

     			for(int colorspin=0; colorspin < num_colorspin; ++colorspin) {
     				int spin=colorspin/3;
     				int color=colorspin%3;

     				// real_part
     				int qdpxx_site = (rb[cb].siteTable())[cbsite];
     				qdpxx_out.elem(qdpxx_site).elem(spin).elem(color).real() = spinor_data[RE+n_complex*colorspin];
     				qdpxx_out.elem(qdpxx_site).elem(spin).elem(color).imag() = spinor_data[IM+n_complex*colorspin];
     			}
     		}
     	}
     }


    void
	QDPPropToCoarseGaugeLink(const LatticePropagator& qdpxx_in,
    			CoarseGauge& coarse_out, int dir)
    {
    	IndexType num_colorspin = coarse_out.GetNumColorSpin();
    	IndexType num_cb_sites = coarse_out.GetInfo().GetNumCBSites();


    	// Assert site tables are equal
    	if ( num_cb_sites != rb[0].numSiteTable() ) {
    		QDPIO::cerr << "Num cb sites in coarse = " << num_cb_sites
    					<< " But QDP++ checkerboard size is" << rb[0].numSiteTable() << std::endl;
    		QDP_abort(1);
    	}

    	// Assert num colorspin is 12 otherwise these are not compatible
    	if ( num_colorspin != 12 ) {
    		QDPIO::cerr << "Num colorspin ( " << num_colorspin<<" != 12 " << std::endl;
    		QDP_abort(1);
    	}

#pragma omp paralel for collapse(2)
    	for(int cb=0; cb < 2; ++cb) {
    		for(int cbsite=0; cbsite < num_cb_sites; ++cbsite) {
    			float *prop_data=coarse_out.GetSiteDirDataPtr(cb,cbsite,dir);


    			    for(int colorspin_col=0; colorspin_col < num_colorspin; ++colorspin_col) {
    			    	int spin_col=colorspin_col/3;
    			    	int color_col=colorspin_col%3;

    			    	for(int colorspin_row=0; colorspin_row < num_colorspin; ++colorspin_row) {
    			    	    			    int spin_row=colorspin_row/3;
    			    	    			    int color_row=colorspin_row%3;


    			    	int qdpxx_site = (rb[cb].siteTable())[cbsite];
    			    	prop_data[RE+n_complex*(colorspin_row + num_colorspin*colorspin_col)]
								= qdpxx_in.elem(qdpxx_site).elem(spin_col,spin_row).elem(color_col,color_row).real();
    			    	prop_data[IM+n_complex*(colorspin_row + num_colorspin*colorspin_col)]
								= qdpxx_in.elem(qdpxx_site).elem(spin_col,spin_row).elem(color_col,color_row).imag();
    			    }
    			}
    		}
    	}
    }

    void
 	CoarseGaugeLinkToQDPProp(const CoarseGauge& coarse_in,
     			LatticePropagator& qdpxx_out, IndexType dir)
     {
    	IndexType num_colorspin = coarse_in.GetNumColorSpin();
    	IndexType num_cb_sites = coarse_in.GetInfo().GetNumCBSites();


    	// Assert site tables are equal
    	if ( num_cb_sites != rb[0].numSiteTable() ) {
    		QDPIO::cerr << "Num cb sites in coarse = " << num_cb_sites
    				<< " But QDP++ checkerboard size is" << rb[0].numSiteTable() << std::endl;
    		QDP_abort(1);
    	}

    	// Assert num colorspin is 12 otherwise these are not compatible
    	if ( num_colorspin != 12 ) {
    		QDPIO::cerr << "Num colorspin ( " << num_colorspin<<" != 12 " << std::endl;
    		QDP_abort(1);
    	}

#pragma omp paralel for collapse(2)
	for (int cb = 0; cb < 2; ++cb) {
		for (int cbsite = 0; cbsite < num_cb_sites; ++cbsite) {
			const float *prop_data = coarse_in.GetSiteDirDataPtr(cb,cbsite,dir);

			for (int colorspin_col = 0; colorspin_col < num_colorspin;
									++colorspin_col) {
								int spin_col = colorspin_col / 3;
								int color_col = colorspin_col % 3;

			for (int colorspin_row = 0; colorspin_row < num_colorspin;
					++colorspin_row) {
				int spin_row = colorspin_row / 3;
				int color_row = colorspin_row % 3;



					int qdpxx_site = (rb[cb].siteTable())[cbsite];
					qdpxx_out.elem(qdpxx_site).elem(spin_col, spin_row).elem(
							color_col, color_row).real() = prop_data[RE
							+ n_complex
									* (colorspin_row
											+ num_colorspin * colorspin_col)];
					qdpxx_out.elem(qdpxx_site).elem(spin_col, spin_row).elem(
							color_col, color_row).imag() =
					prop_data[IM
							+ n_complex
									* (colorspin_row
											+ num_colorspin * colorspin_col)];

				}
			}
		}
	}
}

};
