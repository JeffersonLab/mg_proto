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

#pragma omp parallel for collapse(2)
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

void QDPGaugeLinksToCoarseGaugeLinks( const multi1d<LatticeColorMatrix>& qdp_u_in,
									  CoarseGauge& gauge_out )
{
	const LatticeInfo& info = gauge_out.GetInfo();
	int num_cb_sites = info.GetNumCBSites();

	assert( info.GetNumColors() == 3);
	assert( info.GetNumSpins() == 4);
	assert( num_cb_sites == rb[0].numSiteTable() );

	for(int dir=0; dir < n_dim; ++dir ) {
		LatticeColorMatrix u_back = adj(shift( qdp_u_in[dir], BACKWARD, dir));
		const LatticeColorMatrix& u_forw = qdp_u_in[dir];

		for(int cb=0; cb < n_checkerboard; ++cb) {
#pragma omp parallel for
			for(int cbsite=0; cbsite < num_cb_sites; ++cbsite) {

				float *site_data_forw = gauge_out.GetSiteDirDataPtr(cb,cbsite,2*dir);
				float *site_data_back = gauge_out.GetSiteDirDataPtr(cb,cbsite,2*dir+1);
				for(int row=0; row < 3; ++row) {
					for(int col=0; col < 3; ++col) {
						site_data_forw[ RE + n_complex*(col + 3*row)] = u_forw.elem(rb[cb].siteTable()[cbsite]).elem().elem(row,col).real();
						site_data_forw[ IM + n_complex*(col + 3*row)] = u_forw.elem(rb[cb].siteTable()[cbsite]).elem().elem(row,col).imag();
					} // col
				} // row
				for(int row=0; row < 3; ++row) {
					for(int col=0; col < 3; ++col) {
						site_data_back[ RE + n_complex*(col + 3*row)] = u_back.elem(rb[cb].siteTable()[cbsite]).elem().elem(row,col).real();
						site_data_back[ IM + n_complex*(col + 3*row)] = u_back.elem(rb[cb].siteTable()[cbsite]).elem().elem(row,col).imag();

					} // col
				} // row
			}// cbsite
		} // cb
	} // dir


}

void CoarseGaugeLinksToQDPGaugeLinks( const CoarseGauge& coarse_in,
									  multi1d<LatticeColorMatrix>& qdp_u_out)
{
	const LatticeInfo& info = coarse_in.GetInfo();
	int num_cb_sites = info.GetNumCBSites();

	assert( info.GetNumColors() == 3);
	assert( info.GetNumSpins() == 4);
	assert( num_cb_sites == rb[0].numSiteTable() );

	for(int dir=0; dir < n_dim; ++dir ) {
		//LatticeColorMatrix& u = qdp_u_out[dir];

		for(int cb=0; cb < n_checkerboard; ++cb) {
#pragma omp parallel for
			for(int cbsite=0; cbsite < num_cb_sites; ++cbsite) {

				const float *site_data_forw = coarse_in.GetSiteDirDataPtr(cb,cbsite,2*dir);
				for(int row=0; row < 3; ++row) {
					for(int col=0; col < 3; ++col) {
						qdp_u_out[dir].elem(rb[cb].siteTable()[cbsite]).elem().elem(row,col).real()=site_data_forw[ RE + n_complex*(col + 3*row)];
						qdp_u_out[dir].elem(rb[cb].siteTable()[cbsite]).elem().elem(row,col).imag()=site_data_forw[ IM + n_complex*(col + 3*row)];
					} // col
				} // row
			}// cbsite
		} // cb
	} // dir


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

#pragma omp parallel for collapse(2)
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
					prop_data[RE+n_complex*(colorspin_col + num_colorspin*colorspin_row)]
							  = qdpxx_in.elem(qdpxx_site).elem(spin_row,spin_col).elem(color_row,color_col).real();
					prop_data[IM+n_complex*(colorspin_col + num_colorspin*colorspin_row)]
							  = qdpxx_in.elem(qdpxx_site).elem(spin_row,spin_col).elem(color_row,color_col).imag();
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

#pragma omp parallel for collapse(2)
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
					qdpxx_out.elem(qdpxx_site).elem(spin_row, spin_col).elem(
							color_row, color_col).real() = prop_data[RE
																	 + n_complex
																	 * (colorspin_col
																			 + num_colorspin * colorspin_row)];
					qdpxx_out.elem(qdpxx_site).elem(spin_row, spin_col).elem(
							color_row, color_col).imag() =
									prop_data[IM
											  + n_complex
											  * (colorspin_col
													  + num_colorspin * colorspin_row)];

				}
			}
		}
	}
}

void QDPPropToCoarseClover(const LatticePropagator& qdpxx_in,
						   CoarseClover& coarse_out)
{
	IndexType num_coarse_color = coarse_out.GetNumColor();
	IndexType num_coarse_spin = coarse_out.GetNumChiral();
	IndexType num_cb_sites = coarse_out.GetInfo().GetNumCBSites();


	// Assert site tables are equal
	if ( num_cb_sites != rb[0].numSiteTable() ) {
		QDPIO::cerr << "Num cb sites in coarse = " << num_cb_sites
				<< " But QDP++ checkerboard size is" << rb[0].numSiteTable() << std::endl;
		QDP_abort(1);
	}


	// Assert num colorspin is 12 otherwise these are not compatible
	if ( num_coarse_color != 6 ) {
		QDPIO::cerr << "Num coarse color ( " << num_coarse_color <<" != 6 " << std::endl;
		QDP_abort(1);
	}

	if( num_coarse_spin != 2 ) {
		QDPIO::cerr << "Num coarse chiral blocks ( " << num_coarse_color <<" != 2 " << std::endl;
				QDP_abort(1);
	}


#pragma omp parallel for collapse(3)
	for (int cb = 0; cb < 2; ++cb) {
		for (int cbsite = 0; cbsite < num_cb_sites; ++cbsite) {

			for(int chiral_block = 0; chiral_block < num_coarse_spin; ++chiral_block) {


				// Get data for Chiral Block & Site. This is really an num_coarse_color * num_coarse_color complex matrix
				float *clov_data = coarse_out.GetSiteChiralDataPtr(cb,cbsite,chiral_block);

				// Loop through the colors
				for (int color_col = 0; color_col < num_coarse_color; ++color_col) {




					int pspin_col = (color_col / 3) + 2*chiral_block ; // Convert to fine spin
					int pcolor_col = color_col % 3; // Convert to fine color

					for (int color_row = 0; color_row < num_coarse_color; ++color_row) {

						int pspin_row = (color_row / 3) + 2*chiral_block; // Convert to Fine spin
						int pcolor_row = color_row % 3;    // Convert to fine color

						// Get QDPXX site for this site.
						int qdpxx_site = (rb[cb].siteTable())[cbsite];

						// Store Clov Data in coarse rows fasters
						clov_data[RE + n_complex*(color_col+num_coarse_color*color_row)] =
								qdpxx_in.elem(qdpxx_site).elem(pspin_row, pspin_col).elem(
										pcolor_row, pcolor_col).real();
						clov_data[IM + n_complex*(color_col+num_coarse_color*color_row)] =
								qdpxx_in.elem(qdpxx_site).elem(pspin_row, pspin_col).elem(
										pcolor_row, pcolor_col).imag();
					}

				}
			}
		}
	}
}



};
