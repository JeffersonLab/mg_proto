/*
 * aggregate_qdpxx.cpp
 *
 *  Created on: Dec 9, 2016
 *      Author: bjoo
 */

#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/geometry_utils.h"
#include "lattice/coarse/coarse_l1_blas.h"
#include "lattice/coarse/aggregate_block_coarse.h"
#include "lattice/halo.h"
#include <cassert>

#include<omp.h>

// Eigen Dense header
#include <Eigen/Dense>
using namespace Eigen;

namespace MG {


// Implementation -- where possible call the site versions
//! v *= alpha (alpha is real) over and aggregate in a block, v is a QDP++ Lattice Fermion
void axBlockAggr(const double alpha, CoarseSpinor& v, const Block& block, int aggr)
{
	auto block_sitelist = block.getCBSiteList();
	int num_sites = block.getNumSites();
	const LatticeInfo& info = v.GetInfo();

	const int num_color = v.GetNumColor();
	const int n_per_chiral = ( info.GetNumSpins() == 4 ) ? 2*num_color : num_color;

	const int min_cspin = aggr*n_per_chiral;
	const int max_cspin = (aggr+1)*n_per_chiral;

#pragma omp parallel for
	for(int site=0; site < num_sites; ++site) {
		const CBSite& cbsite = block_sitelist[site];
		float *v_site_data = v.GetSiteDataPtr(cbsite.cb, cbsite.site);
		for(int j=min_cspin; j < max_cspin; ++j) {
			v_site_data[RE + n_complex*j] *= alpha;
			v_site_data[IM + n_complex*j] *= alpha;
		}
	}
}

//! y += alpha * x (alpha is complex) over aggregate in a block, x, y are QDP++ LatticeFermions;
void caxpyBlockAggr(const std::complex<double>& alpha, const CoarseSpinor& x, CoarseSpinor& y,  const Block& block, int aggr)
{

	auto block_sitelist = block.getCBSiteList();
	int num_sites = block.getNumSites();

	AssertCompatible( x.GetInfo(), y.GetInfo() );

	const LatticeInfo& info = y.GetInfo();
	const int num_color = y.GetNumColor();
	const int n_per_chiral = ( info.GetNumSpins() == 4 ) ? 2*num_color : num_color;

	const int min_cspin = aggr*n_per_chiral;
	const int max_cspin = (aggr+1)*n_per_chiral;

#pragma omp parallel for
	for(int site=0; site < num_sites; ++site) {
		const CBSite& cbsite = block_sitelist[site];
		const float *x_site_data = x.GetSiteDataPtr(cbsite.cb, cbsite.site);
		float *y_site_data = y.GetSiteDataPtr(cbsite.cb, cbsite.site);
		for(int colorspin=min_cspin; colorspin < max_cspin; ++colorspin) {
			double ar = real(alpha);
			double ai = imag(alpha);
			double xr = x_site_data[ RE + n_complex*colorspin];
			double xi = x_site_data[ IM + n_complex*colorspin];

			y_site_data[ RE + n_complex*colorspin] += ar*xr - ai*xi;
			y_site_data[ IM + n_complex*colorspin] += ar*xi + ai*xr;

		}
	}

}



//! return || v ||^2 over an aggregate in a block, v is a QDP++ LatticeFermion
double norm2BlockAggr(const CoarseSpinor& v, const Block& block, int aggr)
{
	auto block_sitelist = block.getCBSiteList();
	int num_sites = block.getNumSites();

	const LatticeInfo& info = v.GetInfo();
	const int num_color = v.GetNumColor();
	const int n_per_chiral = ( info.GetNumSpins() == 4 ) ? 2*num_color : num_color;

	const int min_cspin = aggr*n_per_chiral;
	const int max_cspin = (aggr+1)*n_per_chiral;

	double block_sum=0;


#pragma omp parallel for reduction(+:block_sum)
	for(int site=0; site < num_sites; ++site) {
		const CBSite& cbsite = block_sitelist[site];
		const float *v_site_data = v.GetSiteDataPtr(cbsite.cb, cbsite.site);
		for(int colorspin=min_cspin; colorspin < max_cspin; ++colorspin) {

			double vr = v_site_data[ RE + n_complex*colorspin];
			double vi = v_site_data[ IM + n_complex*colorspin];

			block_sum += vr*vr + vi*vi;
		}
	}

	return block_sum;
}

//! return < left | right > = sum left^\dagger_i * right_i for an aggregate, over a block
std::complex<double>
innerProductBlockAggr(const CoarseSpinor& left, const CoarseSpinor& right, const Block& block, int aggr)
{

	auto block_sitelist = block.getCBSiteList();
	int num_sites = block.getNumSites();

	AssertCompatible( left.GetInfo(), right.GetInfo() );

	const LatticeInfo& info = right.GetInfo();
	const int num_color = right.GetNumColor();
	const int n_per_chiral = ( info.GetNumSpins() == 4 ) ? 2*num_color : num_color;

	const int min_cspin = aggr*n_per_chiral;
	const int max_cspin = (aggr+1)*n_per_chiral;

	double real_part=0;
	double imag_part=0;

#pragma omp parallel for reduction(+:real_part) reduction(+:imag_part)
	for(int site=0; site < num_sites; ++site) {

		const CBSite& cbsite = block_sitelist[site];
		const float *left_site_data = left.GetSiteDataPtr(cbsite.cb, cbsite.site);
		const float *right_site_data = right.GetSiteDataPtr(cbsite.cb, cbsite.site);
		for(int colorspin=min_cspin; colorspin < max_cspin; ++colorspin) {

			double left_r = left_site_data[ RE + n_complex*colorspin];
			double left_i = left_site_data[ IM + n_complex*colorspin];

			double right_r = right_site_data[ RE + n_complex*colorspin];
			double right_i = right_site_data[ IM + n_complex*colorspin];

			real_part +=  (left_r*right_r) + (left_i*right_i);
			imag_part +=  (left_r*right_i) - (left_i*right_r);
		}
	}

	std::complex<double> ret_val(real_part,imag_part);
	return ret_val;
}

//! Extract the spins belonging to a given aggregate from QDP++ source vector src, into QDP++ target vector target
void extractAggregateBlock(CoarseSpinor& target, const CoarseSpinor& src, const Block& block, int aggr )
{

	auto block_sitelist = block.getCBSiteList();
	int num_sites = block.getNumSites();


	const LatticeInfo& info = src.GetInfo();
	AssertCompatible( info, target.GetInfo() );

	const int num_color = src.GetNumColor();
	const int n_per_chiral = ( info.GetNumSpins() == 4 ) ? 2*num_color : num_color;

	const int min_cspin = aggr*n_per_chiral;
	const int max_cspin = (aggr+1)*n_per_chiral;

#pragma omp parallel for
	for(int site=0; site < num_sites; ++site) {
		const CBSite& cbsite = block_sitelist[site];
		const float *src_site_data = src.GetSiteDataPtr(cbsite.cb, cbsite.site);
		float *target_site_data = target.GetSiteDataPtr(cbsite.cb, cbsite.site);
		for(int colorspin=min_cspin; colorspin < max_cspin; ++colorspin) {
			target_site_data[ RE + n_complex*colorspin] = src_site_data[ RE + n_complex*colorspin ];
			target_site_data[ IM + n_complex*colorspin] = src_site_data[ IM + n_complex*colorspin ];
		}
	}
}

//! Extract the spins belonging to a given aggregate from QDP++ source vector src, into QDP++ target vector target
void extractAggregate(CoarseSpinor& target, const CoarseSpinor& src, int aggr )
{


	const LatticeInfo& info = src.GetInfo();
	AssertCompatible( info, target.GetInfo() );

	const int num_color = src.GetNumColor();
	const int n_per_chiral = ( info.GetNumSpins() == 4 ) ? 2*num_color : num_color;

	const int min_cspin = aggr*n_per_chiral;
	const int max_cspin = (aggr+1)*n_per_chiral;

	const int num_cbsites = info.GetNumCBSites();

#pragma omp parallel for collapse(2)
	for(int cb =0; cb < n_checkerboard; ++cb) {
		for(int site=0; site < num_cbsites; ++site) {

			const float *src_site_data = src.GetSiteDataPtr(cb, site);
			float *target_site_data = target.GetSiteDataPtr(cb, site);
			for(int colorspin=min_cspin; colorspin < max_cspin; ++colorspin) {
				target_site_data[ RE + n_complex*colorspin] = src_site_data[ RE + n_complex*colorspin ];
				target_site_data[ IM + n_complex*colorspin] = src_site_data[ IM + n_complex*colorspin ];
			}
		}
	}
}

//! Orthonormalize vecs over the spin aggregates within the sites
void orthonormalizeBlockAggregates(std::vector<std::shared_ptr<CoarseSpinor>>& vecs,
		const std::vector<Block>& block_list)
{
	int num_blocks = block_list.size();

#pragma omp parallel for collapse(2)
	for(int aggr=0; aggr < 2; ++aggr) {
		for(int block_id=0; block_id < num_blocks; block_id++) {

			const Block& block = block_list[block_id];

			// This will be over blocks...
			// do vecs[0] ... vecs[N]
			for(IndexType curr_vec=0; curr_vec < static_cast<IndexType>(vecs.size()); curr_vec++) {

				// orthogonalize against previous vectors
				// if curr_vec == 0 this will be skipped
				for(int prev_vec=0; prev_vec < curr_vec; prev_vec++) {

					std::complex<double> iprod = innerProductBlockAggr( *(vecs[prev_vec]), *(vecs[curr_vec]), block, aggr);
					std::complex<double> minus_iprod=std::complex<double>(-real(iprod), -imag(iprod) );

					// curr_vec <- curr_vec - <curr_vec|prev_vec>*prev_vec = -iprod*prev_vec + curr_vec
					caxpyBlockAggr( minus_iprod, *(vecs[prev_vec]), *(vecs[curr_vec]), block, aggr);
				}

				// Normalize current vector
				double inv_norm = ((double)1)/sqrt(norm2BlockAggr(*(vecs[curr_vec]), block, aggr));

				// vecs[curr_vec] = inv_norm * vecs[curr_vec]
				axBlockAggr(inv_norm, *(vecs[curr_vec]), block, aggr);
			}


		}	// block
	}// aggregates
}

//! 'Restrict' a QDP++ spinor to a CoarseSpinor with the same geometry
void restrictSpinor( const std::vector<Block>& blocklist, const std::vector< std::shared_ptr<CoarseSpinor> >& fine_vecs,
		const CoarseSpinor& fine_in, CoarseSpinor& coarse_out)
{

	const LatticeInfo& fine_info = fine_in.GetInfo();
	const LatticeInfo& coarse_info = coarse_out.GetInfo();

	const IndexType num_fine_spins = fine_info.GetNumSpins();
	const IndexType num_fine_colors =fine_info.GetNumColors();
	const IndexType num_fine_colorspins = num_fine_spins*num_fine_colors;

	const IndexType num_coarse_spins = coarse_info.GetNumSpins();
	const IndexType num_coarse_colors = coarse_info.GetNumColors();
	const IndexType num_coarse_colorspins = num_coarse_spins*num_coarse_colors;

	const IndexType num_coarse_cbsites = coarse_info.GetNumCBSites();

	// If the fine vector is 4 component, lump them together
	const int fine_n_per_chiral = ( num_fine_spins == 4 ) ? 2*num_fine_colors : num_fine_colors;

	// Sanity check. The number of sites in the coarse spinor
	// Has to equal the number of blocks
	//assert( n_checkerboard*num_coarse_cbsites == static_cast<IndexType>(blocklist.size()) );
	if ( n_checkerboard * num_coarse_cbsites !=  static_cast<IndexType>(blocklist.size()) ) {
		MasterLog(ERROR, "restrictSpinor error: n_checkerboard=%d num_coarse_cbsites=%d blocklist_size=%d",
				n_checkerboard,num_coarse_cbsites,blocklist.size());

	}
	// The number of vectors has to eaqual the number of coarse colors
	assert( static_cast<IndexType>(fine_vecs.size()) == num_coarse_colors );


	for(IndexType vec=0; vec < static_cast<IndexType>(fine_vecs.size()); ++vec) {
		AssertCompatible( fine_vecs[vec]->GetInfo(), fine_info );
	}

	// This will be a loop over blocks
#pragma omp parallel for collapse (4)
	for(int block_cb=0; block_cb < n_checkerboard; ++block_cb) {
	  for(int block_cbsite = 0; block_cbsite < num_coarse_cbsites; ++block_cbsite) {
	    for(int chiral = 0; chiral < 2; ++chiral ) {
	      for(int coarse_color=0; coarse_color  < num_coarse_colors; coarse_color++) {

	        IndexType block_idx = block_cbsite + block_cb*num_coarse_cbsites;

	        // The coarse site spinor is where we will write the result
	        float* coarse_site_spinor = coarse_out.GetSiteDataPtr(block_cb,block_cbsite);
	        int coarse_colorspin = coarse_color + chiral * num_coarse_colors;

	        // Identify the current block
	        const Block& block = blocklist[block_idx];

	        // Get the list of fine (cb,cbsite) pairs in the blocks
	        auto block_sitelist = block.getCBSiteList();

	        // and their number -- this is redundant, I could get it from block_sitelist.size()
	        auto num_sites_in_block = block.getNumSites();

	        float iprod_re=0;
	        float iprod_im=0;

	        for( IndexType fine_site_idx = 0; fine_site_idx < static_cast<IndexType>(num_sites_in_block); fine_site_idx++ ) {

	          // Find the fine site
	          const CBSite& fine_site = block_sitelist[fine_site_idx];
	          const float *ferm_in_site_data = fine_in.GetSiteDataPtr(fine_site.cb, fine_site.site);
	          const float *vec_in_site_data = fine_vecs[ coarse_color ]->GetSiteDataPtr(fine_site.cb, fine_site.site);


	          int min_fine_cspin = chiral*fine_n_per_chiral;
	          int max_fine_cspin = (chiral+1)*fine_n_per_chiral;

#pragma omp simd reduction(+:iprod_re,iprod_im)
	          for(int fine_colorspin=min_fine_cspin; fine_colorspin < max_fine_cspin; ++fine_colorspin) {


	            //REAL left_r = fine_v[ coarse_color ].elem(fine_site).elem(targ_spin).elem(color).real();
	            // REAL left_i = v[ coarse_color ].elem(fine_site).elem(targ_spin).elem(color).imag();
	            float left_r = vec_in_site_data[RE + n_complex*fine_colorspin];
	            float left_i = vec_in_site_data[IM + n_complex*fine_colorspin];


	            // REAL right_r = ferm_in.elem(fine_site).elem(targ_spin).elem(color).real();
	            // REAL right_i = ferm_in.elem(fine_site).elem(targ_spin).elem(color).imag();
	            float right_r = ferm_in_site_data[RE + n_complex*fine_colorspin];
	            float right_i = ferm_in_site_data[IM + n_complex*fine_colorspin];

	            // It is V_j^H  ferm_in so conj(left)*right.
	            iprod_re += left_r * right_r + left_i * right_i;
	            iprod_im += left_r * right_i - right_r * left_i;


	          }	 // fine-spincolor

	        } // fine site_idx
	        coarse_site_spinor[ RE + n_complex*coarse_colorspin ] = iprod_re;
	        coarse_site_spinor[ IM + n_complex*coarse_colorspin ] = iprod_im;
	      } // coarse_color
	    } // chiral component
	  } // block_cbsite
	} // block_cb
}

//! 'Prolongate' a CoarseSpinor to a QDP++ Fine Spinor
void prolongateSpinor(const std::vector<Block>& blocklist,
		const std::vector<std::shared_ptr<CoarseSpinor> >& fine_v,
		const CoarseSpinor& coarse_in, CoarseSpinor& fine_out)
{
		const LatticeInfo& fine_info = fine_out.GetInfo();
		const LatticeInfo& coarse_info = coarse_in.GetInfo();

		const IndexType num_fine_spins = fine_info.GetNumSpins();
		const IndexType num_fine_colors =fine_info.GetNumColors();
		const IndexType num_fine_colorspins = num_fine_spins*num_fine_colors;

		const IndexType num_coarse_spins = coarse_info.GetNumSpins();
		const IndexType num_coarse_colors = coarse_info.GetNumColors();
		const IndexType num_coarse_colorspins = num_coarse_spins*num_coarse_colors;

	// Prolongate in here
	IndexType num_coarse_cbsites=coarse_info.GetNumCBSites();
	assert( num_coarse_cbsites == static_cast<IndexType>(blocklist.size()/2) );


	assert( static_cast<IndexType>(fine_v.size()) == num_coarse_colors );

	for(unsigned int vecs = 0; vecs < fine_v.size(); ++vecs) {
		AssertCompatible( fine_v[vecs]->GetInfo(), fine_info);
	}

	IndexType n_fine_per_chiral = (num_fine_spins == 4 ) ? 2*num_fine_colors : num_fine_colors;


	// NB: Parallelism wise, this is a scatter. Because we are visiting each block
	// and keeping it fixed we write out all the fine sites in the block which will not
	// be contiguous. One potential optimization is to turn this into a gather...
	// Then we would need to loop through all the fine sites in order. Our current blocklist
	// Only contains coarse site -> list of fine sites
	// The inverse mapping of fine site-> block (many fine sites to same block)
	// does not exist. We could create it ...

	// Loop over the coarse sites (blocks)
	// Do this with checkerboarding, because of checkerboarded index for
	// coarse spinor
#pragma omp parallel for collapse(4)
	for(int block_cb = 0; block_cb < n_checkerboard; ++block_cb) {
	  for(int block_cbsite=0; block_cbsite < num_coarse_cbsites; ++block_cbsite ) {
	    for(int chiral =0; chiral < 2; ++chiral ) {
	      for(int fine_color=0; fine_color < n_fine_per_chiral; fine_color++ ) {

	        // Our block index is always block_cbsite + block_cb * num_coarse_cbsites
	        IndexType block_idx = block_cbsite + block_cb*num_coarse_cbsites;

	        const float *coarse_spinor = coarse_in.GetSiteDataPtr(block_cb,block_cbsite);

	        // Get the list of sites in the block
	        auto fine_sitelist = blocklist[block_idx].getCBSiteList();
	        auto num_fine_sitelist = blocklist[block_idx].getNumSites();

	        int fine_colorspin  = fine_color + n_fine_per_chiral*chiral;

	        // Loop over fine sites
	        for( unsigned int fine_site_idx = 0; fine_site_idx < num_fine_sitelist; ++fine_site_idx) {

	          const CBSite& cbsite = fine_sitelist[fine_site_idx];
	          float *fine_site_data = fine_out.GetSiteDataPtr( cbsite.cb, cbsite.site);

	          float csum_re =0;
	          float csum_im =0;

#pragma omp simd reduction(+:csum_re,csum_im)
	          for(int coarse_color = 0; coarse_color < num_coarse_colors; coarse_color++) {

	            const float* vec_in_data = fine_v[coarse_color]->GetSiteDataPtr( cbsite.cb, cbsite.site);

	            int coarse_colorspin = coarse_color + chiral*num_coarse_colors; // coarse_color(chiral) is the row

	            float left_r = vec_in_data[RE + n_complex*fine_colorspin]; // Fine_color(chiral) is row, coarse_color is column
	            float left_i = vec_in_data[IM + n_complex*fine_colorspin];

	            float right_r = coarse_spinor[ RE + n_complex*coarse_colorspin];
	            float right_i = coarse_spinor[ IM + n_complex*coarse_colorspin];

	            // V_j | out  (rather than V^{H}) so needs regular complex mult?
	            csum_re += left_r * right_r - left_i * right_i;
	            csum_im += left_i * right_r + left_r * right_i;

	          } // coarse color
	          fine_site_data[ RE + n_complex*fine_colorspin ] = csum_re;
	          fine_site_data[ IM + n_complex*fine_colorspin ] = csum_im;

	        } // fine site
	      } // fine color
	    } // chiral
	  } // coarse site
	} // coarse cb

}

//! Coarsen one direction of a 'dslash' link
void dslashTripleProductDir(const CoarseDiracOp& D_op,
		const std::vector<Block>& blocklist, int dir, const CoarseGauge& u,
		const std::vector<std::shared_ptr<CoarseSpinor> >& in_vecs,
		CoarseGauge& u_coarse)
{
	// Dslash triple product in here

	const IndexType num_coarse_colors = u_coarse.GetNumColor();
	const IndexType num_coarse_colorspin = u_coarse.GetNumColorSpin();
	const IndexType num_chiral = 2;
	const IndexType num_coarse_cbsites = u_coarse.GetInfo().GetNumCBSites();

	// in vecs has size Ncolor_c = num_coarse_colorspin/2
	// But this mixes both upper and lower spins
	// Once we deal with those separately we will need num_coarse_colorspin results
	// And we will need to apply the 'DslashDir' separately to each aggregate

	assert(static_cast<IndexType>(in_vecs.size()) == num_coarse_colors);
	const LatticeInfo& fine_info = in_vecs[0]->GetInfo();

	const int num_fine_colors = fine_info.GetNumColors();
	const int num_fine_spins = fine_info.GetNumSpins();
	int num_spincolor_per_chiral =
			(num_fine_spins == 4) ? 2 * num_fine_colors : num_fine_colors;


	// I will need to make a vector of spinors, to which I will have applied
	// Dslash Dir. These need the info
	const LatticeInfo& coarse_info = u_coarse.GetInfo();


	std::vector<std::shared_ptr<CoarseSpinor> > out_vecs(num_coarse_colorspin);
	for (int j = 0; j < num_coarse_colorspin; ++j) {
		out_vecs[j] = std::make_shared<CoarseSpinor>(fine_info);
		ZeroVec(*(out_vecs[j]));
	}

	CoarseSpinor tmp(fine_info);
	// Apply DslashDir to each aggregate separately.
	// DslashDir may mix spins with (1 +/- gamma_mu)
	for (int j = 0; j < num_coarse_colors; ++j) {
		for (int chiral = 0; chiral < 2; ++chiral) {
			ZeroVec(tmp);
			CoarseSpinor& out = *( out_vecs[chiral*num_coarse_colors + j] );

			extractAggregate(tmp, *(in_vecs[j]), chiral);

			for(int cb=0; cb < n_checkerboard; ++cb) {
#pragma omp parallel
				{
					int tid = omp_get_thread_num();
					D_op.DslashDir(out,	u, tmp, cb, dir, tid);
				}

			}	// end cb

		} // chiral
	} // j

	// Loop over the coarse sites (blocks)
	for (IndexType coarse_cb = 0; coarse_cb < n_checkerboard; ++coarse_cb) {
		for (IndexType coarse_cbsite = 0; coarse_cbsite < num_coarse_cbsites;
				++coarse_cbsite) {

			// Get a Block Index
			unsigned int block_idx = coarse_cbsite + coarse_cb * num_coarse_cbsites;

			const Block& block = blocklist[block_idx]; // Get teh block

			//  --------------------------------------------
			//  Now do the faces
			//  ---------------------------------------------
			{
				auto face_sitelist = block.getFaceList(dir);
				auto  num_sites = face_sitelist.size();



				float *coarse_link = u_coarse.GetSiteDirDataPtr(coarse_cb,coarse_cbsite, dir);

				// Do the accumulation in double
				std::vector<double> tmp_link(n_complex*num_coarse_colorspin*num_coarse_colorspin);
				for(int j=0; j < n_complex*num_coarse_colorspin*num_coarse_colorspin; ++j) {
						tmp_link[j] = 0;
				}

				for(IndexType fine_site_idx = 0; fine_site_idx < static_cast<IndexType>(num_sites); ++fine_site_idx) {

					CBSite& fine_cbsite = face_sitelist[ fine_site_idx ];

					for(int aggr_row=0; aggr_row < num_chiral; ++aggr_row) {
						for(int aggr_col=0; aggr_col <num_chiral; ++aggr_col ) {

							// This is an num_coarse_colors x num_coarse_colors matmul
							for(int matmul_row=0; matmul_row < num_coarse_colors; ++matmul_row) {
								for(int matmul_col=0; matmul_col < num_coarse_colors; ++matmul_col) {

									// Offset by the aggr_row and aggr_column
									int row = aggr_row*num_coarse_colors + matmul_row;
									int col = aggr_col*num_coarse_colors + matmul_col;

									//Index in coarse link
									int coarse_link_index = n_complex*(row+ num_coarse_colorspin*col);

									// Inner product loop
									for(int k=0; k < num_spincolor_per_chiral; ++k) {

											// [ V^H_upper   0      ] [  A_upper    B_upper ] = [ V^H_upper A_upper   V^H_upper B_upper  ]
											// [  0       V^H_lower ] [  A_lower    B_lower ]   [ V^H_lower A_lower   V^H_lower B_lower  ]
											//
											// NB: V^H_upper always multiplies an 'upper' (either A_upper or B_upper)
											//     V^H_lower always multiplies a 'lower' (either A_lower or B_lower)
											//
											// So there is no mixing of spins (ie: V^H_upper with B_lower etc)
											// So spins are decided by which portion of V^H we use, ie on aggr_row
											//
											// k / Nc maps to spin_component 0 or 1 in the aggregation
											// aggr_row*(Ns/2) offsets it to either upper or lower
											//
											int sc=k+aggr_row*num_spincolor_per_chiral;

											// Right vector
											//float right_r = (float)(out_vecs[col].elem(qdp_site).elem(spin).elem(color).real());
											//float right_i = (float)(out_vecs[col].elem(qdp_site).elem(spin).elem(color).imag());
											const float* right_vector_data = out_vecs[col]->GetSiteDataPtr(fine_cbsite.cb,fine_cbsite.site);
											const float right_r = right_vector_data[RE + sc*n_complex];
											const float right_i = right_vector_data[IM + sc*n_complex];

											// Left vector -- only num_coarse_colors components with [ V^H_upper V^H_lower ]
											//
											// ie a compact storage
											// rather than:
											//
											// [ V^H_upper   0      ]
											// [  0       V^H_lower ]
											//
											// so index with row % num_coarse_colors = matmul_row
											// float left_r = (float)(in_vecs[matmul_row].elem(qdp_site).elem(spin).elem(color).real());
											// float left_i = (float)(in_vecs[matmul_row].elem(qdp_site).elem(spin).elem(color).imag());
											const float* left_vector_data = in_vecs[matmul_row]->GetSiteDataPtr(fine_cbsite.cb, fine_cbsite.site);
											const float left_r = left_vector_data[RE + sc*n_complex] ;
											const float left_i = left_vector_data[IM + sc*n_complex] ;

											// Accumulate inner product V^H_row A_column
											tmp_link[RE + coarse_link_index ] += (double)(left_r*right_r + left_i*right_i);
											tmp_link[IM + coarse_link_index ] += (double)(left_r*right_i - right_r*left_i);
										} // k

									} // matmul_col
								} // matmul_row
							} // aggr_col
						} // aggr_row


					} // fine_site_idx

					for(int row=0; row < num_coarse_colorspin; ++row) {
						for(int col=0; col < num_coarse_colorspin; ++col) {
							int coarse_link_index = n_complex*(row + num_coarse_colorspin*col);
							coarse_link[RE + coarse_link_index ] += (float)( tmp_link[RE + coarse_link_index] );
							coarse_link[IM + coarse_link_index ] += (float)( tmp_link[IM + coarse_link_index ] );

						} // rows
					} // cols
				}

				//  --------------------------------------------
				//  Now do the Not Faces faces
				//  ---------------------------------------------
				{
					auto not_face_sitelist = block.getNotFaceList(dir);
					auto  num_sites = not_face_sitelist.size();


					// Get teh coarse site for writing
					// Thiis is fixed
					float *coarse_link = u_coarse.GetSiteDiagDataPtr(coarse_cb,coarse_cbsite);

					// Do the accumulation in double
					std::vector<double> tmp_link(n_complex*num_coarse_colorspin*num_coarse_colorspin);

					// Zero the link
					for(int row=0; row < num_coarse_colorspin; ++row) {
						for(int col=0; col < num_coarse_colorspin; ++col) {
							int coarse_link_index = n_complex*(row + num_coarse_colorspin*col);
							tmp_link[RE + coarse_link_index] = 0;
							tmp_link[IM + coarse_link_index] = 0;

						}
					}

					for(IndexType fine_site_idx = 0; fine_site_idx < static_cast<IndexType>(num_sites); ++fine_site_idx) {

						CBSite& fine_cbsite = not_face_sitelist[ fine_site_idx ];

						for(int aggr_row=0; aggr_row < num_chiral; ++aggr_row) {
							for(int aggr_col=0; aggr_col <num_chiral; ++aggr_col ) {

								// This is an num_coarse_colors x num_coarse_colors matmul
								for(int matmul_row=0; matmul_row < num_coarse_colors; ++matmul_row) {
									for(int matmul_col=0; matmul_col < num_coarse_colors; ++matmul_col) {

										// Offset by the aggr_row and aggr_column
										int row = aggr_row*num_coarse_colors + matmul_row;
										int col = aggr_col*num_coarse_colors + matmul_col;

										//Index in coarse link
										int coarse_link_index = n_complex*(row+ num_coarse_colorspin*col);

										// Inner product loop
										for(int k=0; k < num_spincolor_per_chiral; ++k) {


											// [ V^H_upper   0      ] [  A_upper    B_upper ] = [ V^H_upper A_upper   V^H_upper B_upper  ]
											// [  0       V^H_lower ] [  A_lower    B_lower ]   [ V^H_lower A_lower   V^H_lower B_lower  ]
											//
											// NB: V^H_upper always multiplies an 'upper' (either A_upper or B_upper)
											//     V^H_lower always multiplies a 'lower' (either A_lower or B_lower)
											//
											// So there is no mixing of spins (ie: V^H_upper with B_lower etc)
											// So spins are decided by which portion of V^H we use, ie on aggr_row
											//
											// k / Nc maps to spin_component 0 or 1 in the aggregation
											// aggr_row*(Ns/2) offsets it to either upper or lower
											//
											int sc=k+aggr_row*num_spincolor_per_chiral;

											// Right vector
											//float right_r = (float)(out_vecs[col].elem(qdp_site).elem(spin).elem(color).real());
											//float right_i = (float)(out_vecs[col].elem(qdp_site).elem(spin).elem(color).imag());
											const float* right_vector_data = out_vecs[col]->GetSiteDataPtr(fine_cbsite.cb,fine_cbsite.site);
											const float right_r = right_vector_data[RE + sc*n_complex];
											const float right_i = right_vector_data[IM + sc*n_complex];

											// Left vector -- only num_coarse_colors components with [ V^H_upper V^H_lower ]
											//
											// ie a compact storage
											// rather than:
											//
											// [ V^H_upper   0      ]
											// [  0       V^H_lower ]
											//
											// so index with row % num_coarse_colors = matmul_row
											// float left_r = (float)(in_vecs[matmul_row].elem(qdp_site).elem(spin).elem(color).real());
											// float left_i = (float)(in_vecs[matmul_row].elem(qdp_site).elem(spin).elem(color).imag());
											const float* left_vector_data = in_vecs[matmul_row]->GetSiteDataPtr(fine_cbsite.cb, fine_cbsite.site);
											const float left_r = left_vector_data[RE + sc*n_complex] ;
											const float left_i = left_vector_data[IM + sc*n_complex] ;

											// Accumulate inner product V^H_row A_column
											tmp_link[RE + coarse_link_index ] += (double)(left_r*right_r + left_i*right_i);
											tmp_link[IM + coarse_link_index ] += (double)(left_r*right_i - right_r*left_i);
										} // k
									} // matmul_col
								} // matmul_row
							} // aggr_col
						} // aggr_row
					} // fine_site_idx

					for(int row=0; row < num_coarse_colorspin; ++row) {
						for(int col=0; col < num_coarse_colorspin; ++col) {
							int coarse_link_index = n_complex*(row + num_coarse_colorspin*col);
							coarse_link[RE + coarse_link_index ] += (float)( tmp_link[RE + coarse_link_index] );
							coarse_link[IM + coarse_link_index ] += (float)( tmp_link[IM + coarse_link_index ] );

						} // rows
					} // cols
				}

		} // coarse_cbsite
	} // coarse_cb

}

//! Coarsen the clover term (1 block = 1 site )
void clovTripleProduct(const CoarseDiracOp& D_op,
		const std::vector<Block>& blocklist,
		const CoarseGauge& fine_gauge_clov,
		const std::vector<std::shared_ptr<CoarseSpinor> >& in_fine_vecs,
		CoarseGauge& coarse_gauge_clov)
{
	// Clover Triple product in here
	const IndexType num_coarse_colors = coarse_gauge_clov.GetNumColor();
	const IndexType num_coarse_colorspin = coarse_gauge_clov.GetNumColorSpin();

	const IndexType num_chiral_components = 2;
	const LatticeInfo& coarse_info = coarse_gauge_clov.GetInfo();
	const IndexType num_coarse_cbsites = coarse_info.GetNumCBSites();

	// in vecs has size num_coarse_colors = Ncolorspin_c/2
	// But this mixes both upper and lower spins
	// Once we deal with those separately we will need Ncolorspin_c results
	// And we will need to apply the 'DslashDir' separately to each aggregate

	assert(static_cast<IndexType>(in_fine_vecs.size()) == num_coarse_colors);
	assert(num_chiral_components == 2);

	const LatticeInfo& fine_info = in_fine_vecs[0]->GetInfo();
	const int num_fine_colors = fine_info.GetNumColors();
	const int num_fine_spins = fine_info.GetNumSpins();

	const int num_spincolor_per_chiral = (num_fine_spins == 4 ) ? 2*num_fine_colors : num_fine_colors;

	// out_vecs is the result of applying clover term to in_vecs
	// NOTE!!!: Unlike with Dslash where (1 +/- gamma_mu) mixes the upper and lower spin components
	// Clover *does not* do this. In this chiral basis that we use Clover is block diagonal
	// So it acts independently on upper and lower spin components.
	// This means Ncolor vectors are sufficient. The upper components will hold the results of
	// clover_term applied to the upper components while the lower components will hold the results of
	// clover_term applied to the lower components in the same way in_vector combines upper and lower
	// components.
	std::vector<std::shared_ptr<CoarseSpinor>> out_vecs(num_coarse_colorspin);

	// Zero the output
	for (int j = 0; j < num_coarse_colorspin; ++j) {
		out_vecs[j] = std::make_shared<CoarseSpinor>(fine_info);
		ZeroVec(*(out_vecs[j]));
	}

	for (int j = 0; j < num_coarse_colors; ++j) {
		for(int chiral =0; chiral < num_chiral_components; ++chiral ) {

			// Extract the chiral component into a temporary
			CoarseSpinor tmp(fine_info);
			ZeroVec(tmp);
			extractAggregate(tmp, *(in_fine_vecs[j]), chiral);
#pragma omp parallel
			{
				int tid = omp_get_thread_num();
				for (int cb = 0; cb < 2; ++cb) {
					D_op.CloverApply(*(out_vecs[j+chiral*num_coarse_colors]), fine_gauge_clov, tmp, cb, LINOP_OP, tid);
				}
			}
		}

	}

	// Technically these outer loops should be over all the blocks.
	for (int coarse_cb = 0; coarse_cb < n_checkerboard; ++coarse_cb) {
		for (int coarse_cbsite = 0; coarse_cbsite < num_coarse_cbsites;
				++coarse_cbsite) {


			// Convert the CB site into a block index
			int block_idx = coarse_cbsite + coarse_cb * num_coarse_cbsites;
			const Block& block = blocklist[block_idx];

			// Get the sitelist in the block (all sites)
			auto fine_sitelist = block.getCBSiteList();
			auto num_fine_sites = block.getNumSites();


			// Get the coarse clover -- this is the 'local' (dir=8) of the gauge field
			// Which is an N_colorspin x N_colorspin
			// Matrix. Caller must initialize this as both the Dslash Dir and this function
			// write into it.
			float *coarse_clov_data = coarse_gauge_clov.GetSiteDiagDataPtr(coarse_cb, coarse_cbsite);

			// Accumulate into a temporary and zero that out
			std::vector<double> tmp_result(n_complex*num_coarse_colorspin*num_coarse_colorspin);
			for(int j=0; j < n_complex*num_coarse_colorspin*num_coarse_colorspin; ++j) {
				tmp_result[j]=(double)0;
			}

			for (IndexType fine_site_idx = 0; fine_site_idx < static_cast<IndexType>(num_fine_sites);
							++fine_site_idx) {

				auto fine_site = fine_sitelist[fine_site_idx];

				for(int aggr_row=0; aggr_row < num_chiral_components; ++aggr_row) {
					for(int aggr_col=0; aggr_col <num_chiral_components; ++aggr_col ) {

						// This is an num_coarse_colors x num_coarse_colors matmul
						for(int matmul_row=0; matmul_row < num_coarse_colors; ++matmul_row) {
							for(int matmul_col=0; matmul_col < num_coarse_colors; ++matmul_col) {

								// Offset by the aggr_row and aggr_column
								int row = aggr_row*num_coarse_colors + matmul_row;
								int col = aggr_col*num_coarse_colors + matmul_col;

								//Index in coarse link
								int coarse_link_index = n_complex*(row + num_coarse_colorspin*col);

								// Inner product loop
								for(int k=0; k < num_spincolor_per_chiral; ++k) {

									int sc=k+aggr_row*num_spincolor_per_chiral;

									const float* right_vector_data = out_vecs[col]->GetSiteDataPtr(fine_site.cb,fine_site.site);
									const float right_r = right_vector_data[RE + sc*n_complex];
									const float right_i = right_vector_data[IM + sc*n_complex];

									// Left vector -- only num_coarse_colors components with [ V^H_upper V^H_lower ]
									//
									// ie a compact storage
									// rather than:
									//
									// [ V^H_upper   0      ]
									// [  0       V^H_lower ]
									//
									// so index with row % num_coarse_colors = matmul_row
									//float left_r = (float)(in_vecs[matmul_row].elem(qdp_site).elem(spin).elem(color).real());
									//float left_i = (float)(in_vecs[matmul_row].elem(qdp_site).elem(spin).elem(color).imag());
									const float* left_vector_data = in_fine_vecs[matmul_row]->GetSiteDataPtr(fine_site.cb, fine_site.site);
									const float left_r = left_vector_data[ RE + sc*n_complex];
									const float left_i = left_vector_data[ IM + sc*n_complex];

									// Accumulate inner product V^H_row A_column
									tmp_result[RE + coarse_link_index ] += (double)(left_r*right_r + left_i*right_i);
									tmp_result[IM + coarse_link_index ] += (double)(left_r*right_i - right_r*left_i);
								} // k

							} // matmul_col
						} // matmul_row
					} // aggr_col
				} // aggr_row
			}// fine site idx

			for(int row=0; row < num_coarse_colorspin; ++row) {
				for(int col=0; col < num_coarse_colorspin; ++col) {
					int coarse_link_index = n_complex*(row + num_coarse_colorspin*col);
					coarse_clov_data[RE + coarse_link_index ] += (float)( tmp_result[RE + coarse_link_index] );
					coarse_clov_data[IM + coarse_link_index ] += (float)( tmp_result[IM + coarse_link_index ] );

				} // rows
			} // cols


		} // coarse_site
	} // coarse_cb
}



// MatrixXcf is the Eigen Matrix class
// Invert the diagonal part of u, into eo_clov
using ComplexMatrix = Matrix<std::complex<float>, Dynamic, Dynamic, ColMajor>;

void invertCloverDiag(CoarseGauge& u)
{
	MasterLog(INFO, "Inverting Coarse Diagonal Term");

	const LatticeInfo& info = u.GetInfo();
	const int num_cbsites = info.GetNumCBSites();
	const int num_colorspins = info.GetNumColorSpins();

#pragma omp parallel for collapse(2)
	for(int cb = 0; cb < n_checkerboard; ++cb) {
		for(int cbsite=0; cbsite < num_cbsites; ++cbsite) {

			// 0-7 are the off diags, 8 is the diag
			float *diag_site_data = u.GetSiteDiagDataPtr(cb,cbsite);
			float *invdiag_site_data = u.GetSiteInvDiagDataPtr(cb,cbsite);


			Map< ComplexMatrix > in_mat(reinterpret_cast<std::complex<float>*>(diag_site_data),
					num_colorspins,
					num_colorspins);
			Map< ComplexMatrix > out_mat(reinterpret_cast<std::complex<float>*>(invdiag_site_data),
					num_colorspins,
					num_colorspins);

			out_mat = in_mat.inverse();
		} // sites
	} // checkerboards
}

// Multiply the inverse part of the clover into eo_clov
void multInvClovOffDiagLeft(CoarseGauge& u)
{
	MasterLog(INFO, "Computing A^{-1}D links");
	const LatticeInfo& info = u.GetInfo();
	const int num_cbsites = info.GetNumCBSites();
	const int num_colorspins = info.GetNumColorSpins();

#pragma omp parallel for collapse(2)
	for(int cb = 0; cb < n_checkerboard; ++cb) {
		for(int cbsite=0; cbsite < num_cbsites; ++cbsite) {

			// 0-7 are the off diags, 8 is the diag
			float *invdiag_site_data = u.GetSiteInvDiagDataPtr(cb,cbsite);
			for(int mu=0; mu < 8; ++mu) {
				float *resptr = u.GetSiteDirADDataPtr(cb,cbsite,mu);
				float *srcptr = u.GetSiteDirDataPtr(cb,cbsite,mu);


			Map< ComplexMatrix > invdiag_mat(reinterpret_cast<std::complex<float>*>(invdiag_site_data),
					num_colorspins,
					num_colorspins);

			Map< ComplexMatrix > srcmat(reinterpret_cast<std::complex<float>*>(srcptr),
					num_colorspins,
					num_colorspins);

			Map< ComplexMatrix > resmat(reinterpret_cast<std::complex<float>*>(resptr),
								num_colorspins,
								num_colorspins);

			resmat = invdiag_mat * srcmat;
			}
		} // sites
	} // checkerboards
}

template<typename T>
struct InvDiagAccessor {
	static
	inline const float* get(const T& in, int cb, int cbsite, int dir, int fb);
};

template<>
inline
const float*
InvDiagAccessor<CoarseGauge>::get(const CoarseGauge& in, int cb, int cbsite, int dir, int fb)
{
    return in.GetSiteInvDiagDataPtr(cb,cbsite);
}

// Multiply the inverse part of the clover into eo_clov
void multInvClovOffDiagRight(CoarseGauge& u)
{

	MasterLog(INFO, "Computing A^{-1}D links");
	const LatticeInfo& info = u.GetInfo();

	// Halo Buffers
	HaloContainer<CoarseGauge> gauge_halo_cb0(info);
	HaloContainer<CoarseGauge> gauge_halo_cb1(info);

	// This is we can use in a checkerboarded loop
	HaloContainer<CoarseGauge>* halos[2] = { &gauge_halo_cb0, &gauge_halo_cb1 };

	// Communicate all the halos
	for(int target_cb=0; target_cb < 2; ++target_cb) {
		MasterLog(INFO, "Communicating Gauge Halos: target_cb=%d", target_cb);
		CommunicateHaloSync<CoarseGauge,InvDiagAccessor>(*(halos[target_cb]), u, target_cb);
	}

	const int num_cbsites = info.GetNumCBSites();
	const int num_colorspins = info.GetNumColorSpins();

#pragma omp parallel for collapse(2)
	for(int target_cb = 0; target_cb < n_checkerboard; ++target_cb) {
		for(int cbsite=0; cbsite < num_cbsites; ++cbsite) {

			// 0-7 are the off diags, 8 is the diag
			for(int mu=0; mu < 8; ++mu) {

				// This goes to the DA array
				float *resptr = u.GetSiteDirDADataPtr(target_cb,cbsite,mu);

				// The original link (the D part)
				const float *srcptr = u.GetSiteDirDataPtr(target_cb,cbsite,mu);

				// This should get the neighboring A from either u or the halos as needed
				// It ought to have been already inverted by a previous call.
				const float *invdiag_ptr =GetNeighborDir<CoarseGauge,InvDiagAccessor>(*(halos[target_cb]), u, mu, target_cb, cbsite);

				// Dress them up as eigen matrices.
				Map< const ComplexMatrix > invdiag_mat(reinterpret_cast<const std::complex<float>*>(invdiag_ptr),
						num_colorspins,
						num_colorspins);

				Map< const ComplexMatrix > srcmat(reinterpret_cast<const std::complex<float>*>(srcptr),
						num_colorspins,
						num_colorspins);

				Map< ComplexMatrix > resmat(reinterpret_cast<std::complex<float>*>(resptr),
						num_colorspins,
						num_colorspins);

				// Perform the right multiply.
				resmat = srcmat*invdiag_mat;
			}
		} // sites
	} // checkerboards
}

}; // Namespace

