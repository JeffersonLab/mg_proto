/*
 * aggregate_qdpxx.cpp
 *
 *  Created on: Dec 9, 2016
 *      Author: bjoo
 */
#include "aggregate_qdpxx.h"
#include "aggregate_block_coarse.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/geometry_utils.h"
#include "lattice/coarse/coarse_l1_blas.h"


using namespace QDP;
using namespace MG;

namespace MGTesting {


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
void orthonormalizeBlockAggregates(std::vector<std::shared_ptr<CoarseSpinor>>& vecs, const std::vector<Block>& block_list)
{
	int num_blocks = block_list.size();

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
	const IndexType num_fine_spins = fine_info.GetNumSpins();

	const IndexType num_coarse_cbsites = coarse_out.GetInfo().GetNumCBSites();
	const IndexType num_coarse_colors = coarse_out.GetNumColor();

	// Sanity check. The number of sites in the coarse spinor
	// Has to equal the number of blocks
	assert( n_checkerboard*num_coarse_cbsites == static_cast<IndexType>(blocklist.size()) );

	// The number of vectors has to eaqual the number of coarse colors
	assert( static_cast<IndexType>(fine_vecs.size()) == num_coarse_colors );
	for(IndexType vec=0; vec < static_cast<IndexType>(fine_vecs.size()); ++vec) {
		AssertCompatible( fine_vecs[vec]->GetInfo(), fine_info );
	}

	// This will be a loop over blocks
	for(int block_cb=0; block_cb < n_checkerboard; ++block_cb) {
		for(int block_cbsite = 0; block_cbsite < num_coarse_cbsites; ++block_cbsite) {
			IndexType block_idx = block_cbsite + block_cb*num_coarse_cbsites;

			// The coarse site spinor is where we will write the result
			float* coarse_site_spinor = coarse_out.GetSiteDataPtr(block_cb,block_cbsite);


			// Identify the current block
			const Block& block = blocklist[block_idx];

			// Get the list of fine (cb,cbsite) pairs in the blocks
			auto block_sitelist = block.getCBSiteList();

			// and their number -- this is redundant, I could get it from block_sitelist.size()
			auto num_sites_in_block = block.getNumSites();


			// Zero the accumulation in the current site
			for(int chiral = 0; chiral < 2; ++chiral ) {
				for(int coarse_color=0; coarse_color  < num_coarse_colors; coarse_color++) {
					int coarse_colorspin = coarse_color + chiral * num_coarse_colors;
					coarse_site_spinor[ RE + n_complex*coarse_colorspin ] = 0;
					coarse_site_spinor[ IM + n_complex*coarse_colorspin ] = 0;
				}
			}

			// Our loop is over coarse_colors and chiralities -- to fill out the colorspin components
			// However, each colorspin component will involve a site loop, and we can compute the contributions
			// to both chiralities of the color component from a vector in a single loop. So I put the fine site
			// loop outside of the chirality loop.
			//
			// An optimization/stabilization will be to  accumulate these loops in double since they are
			// over potentially large number of sites (e.g. 4^4)

			// Remember that coarse color picks the vector so have this outermost now,
			// Since we will be working in a vector at a time
			for(int coarse_color=0; coarse_color  < num_coarse_colors; coarse_color++) {

				// Now aggregate over all the sites in the block -- this will be over a single vector...
				// NB: The loop indices may be later rerolled, e.g. if we can restrict multiple vectors at once
				// Then having the coarse_color loop inner will be better.
				for( IndexType fine_site_idx = 0; fine_site_idx < static_cast<IndexType>(num_sites_in_block); fine_site_idx++ ) {

					// Find the fine site
					const CBSite& fine_site = block_sitelist[fine_site_idx];
					const float *ferm_in_site_data = fine_in.GetSiteDataPtr(fine_site.cb, fine_site.site);
					float *vec_in_site_data = fine_vecs[ coarse_color ]->GetSiteDataPtr(fine_site.cb, fine_site.site);

					// Now loop over the chiral components. These are local in a site at the level of spin
					for(int chiral = 0; chiral < 2; ++chiral ) {

						// Identify the color spin component we are accumulating
						int coarse_colorspin = coarse_color + chiral * num_coarse_colors;

						// If the fine vector is 4 component, lump them together
						int fine_n_per_chiral = ( num_fine_spins == 4 ) ? 2*num_coarse_colors : num_coarse_colors;
						int min_fine_cspin = chiral*fine_n_per_chiral;
						int max_fine_cspin = (chiral+1)*fine_n_per_chiral;
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
							coarse_site_spinor[ RE + n_complex*coarse_colorspin ] += left_r * right_r + left_i * right_i;
							coarse_site_spinor[ IM + n_complex*coarse_colorspin ] += left_r * right_i - right_r * left_i;


						}	 // fine-spincolor
					} // chiral component
				} // fine site_idx
			} // coarse_color
		} // block_cbsite
	} // block_cb
}

//! 'Prolongate' a CoarseSpinor to a QDP++ Fine Spinor
void prolongateSpinor(const std::vector<Block>& blocklist,
		const std::vector<std::shared_ptr<CoarseSpinor> >& fine_v,
		const CoarseSpinor& coarse_in, CoarseSpinor& fine_out)
{
	// Prolongate in here
	IndexType num_coarse_cbsites=coarse_in.GetInfo().GetNumCBSites();
	assert( num_coarse_cbsites == static_cast<IndexType>(blocklist.size()/2) );

	IndexType num_coarse_color = coarse_in.GetNumColor();
	assert( static_cast<IndexType>(fine_v.size()) == num_coarse_color );

	const LatticeInfo& fine_info = fine_out.GetInfo();
	for(unsigned int vecs = 0; vecs < fine_v.size(); ++vecs) {
		AssertCompatible( fine_v[vecs]->GetInfo(), fine_info);
	}

	IndexType num_fine_colors = fine_info.GetNumColors();
	IndexType num_fine_spins = fine_info.GetNumSpins();
	IndexType n_per_chiral = (num_fine_spins == 4 ) ? 2*num_fine_colors : num_fine_colors;


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
	for(int block_cb = 0; block_cb < n_checkerboard; ++block_cb) {
		for(int block_cbsite=0; block_cbsite < num_coarse_cbsites; ++block_cbsite ) {

			// Our block index is always block_cbsite + block_cb * num_coarse_cbsites
			IndexType block_idx = block_cbsite + block_cb*num_coarse_cbsites;

			const float *coarse_spinor = coarse_in.GetSiteDataPtr(block_cb,block_cbsite);

			// Get the list of sites in the block
			auto fine_sitelist = blocklist[block_idx].getCBSiteList();
			auto num_fine_sitelist = blocklist[block_idx].getNumSites();


			for( unsigned int fine_site_idx = 0; fine_site_idx < num_fine_sitelist; ++fine_site_idx) {
				const CBSite& cbsite = fine_sitelist[fine_site_idx];

				float *fine_site_data = fine_out.GetSiteDataPtr( cbsite.cb, cbsite.site);



				//for(int fine_spin=0; fine_spin < Ns; ++fine_spin) {
				for(int chiral =0; chiral < 2; ++chiral ) {

					for(int fine_color=0; fine_color < n_per_chiral; fine_color++ ) {

						//fine_out.elem(qdpsite).elem(fine_spin).elem(fine_color).real() = 0;
						//fine_out.elem(qdpsite).elem(fine_spin).elem(fine_color).imag() = 0;
						int fine_colorspin  = fine_color + n_per_chiral*chiral;

						fine_site_data[ RE + n_complex*fine_colorspin ] = 0;
						fine_site_data[ IM + n_complex*fine_colorspin ] = 0;

						for(int coarse_color = 0; coarse_color < num_coarse_color; coarse_color++) {

							//REAL left_r = v[coarse_color].elem(qdpsite).elem(fine_spin).elem(fine_color).real();
							// REAL left_i = v[coarse_color].elem(qdpsite).elem(fine_spin).elem(fine_color).imag();
							const float* vec_in_data = fine_v[coarse_color]->GetSiteDataPtr( cbsite.cb, cbsite.site);

							float left_r = vec_in_data[RE + n_complex*fine_colorspin]; // Fine_color(chiral) is row, coarse_color is column
							float left_i = vec_in_data[IM + n_complex*fine_colorspin];

							int coarse_colorspin = coarse_color + chiral*num_coarse_color; // coarse_color(chiral) is the row
							float right_r = coarse_spinor[ RE + n_complex*coarse_colorspin];
							float right_i = coarse_spinor[ IM + n_complex*coarse_colorspin];

							// V_j | out  (rather than V^{H}) so needs regular complex mult?
							//fine_out.elem(qdpsite).elem(fine_spin).elem(fine_color).real() += left_r * right_r - left_i * right_i;
							// fine_out.elem(qdpsite).elem(fine_spin).elem(fine_color).imag() += left_i * right_r + left_r * right_i;
							fine_site_data[ RE + n_complex*fine_colorspin ] += left_r * right_r - left_i * right_i;
							fine_site_data[ IM + n_complex*fine_colorspin ] += left_i * right_r + left_r * right_i;
						}
					}
				}
			}
		}
	}

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

	int n_per_chiral =
			(num_fine_spins == 4) ? 2 * num_fine_colors : num_fine_colors;

	// I will need to make a vector of spinors, to which I will have applied
	// Dslash Dir. These need the info
	const LatticeInfo& coarse_info = u_coarse.GetInfo();

	std::vector<std::shared_ptr<CoarseSpinor> > out_vecs(num_coarse_colorspin);

	// Hit in_vecs with DslashDir -- this leaves
	for (int j = 0; j < num_coarse_colorspin; ++j) {
		out_vecs[j] = std::make_shared<CoarseSpinor>(coarse_info);
		ZeroVec(*(out_vecs[j]));
	}

	// Apply DslashDir to each aggregate separately.
	// DslashDir may mix spins with (1 +/- gamma_mu)
	for (int j = 0; j < num_coarse_colors; ++j) {
		for (int chiral = 0; chiral < 2; ++chiral) {
			CoarseSpinor tmp(fine_info);

			extractAggregate(tmp, *(in_vecs[j]), chiral);
#pragma omp parallel
			{
				int tid = omp_get_thread_num();
				for (int cb = 0; cb < n_checkerboard; ++cb) {
					D_op.DslashDir(*(out_vecs[chiral * num_coarse_colors + j]),
							u, tmp, cb, dir, tid);
				}
			}	// end parallel
		} // chiral
	} // j

	// Loop over the coarse sites (blocks)
	for (IndexType coarse_cb = 0; coarse_cb < n_checkerboard; ++coarse_cb) {
		for (IndexType coarse_cbsite = 0; coarse_cbsite < num_coarse_cbsites;
				++coarse_cbsite) {

			// Get a Block Index
			unsigned int block_idx = coarse_cbsite
					+ coarse_cb * num_coarse_cbsites;
			const Block& block = blocklist[block_idx]; // Get teh block
			auto block_sitelist = block.getCBSiteList();
			auto num_block_sites = block.getNumSites();

			// Get teh coarse site for writing
			float *coarse_link = u_coarse.GetSiteDirDataPtr(coarse_cb,
					coarse_cbsite, dir);

			std::vector<double> tmp_link(
					n_complex * num_coarse_colorspin * num_coarse_colorspin);

			// Zero the link
			for (int row = 0; row < num_coarse_colorspin; ++row) {
				for (int col = 0; col < num_coarse_colorspin; ++col) {
					int coarse_link_index = n_complex
							* (col + num_coarse_colorspin * row);
					tmp_link[RE + coarse_link_index] = 0;
					tmp_link[IM + coarse_link_index] = 0;

				}
			}

			// Loop through the sites of the block
			for (IndexType fine_site_idx = 0; fine_site_idx < static_cast<IndexType>(num_block_sites);
					++fine_site_idx) {

				// Pick the cbsite
				const CBSite& cbsite = block_sitelist[fine_site_idx];

				// Matrix mutliply in chiral space.
				for (int aggr_row = 0; aggr_row < num_chiral; ++aggr_row) {
					for (int aggr_col = 0; aggr_col < num_chiral; ++aggr_col) {

						// This is an num_coarse_colors x num_coarse_colors matmul
						for (int matmul_row = 0; matmul_row < num_coarse_colors;
								++matmul_row) {
							float *invec_site_data =
									in_vecs[matmul_row]->GetSiteDataPtr(
											cbsite.cb, cbsite.site);

							for (int matmul_col = 0;
									matmul_col < num_coarse_colors;
									++matmul_col) {

								// Offset by the aggr_row and aggr_column
								int row = aggr_row * num_coarse_colors
										+ matmul_row;
								int col = aggr_col * num_coarse_colors
										+ matmul_col;

								float *outvec_site_data =
										out_vecs[col]->GetSiteDataPtr(cbsite.cb,
												cbsite.site);

								//Index in coarse link
								int coarse_link_index = n_complex
										* (col + num_coarse_colorspin * row);

								// Inner product loop
								for (int k = 0; k < n_per_chiral; ++k) {

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
									// But I don't need this now  since I am working with coarse types.
									// so I can just use k to map in the aggregation as intendede
									// and aggr_row to map as upper or lower

									int fine_spincolor = k
											+ aggr_row * n_per_chiral;

									// Right vector
									REAL64 right_r = outvec_site_data[RE
																	  + n_complex * fine_spincolor];
									REAL64 right_i = outvec_site_data[IM
																	  + n_complex * fine_spincolor];

									// Left vector -- only num_coarse_colors components with [ V^H_upper V^H_lower ]
									//
									// ie a compact storage
									// rather than:
									//
									// [ V^H_upper   0      ]
									// [  0       V^H_lower ]
									//
									// so index with row % num_coarse_colors = matmul_row
									// REAL64 left_r = in_vecs[matmul_row].elem(site).elem(spin).elem(color).real();
									// REAL64 left_i = in_vecs[matmul_row].elem(site).elem(spin).elem(color).imag();
									REAL64 left_r = invec_site_data[RE
																	+ n_complex * fine_spincolor];
									REAL64 left_i = invec_site_data[IM
																	+ n_complex * fine_spincolor];

									// Accumulate inner product V^H_row A_column
									tmp_link[RE + coarse_link_index] += left_r
											* right_r + left_i * right_i;
									tmp_link[IM + coarse_link_index] += left_r
											* right_i - right_r * left_i;
								} // k

							} // matmul_col
						} // matmul_row
					} // aggr_col
				} // aggr_row
			} // fine_site_idx
			// Zero the link

			for (int row = 0; row < num_coarse_colorspin; ++row) {
				for (int col = 0; col < num_coarse_colorspin; ++col) {
					int coarse_link_index = n_complex
							* (col + num_coarse_colorspin * row);
					coarse_link[RE + coarse_link_index] = tmp_link[RE
																   + coarse_link_index];
					coarse_link[IM + coarse_link_index] = tmp_link[IM
																   + coarse_link_index];

				}
			}

		} // coarse_cbsite
	} // coarse_cb

}

//! Coarsen the clover term (1 block = 1 site )
void clovTripleProduct(const CoarseDiracOp& D_op,
		const std::vector<Block>& blocklist, const CoarseClover& fine_clov,
		const std::vector<std::shared_ptr<CoarseSpinor> >& in_fine_vecs,
		CoarseClover& coarse_clov)
{
	// Clover Triple product in here
	const IndexType num_coarse_colors = coarse_clov.GetNumColor();
	const IndexType num_chiral_components = coarse_clov.GetNumChiral();
	const LatticeInfo& coarse_info = coarse_clov.GetInfo();
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

	const int n_per_chiral = (num_fine_spins == 4 ) ? 2*num_fine_colors : num_fine_colors;

	// out_vecs is the result of applying clover term to in_vecs
	// NOTE!!!: Unlike with Dslash where (1 +/- gamma_mu) mixes the upper and lower spin components
	// Clover *does not* do this. In this chiral basis that we use Clover is block diagonal
	// So it acts independently on upper and lower spin components.
	// This means Ncolor vectors are sufficient. The upper components will hold the results of
	// clover_term applied to the upper components while the lower components will hold the results of
	// clover_term applied to the lower components in the same way in_vector combines upper and lower
	// components.
	std::vector<std::shared_ptr<CoarseSpinor>> out_vecs(num_coarse_colors);

	// Zero the output
	for (int j = 0; j < num_coarse_colors; ++j) {
		out_vecs[j] = std::make_shared<CoarseSpinor>(fine_info);
		ZeroVec(*(out_vecs[j]));
	}

	// for each in-vector pull out respectively the lower and upper spins
	// multiply by clover and store in out_vecs. There will be num_coarse_colors*num_chiral_components output
	// vectors
	for (int j = 0; j < num_coarse_colors; ++j) {

		// Clover term is block diagonal
		// So I can apply it once, the upper and lower spin components will
		// be acted on independently. No need to separate the aggregates before
		// applying
		CoarseSpinor tmp(fine_info);

#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			for (int cb = 0; cb < 2; ++cb) {
				D_op.CloverApply(*(out_vecs[j]), fine_clov, *(in_fine_vecs[j]),
						cb, LINOP_OP, tid);
			}
		}

	}

	// Technically these outer loops should be over all the blocks.
	for (int coarse_cb = 0; coarse_cb < n_checkerboard; ++coarse_cb) {
		for (int coarse_cbsite = 0; coarse_cbsite < num_coarse_cbsites;
				++coarse_cbsite) {

			int block_idx = coarse_cbsite + coarse_cb * num_coarse_cbsites;

			const Block& block = blocklist[block_idx];
			auto block_sitelist = block.getCBSiteList();
			auto num_block_sites = block.getNumSites();

			// Zero out both chiral blocks of the result
			for (int chiral = 0; chiral < num_chiral_components; ++chiral) {
				// This is now an Ncomplex*num_coarse_colors x Ncomplex*num_coarse_colors
				float *coarse_clov_data = coarse_clov.GetSiteChiralDataPtr(
						coarse_cb, coarse_cbsite, chiral);

				// Zero the link
				for (int row = 0; row < num_coarse_colors; ++row) {
					for (int col = 0; col < num_coarse_colors; ++col) {
						int coarse_clov_index = n_complex
								* (col + num_coarse_colors * row);
						coarse_clov_data[RE + coarse_clov_index] = 0;
						coarse_clov_data[IM + coarse_clov_index] = 0;

					}
				}
			}

			for (IndexType fine_site_idx = 0; fine_site_idx < static_cast<IndexType>(num_block_sites);
					++fine_site_idx) {
				const CBSite& cbsite = block_sitelist[fine_site_idx];

				for (int chiral = 0; chiral < num_chiral_components; ++chiral) {

					// This is now an Ncomplex*num_coarse_colors x Ncomplex*num_coarse_colors
					float *coarse_clov_data = coarse_clov.GetSiteChiralDataPtr(
							coarse_cb, coarse_cbsite, chiral);

					// This is an num_coarse_colors x num_coarse_colors matmul
					for (int matmul_row = 0; matmul_row < num_coarse_colors;
							++matmul_row) {

						float* invecs_site_data =
								in_fine_vecs[matmul_row]->GetSiteDataPtr(
										cbsite.cb, cbsite.site);

						for (int matmul_col = 0; matmul_col < num_coarse_colors;
								++matmul_col) {

							//Index in coarse link
							int coarse_clov_index = n_complex
									* (matmul_col
											+ num_coarse_colors * matmul_row);
							float* outvecs_site_data =
									out_vecs[matmul_col]->GetSiteDataPtr(
											cbsite.cb, cbsite.site);

							// Inner product loop
							for (int k = 0; k < n_per_chiral; ++k) {

								// [ V^H_upper   0      ] [  A_upper    B_upper ] = [ V^H_upper A_upper   V^H_upper B_upper  ]
								// [  0       V^H_lower ] [  A_lower    B_lower ]   [ V^H_lower A_lower   V^H_lower B_lower  ]
								//
								// But
								//  [ A_upper B_upper ] = [ Clov_upper      0      ] [ V_upper        0    ] = [ A_upper    0    ]
								//  [ A_lower B_lower ]   [      0      Clov_lower ] [    0       V_lower  ]   [   0     B_lower ]
								//
								// So really I need to just evaluate:  V^H_upper A_upper and V^H_lower B_lower
								//
								//
								int fine_colorspin = k
										+ n_per_chiral * chiral;

								REAL64 right_r = outvecs_site_data[RE
																   + n_complex * fine_colorspin];
								REAL64 right_i = outvecs_site_data[IM
																   + n_complex * fine_colorspin];

								// Left vector -- only num_coarse_colors components with [ V^H_upper V^H_lower ]
								//
								// ie a compact storage
								// rather than:
								//
								// [ V^H_upper   0      ]
								// [	  0       V^H_lower ]
								//
								// so index with row % num_coarse_colors = matmul_row
								REAL64 left_r = invecs_site_data[RE
																 + n_complex * fine_colorspin];
								REAL64 left_i = invecs_site_data[IM
																 + n_complex * fine_colorspin];
								// Accumulate inner product V^H_row A_column
								coarse_clov_data[RE + coarse_clov_index] +=
										left_r * right_r + left_i * right_i;
								coarse_clov_data[IM + coarse_clov_index] +=
										left_r * right_i - right_r * left_i;
							}								// k

						} // matmul col
					} // matmul row

				} // chiral

			} // fine_site_idx
		} // coarse_site
	} // coarse_cb
}

}; // Namespace

