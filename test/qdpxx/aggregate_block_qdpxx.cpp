/*
 * aggregate_qdpxx.cpp
 *
 *  Created on: Dec 9, 2016
 *      Author: bjoo
 */
#include "aggregate_qdpxx.h"
#include "aggregate_block_qdpxx.h"
#include "lattice/constants.h"
#include "lattice/geometry_utils.h"
#include "transf.h"

using namespace QDP;
using namespace MG;

namespace MGTesting {


// Implementation -- where possible call the site versions
//! v *= alpha (alpha is real) over and aggregate in a block, v is a QDP++ Lattice Fermion
void axBlockAggrQDPXX(const double alpha, LatticeFermion& v, const Block& block, int aggr)
{
	auto block_sitelist = block.getSiteList();
	int num_sites = block.getNumSites();

#pragma omp parallel for
	for(int site=0; site < num_sites; ++site) {

		axAggrQDPXX(alpha,v,block_sitelist[site],aggr);
	}
}

//! y += alpha * x (alpha is complex) over aggregate in a block, x, y are QDP++ LatticeFermions;
void caxpyBlockAggrQDPXX(const std::complex<double>& alpha, const LatticeFermion& x, LatticeFermion& y,  const Block& block, int aggr)
{
	auto block_sitelist = block.getSiteList();
	int num_sites = block.getNumSites();

#pragma omp parallel for
	for(int site=0; site < num_sites; ++site) {
		// Map to sites...
		caxpyAggrQDPXX(alpha,x,y,block_sitelist[site],aggr);
	}

}

//! return || v ||^2 over an aggregate in a block, v is a QDP++ LatticeFermion
double norm2BlockAggrQDPXX(const LatticeFermion& v, const Block& block, int aggr)
{
	auto block_sitelist = block.getSiteList();
	int num_sites = block.getNumSites();
	double block_sum=0;


#pragma omp parallel for reduction(+:block_sum)
	for(int site=0; site < num_sites; ++site) {
		// Map to sites...
		block_sum += norm2AggrQDPXX(v,block_sitelist[site],aggr);
	}

	return block_sum;
}

//! return < left | right > = sum left^\dagger_i * right_i for an aggregate, over a block
std::complex<double>
innerProductBlockAggrQDPXX(const LatticeFermion& left, const LatticeFermion& right, const Block& block, int aggr)
{
	auto block_sitelist = block.getSiteList();
	int num_sites = block.getNumSites();
	double real_part=0;
	double imag_part=0;

#pragma omp parallel for reduction(+:real_part) reduction(+:imag_part)
	for(int site=0; site < num_sites; ++site) {
		std::complex<double> site_prod=innerProductAggrQDPXX(left,right,block_sitelist[site],aggr);
		real_part += real(site_prod);
		imag_part += imag(site_prod);
	}

	std::complex<double> ret_val(real_part,imag_part);
	return ret_val;
}

//! Extract the spins belonging to a given aggregate from QDP++ source vector src, into QDP++ target vector target
void extractAggregateQDPXX(LatticeFermion& target, const LatticeFermion& src, const Block& block, int aggr )
{
	auto block_sitelist = block.getSiteList();
	int num_sites = block.getNumSites();

#pragma omp parallel for
	for(int site=0; site < num_sites; ++site) {
		int qdpsite = block_sitelist[site];
		for(int spin=aggr*Ns/2; spin < (aggr+1)*Ns/2; ++spin) {
			for(int color=0; color < 3; ++color) {
				target.elem(qdpsite).elem(spin).elem(color).real() = src.elem(qdpsite).elem(spin).elem(color).real();
				target.elem(qdpsite).elem(spin).elem(color).imag() = src.elem(qdpsite).elem(spin).elem(color).imag();
			}
		}
	}
}

//! Orthonormalize vecs over the spin aggregates within the sites
void orthonormalizeBlockAggregatesQDPXX(multi1d<LatticeFermion>& vecs, const std::vector<Block>& block_list)
{
	int num_blocks = block_list.size();

	for(int aggr=0; aggr < 2; ++aggr) {

		for(int block_id=0; block_id < num_blocks; block_id++) {

			const Block& block = block_list[block_id];

			// This will be over blocks...
			// do vecs[0] ... vecs[N]
			for(int curr_vec=0; curr_vec < vecs.size(); curr_vec++) {

				// orthogonalize against previous vectors
				// if curr_vec == 0 this will be skipped
				for(int prev_vec=0; prev_vec < curr_vec; prev_vec++) {

					std::complex<double> iprod = innerProductBlockAggrQDPXX( vecs[prev_vec], vecs[curr_vec], block, aggr);
					std::complex<double> minus_iprod=std::complex<double>(-real(iprod), -imag(iprod) );

					// curr_vec <- curr_vec - <curr_vec|prev_vec>*prev_vec = -iprod*prev_vec + curr_vec
					caxpyBlockAggrQDPXX( minus_iprod, vecs[prev_vec], vecs[curr_vec], block, aggr);
				}

				// Normalize current vector
				double inv_norm = ((double)1)/sqrt(norm2BlockAggrQDPXX(vecs[curr_vec], block, aggr));

				// vecs[curr_vec] = inv_norm * vecs[curr_vec]
				axBlockAggrQDPXX(inv_norm, vecs[curr_vec], block, aggr);
			}


		}	// block
	}// aggregates
}

//! 'Restrict' a QDP++ spinor to a CoarseSpinor with the same geometry
void restrictSpinorQDPXXFineToCoarse( const std::vector<Block>& blocklist, const multi1d<LatticeFermion>& v,
		const LatticeFermion& ferm_in, CoarseSpinor& out)
{

	const int num_coarse_cbsites = out.GetInfo().GetNumCBSites();
	const int num_coarse_color = out.GetNumColor();

	// Sanity check. The number of sites in the coarse spinor
	// Has to equal the number of blocks
	assert( n_checkerboard*num_coarse_cbsites == static_cast<const int>(blocklist.size()) );

	// The number of vectors has to eaqual the number of coarse colors
	assert( v.size() == num_coarse_color );

	// This will be a loop over blocks
	for(int block_cb=0; block_cb < n_checkerboard; ++block_cb) {
		for(int block_cbsite = 0; block_cbsite < num_coarse_cbsites; ++block_cbsite) {
			// The coarse site spinor is where we will write the result
			float* coarse_site_spinor = out.GetSiteDataPtr(block_cb,block_cbsite);

			IndexType block_idx = block_cbsite + block_cb*num_coarse_cbsites;

			// Identify the current block
			const Block& block = blocklist[block_idx];

			// Get the list of fine sites in the blocks
			auto block_sitelist = block.getSiteList();

			// and their number -- this is redundant, I could get it from block_sitelist.size()
			auto num_sites_in_block = block.getNumSites();


			// Zero the accumulation in the current site
			for(int chiral = 0; chiral < 2; ++chiral ) {
				for(int coarse_color=0; coarse_color  < num_coarse_color; coarse_color++) {
					int coarse_colorspin = coarse_color + chiral * num_coarse_color;
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
			for(int coarse_color=0; coarse_color  < num_coarse_color; coarse_color++) {

				// Now aggregate over all the sites in the block -- this will be over a single vector...
				// NB: The loop indices may be later rerolled, e.g. if we can restrict multiple vectors at once
				// Then having the coarse_color loop inner will be better.
				for( IndexType fine_site_idx = 0; fine_site_idx < static_cast<IndexType>(num_sites_in_block); fine_site_idx++ ) {

					// Find the fine site
					int fine_site = block_sitelist[fine_site_idx];

					// Now loop over the chiral components. These are local in a site at the level of spin
					for(int chiral = 0; chiral < 2; ++chiral ) {

						// Identify the color spin component we are accumulating
						int coarse_colorspin = coarse_color + chiral * num_coarse_color;

						// Aggregate the spins for the site.
						for(int spin=0; spin < Ns/2; ++spin ) {
							for(int color=0; color < Nc; ++color ) {
								int targ_spin = spin + chiral*(Ns/2); // Offset by whether upper/lower

								REAL left_r = v[ coarse_color ].elem(fine_site).elem(targ_spin).elem(color).real();
								REAL left_i = v[ coarse_color ].elem(fine_site).elem(targ_spin).elem(color).imag();

								REAL right_r = ferm_in.elem(fine_site).elem(targ_spin).elem(color).real();
								REAL right_i = ferm_in.elem(fine_site).elem(targ_spin).elem(color).imag();

								// It is V_j^H  ferm_in so conj(left)*right.
								coarse_site_spinor[ RE + n_complex*coarse_colorspin ] += left_r * right_r + left_i * right_i;
								coarse_site_spinor[ IM + n_complex*coarse_colorspin ] += left_r * right_i - right_r * left_i;

							} // color
						}	 // spin aggregates
					} // chiral component
				} // fine site_idx
			} // coarse_color
		} // block_cbsite
	} // block_cb
}

//! 'Prolongate' a CoarseSpinor to a QDP++ Fine Spinor
void prolongateSpinorCoarseToQDPXXFine(const std::vector<Block>& blocklist,
									   const multi1d<LatticeFermion>& v,
									   const CoarseSpinor& coarse_in, LatticeFermion& fine_out)
{
		// Prolongate in here
	IndexType num_coarse_cbsites=coarse_in.GetInfo().GetNumCBSites();

	assert( num_coarse_cbsites == static_cast<IndexType>(blocklist.size()/2) );

	IndexType num_coarse_color = coarse_in.GetNumColor();
	assert( static_cast<IndexType>(v.size()) == num_coarse_color);

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
		for(IndexType block_cbsite=0; block_cbsite < num_coarse_cbsites; ++block_cbsite ) {

			// Our block index is always block_cbsite + block_cb * num_coarse_cbsites
			IndexType block_idx = block_cbsite + block_cb*num_coarse_cbsites;

			const float *coarse_spinor = coarse_in.GetSiteDataPtr(block_cb,block_cbsite);

			// Get the list of sites in the block
			auto fine_sitelist = blocklist[block_idx].getSiteList();
			auto num_fine_sitelist = blocklist[block_idx].getNumSites();


			for( unsigned int fine_site_idx = 0; fine_site_idx < num_fine_sitelist; ++fine_site_idx) {

				int qdpsite = fine_sitelist[fine_site_idx];


				for(int fine_spin=0; fine_spin < Ns; ++fine_spin) {

					int chiral = fine_spin < (Ns/2) ? 0 : 1;

					for(int fine_color=0; fine_color < Nc; fine_color++ ) {

						fine_out.elem(qdpsite).elem(fine_spin).elem(fine_color).real() = 0;
						fine_out.elem(qdpsite).elem(fine_spin).elem(fine_color).imag() = 0;


						for(int coarse_color = 0; coarse_color < num_coarse_color; coarse_color++) {

							REAL left_r = v[coarse_color].elem(qdpsite).elem(fine_spin).elem(fine_color).real();
							REAL left_i = v[coarse_color].elem(qdpsite).elem(fine_spin).elem(fine_color).imag();

							int colorspin = coarse_color + chiral*num_coarse_color;
							REAL right_r = coarse_spinor[ RE + n_complex*colorspin];
							REAL right_i = coarse_spinor[ IM + n_complex*colorspin];

							// V_j | out  (rather than V^{H}) so needs regular complex mult?
							fine_out.elem(qdpsite).elem(fine_spin).elem(fine_color).real() += left_r * right_r - left_i * right_i;
							fine_out.elem(qdpsite).elem(fine_spin).elem(fine_color).imag() += left_i * right_r + left_r * right_i;
						}
					}
				}
			}
		}
	}

}

//! Coarsen one direction of a 'dslash' link
void dslashTripleProductDirQDPXX(const std::vector<Block>& blocklist,
								int dir, const multi1d<LatticeColorMatrix>& u,
								const multi1d<LatticeFermion>& in_vecs, CoarseGauge& u_coarse)
{
	  // Dslash triple product in here


	int num_coarse_colors = u_coarse.GetNumColor();
	int num_coarse_colorspin = u_coarse.GetNumColorSpin();

	int num_coarse_cbsites = u_coarse.GetInfo().GetNumCBSites();
	const int n_chiral = 2;
	const int num_spincolor_per_chiral = (Nc*Ns)/n_chiral;
	const int num_spin_per_chiral = Ns/n_chiral;



	// in vecs has size Ncolor_c = num_coarse_colorspin/2
	// But this mixes both upper and lower spins
	// Once we deal with those separately we will need num_coarse_colorspin results
	// And we will need to apply the 'DslashDir' separately to each aggregate

	assert( in_vecs.size() == num_coarse_colors);

	multi1d<LatticeFermion> out_vecs( num_coarse_colorspin );

	// Hit in_vecs with DslashDir -- this leaves
	for(int j=0; j < num_coarse_colorspin; ++j) {
		out_vecs[j]=zero;
	}


	// Apply DslashDir to each aggregate separately.
	// DslashDir may mix spins with (1 +/- gamma_mu)
	for(int j=0; j < num_coarse_colors; ++j) {
		for(int aggr=0; aggr < n_chiral; ++aggr) {
			LatticeFermion tmp=zero;
			extractAggregateQDPXX(tmp, in_vecs[j], aggr);
			DslashDirQDPXX(out_vecs[aggr*num_coarse_colors+j], u, tmp, dir);
		}
	}

	// Loop over the coarse sites (blocks)
	for( IndexType coarse_cb=0; coarse_cb < 2; ++coarse_cb) {
		for( IndexType coarse_cbsite=0; coarse_cbsite < num_coarse_cbsites; ++coarse_cbsite) {

			// Get a Block Index
			unsigned int block_idx = coarse_cbsite + coarse_cb*num_coarse_cbsites;
			const Block& block = blocklist[ block_idx ]; // Get the block
			auto  block_sitelist = block.getSiteList();
			auto  num_block_sites = block.getNumSites();

			// Get teh coarse site for writing
			float *coarse_link = u_coarse.GetSiteDirDataPtr(coarse_cb,coarse_cbsite,dir);

			std::vector<double> tmp_link(n_complex*num_coarse_colorspin*num_coarse_colorspin);

			// Zero the link
			for(int row=0; row < num_coarse_colorspin; ++row) {
				for(int col=0; col < num_coarse_colorspin; ++col) {
					int coarse_link_index = n_complex*(col + num_coarse_colorspin*row);
					tmp_link[RE + coarse_link_index] = 0;
					tmp_link[IM + coarse_link_index] = 0;

				}
			}

			for(IndexType fine_site_idx = 0; fine_site_idx < static_cast<IndexType>(num_block_sites); ++fine_site_idx) {

				const int site = block_sitelist[fine_site_idx];

				for(int aggr_row=0; aggr_row < n_chiral; ++aggr_row) {
					for(int aggr_col=0; aggr_col <n_chiral; ++aggr_col ) {

						// This is an num_coarse_colors x num_coarse_colors matmul
						for(int matmul_row=0; matmul_row < num_coarse_colors; ++matmul_row) {
							for(int matmul_col=0; matmul_col < num_coarse_colors; ++matmul_col) {

								// Offset by the aggr_row and aggr_column
								int row = aggr_row*num_coarse_colors + matmul_row;
								int col = aggr_col*num_coarse_colors + matmul_col;

								//Index in coarse link
								int coarse_link_index = n_complex*(col+ num_coarse_colorspin*row);

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
									int spin=k/Nc+aggr_row*num_spin_per_chiral;

									// k % Nc maps to color component (0,1,2)
									int color=k%Nc;

									// Right vector
									REAL64 right_r = out_vecs[col].elem(site).elem(spin).elem(color).real();
									REAL64 right_i = out_vecs[col].elem(site).elem(spin).elem(color).imag();

									// Left vector -- only num_coarse_colors components with [ V^H_upper V^H_lower ]
									//
									// ie a compact storage
									// rather than:
									//
									// [ V^H_upper   0      ]
									// [  0       V^H_lower ]
									//
									// so index with row % num_coarse_colors = matmul_row
									REAL64 left_r = in_vecs[matmul_row].elem(site).elem(spin).elem(color).real();
									REAL64 left_i = in_vecs[matmul_row].elem(site).elem(spin).elem(color).imag();

									// Accumulate inner product V^H_row A_column
									tmp_link[RE + coarse_link_index ] += left_r*right_r + left_i*right_i;
									tmp_link[IM + coarse_link_index ] += left_r*right_i - right_r*left_i;
								} // k

							} // matmul_col
						} // matmul_row
					} // aggr_col
				} // aggr_row
			} // fine_site_idx
			// Zero the link

			for(int row=0; row < num_coarse_colorspin; ++row) {
				for(int col=0; col < num_coarse_colorspin; ++col) {
					int coarse_link_index = n_complex*(col + num_coarse_colorspin*row);
					coarse_link[RE + coarse_link_index ] = tmp_link[RE + coarse_link_index];
					coarse_link[IM + coarse_link_index ] = tmp_link[IM + coarse_link_index ];

				}
			}


		} // coarse_cbsite
	} // coarse_cb

}

//! Coarsen the clover term (1 block = 1 site )
void clovTripleProductQDPXX(const std::vector<Block>& blocklist, const QDPCloverTerm& clov,const multi1d<LatticeFermion>& in_vecs, CoarseClover& cl_coarse)
{
	// Clover Triple product in here
	int num_coarse_colors = cl_coarse.GetNumColor();
	int num_chiral_components = cl_coarse.GetNumChiral();
	int num_coarse_cbsites = cl_coarse.GetInfo().GetNumCBSites();
	const int num_spincolor_per_chiral = (Nc*Ns)/2;
	const int num_spin_per_chiral = Ns/2;

	// in vecs has size num_coarse_colors = Ncolorspin_c/2
	// But this mixes both upper and lower spins
	// Once we deal with those separately we will need Ncolorspin_c results
	// And we will need to apply the 'DslashDir' separately to each aggregate

	assert( in_vecs.size() == num_coarse_colors );
	assert( num_chiral_components == 2);

	// out_vecs is the result of applying clover term to in_vecs
	// NOTE!!!: Unlike with Dslash where (1 +/- gamma_mu) mixes the upper and lower spin components
	// Clover *does not* do this. In this chiral basis that we use Clover is block diagonal
	// So it acts independently on upper and lower spin components.
	// This means Ncolor vectors are sufficient. The upper components will hold the results of
	// clover_term applied to the upper components while the lower components will hold the results of
	// clover_term applied to the lower components in the same way in_vector combines upper and lower
	// components.
	multi1d<LatticeFermion> out_vecs( num_coarse_colors );

	// Zero the output
	for(int j=0; j < num_coarse_colors; ++j) {
		out_vecs[j]=zero;
	}

	// for each in-vector pull out respectively the lower and upper spins
	// multiply by clover and store in out_vecs. There will be num_coarse_colors*num_chiral_components output
	// vectors
	for(int j=0; j < num_coarse_colors; ++j) {

		// Clover term is block diagonal
		// So I can apply it once, the upper and lower spin components will
		// be acted on independently. No need to separate the aggregates before
		// applying

		LatticeFermion tmp=zero;
		for(int cb=0; cb < 2; ++cb) {
			clov.apply(out_vecs[j], in_vecs[j], 0, cb);

		}
	}


	// Technically these outer loops should be over all the blocks.
	for(int coarse_cb=0; coarse_cb < n_checkerboard; ++coarse_cb) {
		for(int coarse_cbsite=0; coarse_cbsite < num_coarse_cbsites; ++coarse_cbsite) {

			int block_idx = coarse_cbsite + coarse_cb*num_coarse_cbsites;
			const Block& block = blocklist[block_idx];
			auto block_sitelist = block.getSiteList();
			auto num_block_sites = block.getNumSites();


			// Zero out both chiral blocks of the result
			for(int chiral = 0; chiral < num_chiral_components; ++chiral ) {
				// This is now an Ncomplex*num_coarse_colors x Ncomplex*num_coarse_colors
				float *coarse_clov = cl_coarse.GetSiteChiralDataPtr(coarse_cb,coarse_cbsite, chiral);

				// Zero the link
				for(int row=0; row < num_coarse_colors; ++row) {
					for(int col=0; col < num_coarse_colors; ++col) {
						int coarse_clov_index = n_complex*(col + num_coarse_colors*row);
						coarse_clov[RE + coarse_clov_index] = 0;
						coarse_clov[IM + coarse_clov_index] = 0;

					}
				}
			}


			for(IndexType fine_site_idx=0; fine_site_idx < static_cast<IndexType>(num_block_sites); ++fine_site_idx) {
				int site = block_sitelist[fine_site_idx];

				for(int chiral =0; chiral < num_chiral_components; ++chiral) {

					// This is now an Ncomplex*num_coarse_colors x Ncomplex*num_coarse_colors
					float *coarse_clov = cl_coarse.GetSiteChiralDataPtr(coarse_cb,coarse_cbsite, chiral);

					// This is an num_coarse_colors x num_coarse_colors matmul
					for(int matmul_row=0; matmul_row < num_coarse_colors; ++matmul_row) {
						for(int matmul_col=0; matmul_col < num_coarse_colors; ++matmul_col) {


							//Index in coarse link
							int coarse_clov_index = n_complex*(matmul_col+ num_coarse_colors*matmul_row);


							// Inner product loop
							for(int k=0; k < num_spincolor_per_chiral; ++k) {

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
								int spin=k/Nc + chiral * num_spin_per_chiral;  // Upper or lower spin depending on chiral

								// k % Nc maps to color component (0,1,2)
								int color=k%Nc;

								// Right vector - chiral*num_coarse_colors selects A_upper ( chiral=0 ) or B_lower (chiral=1)

								// NB: Out vecs has only NColor members
								REAL64 right_r = out_vecs[matmul_col].elem(site).elem(spin).elem(color).real();
								REAL64 right_i = out_vecs[matmul_col].elem(site).elem(spin).elem(color).imag();

								// Left vector -- only num_coarse_colors components with [ V^H_upper V^H_lower ]
								//
								// ie a compact storage
								// rather than:
								//
								// [ V^H_upper   0      ]
								// [	  0       V^H_lower ]
								//
								// so index with row % num_coarse_colors = matmul_row
								REAL64 left_r = in_vecs[matmul_row ].elem(site).elem(spin).elem(color).real();
								REAL64 left_i = in_vecs[matmul_row ].elem(site).elem(spin).elem(color).imag();

								// Accumulate inner product V^H_row A_column
								coarse_clov[RE + coarse_clov_index ] += left_r*right_r + left_i*right_i;
								coarse_clov[IM + coarse_clov_index ] += left_r*right_i - right_r*left_i;
							} // k

						} // matmul col
					} // matmul row

				} // chiral

			}// fine_site_idx
		} // coarse_site
	} // coarse_cb

}



};

