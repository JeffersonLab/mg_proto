/*
 * aggregate_qdpxx.cpp
 *
 *  Created on: Dec 9, 2016
 *      Author: bjoo
 */
#include "lattice/fine_qdpxx/aggregate_qdpxx.h"
#include "lattice/fine_qdpxx/aggregate_block_qdpxx.h"
#include "lattice/fine_qdpxx/transf.h"
#include "lattice/constants.h"
#include "lattice/geometry_utils.h"


using namespace QDP;

namespace MG {


// Implementation -- where possible call the site versions
//! v *= alpha (alpha is real) over and aggregate in a block, v is a QDP++ Lattice Fermion
void axBlockAggrQDPXX(const double alpha, LatticeFermion& v, const Block& block, int aggr)
{
	auto block_sitelist = block.getCBSiteList();
	int num_sites = block.getNumSites();

#pragma omp parallel for
	for(int site=0; site < num_sites; ++site) {
		const CBSite& cbsite = block_sitelist[site];
		int qdpsite = rb[ cbsite.cb ].siteTable()[ cbsite.site ];
		axAggrQDPXX(alpha,v,qdpsite,aggr);
	}
}

//! y += alpha * x (alpha is complex) over aggregate in a block, x, y are QDP++ LatticeFermions;
void caxpyBlockAggrQDPXX(const std::complex<double>& alpha, const LatticeFermion& x, LatticeFermion& y,  const Block& block, int aggr)
{
	auto block_sitelist = block.getCBSiteList();
	int num_sites = block.getNumSites();

#pragma omp parallel for
	for(int site=0; site < num_sites; ++site) {
		// Map to sites...
		const CBSite& cbsite = block_sitelist[site];
		int qdpsite = rb[ cbsite.cb ].siteTable()[cbsite.site ];

		caxpyAggrQDPXX(alpha,x,y,qdpsite,aggr);
	}

}

//! return || v ||^2 over an aggregate in a block, v is a QDP++ LatticeFermion
double norm2BlockAggrQDPXX(const LatticeFermion& v, const Block& block, int aggr)
{
	auto block_sitelist = block.getCBSiteList();
	int num_sites = block.getNumSites();
	double block_sum=0;


#pragma omp parallel for reduction(+:block_sum)
	for(int site=0; site < num_sites; ++site) {
		const CBSite& cbsite = block_sitelist[site];
		int qdpsite = rb[ cbsite.cb ].siteTable()[cbsite.site ];

		// Map to sites...
		block_sum += norm2AggrQDPXX(v,qdpsite,aggr);
	}

	return block_sum;
}

//! return < left | right > = sum left^\dagger_i * right_i for an aggregate, over a block
std::complex<double>
innerProductBlockAggrQDPXX(const LatticeFermion& left, const LatticeFermion& right, const Block& block, int aggr)
{
	auto block_sitelist = block.getCBSiteList();
	int num_sites = block.getNumSites();
	double real_part=0;
	double imag_part=0;

#pragma omp parallel for reduction(+:real_part) reduction(+:imag_part)
	for(int site=0; site < num_sites; ++site) {
		const CBSite& cbsite = block_sitelist[site];
		int qdpsite = rb[ cbsite.cb ].siteTable()[cbsite.site ];

		std::complex<double> site_prod=innerProductAggrQDPXX(left,right,qdpsite,aggr);
		real_part += real(site_prod);
		imag_part += imag(site_prod);
	}

	std::complex<double> ret_val(real_part,imag_part);
	return ret_val;
}

//! Extract the spins belonging to a given aggregate from QDP++ source vector src, into QDP++ target vector target
void extractAggregateQDPXX(LatticeFermion& target, const LatticeFermion& src, const Block& block, int aggr )
{
	auto block_sitelist = block.getCBSiteList();
	int num_sites = block.getNumSites();

#pragma omp parallel for
	for(int site=0; site < num_sites; ++site) {
		const CBSite& blocksite = block_sitelist[site];
		const int qdpsite = rb[blocksite.cb].siteTable()[blocksite.site];
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
//	assert( n_checkerboard*num_coarse_cbsites == static_cast<const int>(blocklist.size()) );

	// The number of vectors has to eaqual the number of coarse colors
	assert( v.size() == num_coarse_color );

	// This will be a loop over blocks


	for(int block_cb = 0; block_cb < n_checkerboard; ++block_cb ) {
		for(int block_cbsite = 0 ; block_cbsite < num_coarse_cbsites; ++block_cbsite) {

			int block_idx = block_cbsite + block_cb*num_coarse_cbsites;

			// The coarse site spinor is where we will write the result
			float* coarse_site_spinor = out.GetSiteDataPtr(block_cb,block_cbsite);

			// Identify the current block
			const Block& block = blocklist[block_idx];

			// Get the list of fine sites in the blocks
			auto block_sitelist = block.getCBSiteList();

			// and their number -- this is redundant, I could get it from block_sitelist.size()
			auto num_sites_in_block = block_sitelist.size();


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
					const CBSite& fine_cbsite = block_sitelist[fine_site_idx];
					const int fine_site = (rb[ fine_cbsite.cb ].siteTable())[fine_cbsite.site ];


					// Now loop over the chiral components. These are local in a site at the level of spin
					for(int chiral = 0; chiral < 2; ++chiral ) {
						//						QDPIO::cout << "RESTRICT: block=" << block_idx << " coord=" << fine_cbsite.coords[0] << ", " << fine_cbsite.coords[1]
						//								<< ", " << fine_cbsite.coords[2] << ", " << fine_cbsite.coords[3] <<" )" << " chiral=" << chiral << std::endl;

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
		} // block_cbsites
	} // block_cb
}

//! 'Prolongate' a CoarseSpinor to a QDP++ Fine Spinor
void prolongateSpinorCoarseToQDPXXFine(const std::vector<Block>& blocklist,
									   const multi1d<LatticeFermion>& v,
									   const CoarseSpinor& coarse_in, LatticeFermion& fine_out)
{
		// Prolongate in here
	IndexType num_coarse_cbsites=coarse_in.GetInfo().GetNumCBSites();

	// assert( num_coarse_cbsites == static_cast<IndexType>(blocklist.size()/2) );

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

			for(int block_cb = 0; block_cb < n_checkerboard; ++block_cb ) {
				for(int block_cbsite = 0 ; block_cbsite < num_coarse_cbsites; ++block_cbsite) {

					int block_idx = block_cbsite + block_cb*num_coarse_cbsites;

			const float *coarse_spinor = coarse_in.GetSiteDataPtr(block_cb,block_cbsite);

			// Get the list of sites in the block
			auto fine_sitelist = blocklist[block_idx].getCBSiteList();
			auto num_fine_sitelist = fine_sitelist.size();


			for( unsigned int fine_site_idx = 0; fine_site_idx < num_fine_sitelist; ++fine_site_idx) {

				const CBSite& fine_cbsite = fine_sitelist[fine_site_idx];
				int qdpsite = (rb[fine_cbsite.cb].siteTable())[ fine_cbsite.site ] ;


				for(int fine_spin=0; fine_spin < Ns; ++fine_spin) {

					int chiral = fine_spin < (Ns/2) ? 0 : 1;
		//			QDPIO::cout << "PROLONGATE: block=" << block_idx << " coord=( " << fine_cbsite.coords[0] << ", " << fine_cbsite.coords[1]
		//												<< ", " << fine_cbsite.coords[2] << ", " << fine_cbsite.coords[3] <<" )" << " fine_spin=" << fine_spin << " chiral=" << chiral << std::endl;
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

	const LatticeInfo& c_info=u_coarse.GetInfo();
	IndexArray coarse_dims = c_info.GetLatticeDimensions();


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
	for(int coarse_cb=0; coarse_cb < n_checkerboard; ++coarse_cb) {
		for(int coarse_cbsite =0; coarse_cbsite < num_coarse_cbsites; ++coarse_cbsite) {



			int block_idx = coarse_cbsite + coarse_cb*num_coarse_cbsites;

#if 0
			QDPIO::cout << "TPD: coarse_cb=" << coarse_cb << " coarse_cbsite=" << coarse_cbsite;
			IndexArray coarse_coords; CBIndexToCoords(coarse_cbsite, coarse_cb, coarse_dims,coarse_coords);
			QDPIO::cout <<  " coarse_coords=(" << coarse_coords[0]
						<< ", " << coarse_coords[1]
						<< ", " << coarse_coords[2]
						<< ", " << coarse_coords[3] << ") " << std::endl;
#endif

			// Get a Block Index

			const Block& block = blocklist[ block_idx ]; // Get the block


			//  --------------------------------------------
			//  Now do the faces
			//  ---------------------------------------------
			{
				auto face_sitelist = block.getFaceList(dir);
				auto  num_sites = face_sitelist.size();

				float *coarse_link = u_coarse.GetSiteDirDataPtr(coarse_cb,coarse_cbsite, dir);

				// Do the accumulation in double
				std::vector<double> tmp_link(n_complex*num_coarse_colorspin*num_coarse_colorspin);

				// Zero the link
        for(int col=0; col < num_coarse_colorspin; ++col) {

          for(int row=0; row < num_coarse_colorspin; ++row) {
            int coarse_link_index = n_complex*(row + num_coarse_colorspin*col);
						tmp_link[RE + coarse_link_index] = 0;
						tmp_link[IM + coarse_link_index] = 0;

					}
				}

				for(IndexType fine_site_idx = 0; fine_site_idx < static_cast<IndexType>(num_sites); ++fine_site_idx) {

					CBSite& fine_cbsite = face_sitelist[ fine_site_idx ];
					int qdp_site = rb[ fine_cbsite.cb ].siteTable()[ fine_cbsite.site ] ;
					multi1d<int> qdp_coords = Layout::siteCoords(Layout::nodeNumber(), qdp_site);

#if 0
					QDPIO::cout << "   aggregating fine_cb=" << fine_cbsite.cb << " fine_cbsite=" << fine_cbsite.site
							<< " coords=(" << qdp_coords[0] << ", " << qdp_coords[1] << ", " << qdp_coords[2] << ", " << qdp_coords[3] << " )"
							<< std::endl;
#endif

					for(int aggr_col=0; aggr_col <n_chiral; ++aggr_col ) {
					  for(int aggr_row=0; aggr_row < n_chiral; ++aggr_row) {


							// This is an num_coarse_colors x num_coarse_colors matmul
              for(int matmul_col=0; matmul_col < num_coarse_colors; ++matmul_col) {

                for(int matmul_row=0; matmul_row < num_coarse_colors; ++matmul_row) {

									// Offset by the aggr_row and aggr_column
									int row = aggr_row*num_coarse_colors + matmul_row;
									int col = aggr_col*num_coarse_colors + matmul_col;

									//Index in coarse link
									int coarse_link_index = n_complex*(row + num_coarse_colorspin*col);

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
										int spin=k/Nc+aggr_row*(Ns/2);

										// k % Nc maps to color component (0,1,2)
										int color=k%Nc;

										// Right vector
										float right_r = (float)(out_vecs[col].elem(qdp_site).elem(spin).elem(color).real());
										float right_i = (float)(out_vecs[col].elem(qdp_site).elem(spin).elem(color).imag());

										// Left vector -- only num_coarse_colors components with [ V^H_upper V^H_lower ]
										//
										// ie a compact storage
										// rather than:
										//
										// [ V^H_upper   0      ]
										// [  0       V^H_lower ]
										//
										// so index with row % num_coarse_colors = matmul_row
										float left_r = (float)(in_vecs[matmul_row].elem(qdp_site).elem(spin).elem(color).real());
										float left_i = (float)(in_vecs[matmul_row].elem(qdp_site).elem(spin).elem(color).imag());

										// Accumulate inner product V^H_row A_column
										tmp_link[RE + coarse_link_index ] += (left_r*right_r + left_i*right_i);
										tmp_link[IM + coarse_link_index ] += (left_r*right_i - right_r*left_i);
									} // k

								} // matmul_col
							} // matmul_row
						} // aggr_col
					} // aggr_row


				} // fine_site_idx
        for(int col=0; col < num_coarse_colorspin; ++col) {

          for(int row=0; row < num_coarse_colorspin; ++row) {
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
        for(int col=0; col < num_coarse_colorspin; ++col) {

          for(int row=0; row < num_coarse_colorspin; ++row) {
						int coarse_link_index = n_complex*(row + num_coarse_colorspin*col);
						tmp_link[RE + coarse_link_index] = 0;
						tmp_link[IM + coarse_link_index] = 0;

					}
				}

				for(IndexType fine_site_idx = 0; fine_site_idx < static_cast<IndexType>(num_sites); ++fine_site_idx) {

					CBSite& fine_cbsite = not_face_sitelist[ fine_site_idx ];
					int qdp_site = rb[ fine_cbsite.cb ].siteTable()[ fine_cbsite.site ] ;
					multi1d<int> qdp_coords = Layout::siteCoords(Layout::nodeNumber(), qdp_site);

#if 0
					QDPIO::cout << "   aggregating fine_cb=" << fine_cbsite.cb << " fine_cbsite=" << fine_cbsite.site
							<< " coords=(" << qdp_coords[0] << ", " << qdp_coords[1] << ", " << qdp_coords[2] << ", " << qdp_coords[3] << " )"
							<< std::endl;
#endif

					for(int aggr_col=0; aggr_col <n_chiral; ++aggr_col ) {

					  for(int aggr_row=0; aggr_row < n_chiral; ++aggr_row) {

							// This is an num_coarse_colors x num_coarse_colors matmul
              for(int matmul_col=0; matmul_col < num_coarse_colors; ++matmul_col) {

                for(int matmul_row=0; matmul_row < num_coarse_colors; ++matmul_row) {

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
										int spin=k/Nc+aggr_row*(Ns/2);

										// k % Nc maps to color component (0,1,2)
										int color=k%Nc;

										// Right vector
										float right_r = (float)(out_vecs[col].elem(qdp_site).elem(spin).elem(color).real());
										float right_i = (float)(out_vecs[col].elem(qdp_site).elem(spin).elem(color).imag());

										// Left vector -- only num_coarse_colors components with [ V^H_upper V^H_lower ]
										//
										// ie a compact storage
										// rather than:
										//
										// [ V^H_upper   0      ]
										// [  0       V^H_lower ]
										//
										// so index with row % num_coarse_colors = matmul_row
										float left_r = (float)(in_vecs[matmul_row].elem(qdp_site).elem(spin).elem(color).real());
										float left_i = (float)(in_vecs[matmul_row].elem(qdp_site).elem(spin).elem(color).imag());

										// Accumulate inner product V^H_row A_column
										tmp_link[RE + coarse_link_index ] += (left_r*right_r + left_i*right_i);
										tmp_link[IM + coarse_link_index ] += (left_r*right_i - right_r*left_i);
									} // k

								} // matmul_row
							} // matmul_col
						} // aggr_row
					} // aggr_col


				} // fine_site_idx

				for(int col=0; col < num_coarse_colorspin; ++col) {
				  for(int row=0; row < num_coarse_colorspin; ++row) {

						int coarse_link_index = n_complex*(row + num_coarse_colorspin*col);
						coarse_link[RE + coarse_link_index ] += (float)( tmp_link[RE + coarse_link_index] );
						coarse_link[IM + coarse_link_index ] += (float)( tmp_link[IM + coarse_link_index ] );

					} // rows
				} // cols
			}

		} // coarse cbsite

	} //coarse_cb

}

//! Coarsen the clover term (1 block = 1 site )
void clovTripleProductQDPXX(const std::vector<Block>& blocklist, const QDPCloverTerm& clov,const multi1d<LatticeFermion>& in_vecs, CoarseGauge& gauge_clover)
{
	// Clover Triple product in here
	int num_coarse_colors = gauge_clover.GetNumColor();
	int num_chiral_components = 2;
	int num_coarse_colorspin = num_coarse_colors*num_chiral_components;
	int num_coarse_cbsites = gauge_clover.GetInfo().GetNumCBSites();
	const int num_spincolor_per_chiral = (Nc*Ns)/num_chiral_components;
	const int num_spin_per_chiral = Ns/num_chiral_components;

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
			auto block_sitelist = block.getCBSiteList();
			auto num_block_sites = block.getNumSites();


			float *coarse_clov = gauge_clover.GetSiteDiagDataPtr(coarse_cb,coarse_cbsite);

			// Accumulate into this tmp_link and in double
			std::vector<double> tmp_link(2*n_complex*num_coarse_colors*num_coarse_colors);
			for(int j=0; j < 2*n_complex*num_coarse_colors*num_coarse_colors; ++j) {
				tmp_link[j] = 0;
			}

			for(IndexType fine_site_idx=0; fine_site_idx < static_cast<IndexType>(num_block_sites); ++fine_site_idx) {

				const CBSite& fine_cbsite = block_sitelist[ fine_site_idx ];
				int site = rb[ fine_cbsite.cb ].siteTable()[ fine_cbsite.site ];

				for(int chiral =0; chiral < num_chiral_components; ++chiral) {


					// This is an num_coarse_colors x num_coarse_colors matmul
          for(int matmul_col=0; matmul_col < num_coarse_colors; ++matmul_col) {

            for(int matmul_row=0; matmul_row < num_coarse_colors; ++matmul_row) {


							//Index in coarse link
							int coarse_clov_index = n_complex*( (matmul_row+ num_coarse_colors*matmul_col)
		                                                        + chiral*num_coarse_colors*num_coarse_colors );


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
								float right_r = out_vecs[matmul_col].elem(site).elem(spin).elem(color).real();
								float right_i = out_vecs[matmul_col].elem(site).elem(spin).elem(color).imag();

								// Left vector -- only num_coarse_colors components with [ V^H_upper V^H_lower ]
								//
								// ie a compact storage
								// rather than:
								//
								// [ V^H_upper   0      ]
								// [	  0       V^H_lower ]
								//
								// so index with row % num_coarse_colors = matmul_row
								float left_r = in_vecs[matmul_row ].elem(site).elem(spin).elem(color).real();
								float left_i = in_vecs[matmul_row ].elem(site).elem(spin).elem(color).imag();

								// Accumulate inner product V^H_row A_column
								tmp_link[RE + coarse_clov_index ] += (double)(left_r*right_r + left_i*right_i);
								tmp_link[IM + coarse_clov_index ] += (double)(left_r*right_i - right_r*left_i);
							} // k

						} // matmul col
					} // matmul row

				} // chiral

			}// fine_site_idx

			// accumulate it
			for(int chiral=0; chiral < num_chiral_components; ++chiral) {
				int row_col_min = (chiral == 0) ? 0 : num_coarse_colors;

				for(int col=0; col < num_coarse_colors; ++col) {
				  for(int row=0 ; row < num_coarse_colors; ++row) {

						int outrow = row + row_col_min;
						int outcol = col + row_col_min;
						coarse_clov[ RE + n_complex*(outrow + num_coarse_colorspin*outcol) ] +=
						    tmp_link[ RE + n_complex*(row + num_coarse_colors*col)
						              + chiral*n_complex*num_coarse_colors*num_coarse_colors];

						coarse_clov[ IM + n_complex*(outrow + num_coarse_colorspin*outcol) ] +=
						      tmp_link[ IM + n_complex*(row + num_coarse_colors*col)
						                + chiral*n_complex*num_coarse_colors*num_coarse_colors];

					} // col
				} // row
			} // chiral

		} // coarse_site
	} // coarse_cb

}



};

