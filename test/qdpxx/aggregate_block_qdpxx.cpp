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



void Block::create(const IndexArray local_lattice_dimensions,
			const IndexArray block_origin,
			const IndexArray block_dimensions)
{

	// Check the lock is feasible: origin must be nonnegative
	// origin + extent must not be greater than local_lattice size
	for(int mu=0; mu < n_dim; ++mu) {
		assert( block_origin[mu] >= 0 );
		assert( block_origin[mu] + block_dimensions[mu] <= local_lattice_dimensions[mu]);
	}

	_origin = block_origin;
	_dimensions = block_dimensions;
	_num_sites = block_dimensions[X_DIR]*block_dimensions[Y_DIR]*block_dimensions[Z_DIR]*block_dimensions[T_DIR];
	// Resize the site list
	_site_list.resize(_num_sites);

	// Loop through the blocks sites and build up a map mapping from block index to
	// Site index. We can use this for accessing the block sites in the global lattice.

	// Within a block I am not checkerboarding. So lexicographic traversal *INSIDE*
	// a block.

//#pragma omp parallel for collapse(4) shared(_site_list)
	for(IndexType t=0; t < _dimensions[T_DIR]; ++t) {
		for(IndexType z=0; z < _dimensions[Z_DIR]; ++z) {
			for(IndexType y=0; y < _dimensions[Y_DIR]; ++y) {
				for(IndexType x=0; x < _dimensions[X_DIR]; ++x) {


					IndexArray block_coords = {{x,y,z,t}};
					IndexArray local_lattice_coords;

					// Offset the block coords by the block origin in the local lattice
					for(int mu=0; mu < n_dim; ++mu) {
						local_lattice_coords[mu] = block_coords[mu] + _origin[mu];
					}

					IndexType block_idx = CoordsToIndex( block_coords, _dimensions);
					IndexType lattice_idx = CoordsToIndex( local_lattice_coords, local_lattice_dimensions) ;

#if 0
					std::cout << "Block Coordinates: ("<<_origin[0]<<","<<_origin[1]<<","<<_origin[2]<<","<<_origin[3]<<"): "
						<< "Coord: ("<<x<<","<<y<<","<<z<<","<<t <<")" << " LatticeCoord: (" << local_lattice_coords[0] <<","
							<< local_lattice_coords[1] <<"," << local_lattice_coords[2] << "," << local_lattice_coords[3] <<") "
							<< " computed sitelist index=" << block_idx << " computed lattice site index=" << lattice_idx << std::endl;
#endif

					_site_list[ block_idx ] = lattice_idx;

				} // x
			} // y

		} // z
	}// t

	_created = true; // Mark it as done

}

// Create a list of blocks
void CreateBlockList(std::vector<Block>& blocklist, IndexArray& blocked_lattice_dimensions, const IndexArray& local_lattice_dimensions, const IndexArray& block_dimensions )
{
	// Compute the dimensions of the blocked lattice. Check local lattice is divisible by block size
	for(int mu=0; mu < n_dim; mu++) {
		blocked_lattice_dimensions[mu] = local_lattice_dimensions[mu] / block_dimensions[mu];
		if( blocked_lattice_dimensions[mu] % block_dimensions[mu] != 0 ) {
			MasterLog(ERROR,"CreateBlockList: block_dimensions[%d]=%d does not divide local_lattice_dimensions[%d]=%d",
						mu,block_dimensions[mu], mu, local_lattice_dimensions[mu]);
		}
	}

	// Compute the number of blocks
	IndexType num_blocks = 1;
	for(IndexType mu=0; mu < n_dim; ++mu ) num_blocks *= blocked_lattice_dimensions[mu];

	// The ordering of the blocks *IS* checkerboarded for now, since the coarse lattice
	// is checkerboarded. But within the blocks things are lexicographic still

	IndexArray blocked_lattice_cb_dims(blocked_lattice_dimensions);
	blocked_lattice_cb_dims[0]/=n_checkerboard;
	IndexType num_cb_blocks = num_blocks/n_checkerboard;

	// Now create the blocks: I can loop through the checkerboarded 'coarse sites'
	// to index the block list in checkerboarded order.
	blocklist.resize(num_blocks);
	for(int block_cb=0; block_cb < n_checkerboard; ++block_cb) {
		for(int block_cbsite=0; block_cbsite < num_cb_blocks; ++block_cbsite ) {

			// Checkerboarded 'coarse site' index
			unsigned int block_idx = block_cbsite + block_cb*num_cb_blocks;

			/*! FIXME: is this strictly correct? How would I fold in a node checkerboard
			 *  later? Should I first create a site from the block_cbsite and then depending
			 *  then combine my node checkerboard with the one in the loop here to work out
			 *  the global checkerboard? It passes tests currently in a single node
			 *  setting so I will leave it.
			 */
			IndexArray block_coords;
			IndexToCoords(block_idx, blocked_lattice_dimensions, block_coords);

			// Compute BlockOrigin
			IndexArray block_origin(block_coords);
			for(int mu=0; mu < n_dim; ++mu) block_origin[mu]*=block_dimensions[mu];

#if 0
			std::cout << "Block Cooords=("<<block_coords[0]<<","<<block_coords[1]<<","<<block_coords[2]<<","<< block_coords[3]<<") block_index="<<block_idx
											<< " block_cb=" << block_cb << " Block Origin=("<< block_origin[0]<<","<< block_origin[1]<<","
											<< block_origin[2]<<","<<block_origin[3]<<") " <<std::endl;
#endif

			// Create Block structure
			blocklist[block_idx].create(local_lattice_dimensions, block_origin, block_dimensions);

		}
	}
}

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

			MasterLog(DEBUG, "Orthonormalizing Aggregate: %d on Block: %d",aggr, block_id);

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

	int num_coarse_cbsites = out.GetInfo().GetNumCBSites();
	int num_coarse_color = out.GetNumColor();

	// Sanity check. The number of sites in the coarse spinor
	// Has to equal the number of blocks
	assert( n_checkerboard*num_coarse_cbsites == blocklist.size() );

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
				for( IndexType fine_site_idx = 0; fine_site_idx < num_sites_in_block; fine_site_idx++ ) {

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
	int num_coarse_cbsites=coarse_in.GetInfo().GetNumCBSites();

	assert( num_coarse_cbsites == blocklist.size()/2 );

	int num_coarse_color = coarse_in.GetNumColor();
	assert( v.size() == num_coarse_color );

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
}

//! Coarsen the clover term (1 block = 1 site )
void clovTripleProductQDPXX(const std::vector<Block>& blocklist, const QDPCloverTerm& clov,const multi1d<LatticeFermion>& in_vecs, CoarseClover& cl_coarse)
{
		// Clover Triple product in here
}



};

