/*
 * block.cpp
 *
 *  Created on: Jan 16, 2017
 *      Author: bjoo
 */

#include "lattice/coarse/block.h"
#include "lattice/geometry_utils.h"
#include "utils/print_utils.h"
#include <cassert>
#include <utility>
using namespace std;

namespace MG {



void Block::create(const IndexArray local_lattice_dimensions,
			const IndexArray block_origin,
			const IndexArray block_dimensions,
			const IndexArray local_lattice_origin )
{

	for(int mu=0; mu < n_dim; ++mu) {
		_local_latt_origin[mu] = local_lattice_origin[mu];
	}

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
	_cbsite_list.resize(_num_sites);

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

					CBSite cbsite;

					// Now sum global coordinates of that lattice site to determine checkerboard
					int g_coord_sum=0;
					for(int mu=0; mu < n_dim; ++mu ) {
						g_coord_sum += (local_lattice_coords[mu] + local_lattice_origin[mu]);
					}
					cbsite.cb =  g_coord_sum & 1;

					// Convert coords and dims to a checkerboarded coords and dims
					IndexArray cb_coords(local_lattice_coords); cb_coords[0] /= 2;
					IndexArray cb_dims(local_lattice_dimensions); cb_dims[0] /= 2;

					// Convert cb_coords and dims to cbsite index
					cbsite.site = CoordsToIndex( cb_coords, cb_dims );

					_cbsite_list[ block_idx ] = cbsite;




				} // x
			} // y

		} // z
	}// t

	_created = true; // Mark it as done

}

// Create a list of blocks
void CreateBlockList(std::vector<Block>& blocklist, IndexArray& blocked_lattice_dimensions, const IndexArray& local_lattice_dimensions, const IndexArray& block_dimensions,
		const IndexArray& local_lattice_origin)
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
			blocklist[block_idx].create(local_lattice_dimensions, block_origin, block_dimensions,local_lattice_origin);

		}
	}
}


}


