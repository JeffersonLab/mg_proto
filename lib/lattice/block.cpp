/*
 * block.cpp
 *
 *  Created on: Jan 16, 2017
 *      Author: bjoo
 */

#include "lattice/coarse/block.h"
#include "lattice/geometry_utils.h"
#include "utils/print_utils.h"
#include <iostream>
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
	_num_cbsites = _num_sites/n_checkerboard;
	// Resize the site list

	_cbsite_list.resize(_num_sites);
	_inner_body.reserve(_num_sites);


#if 0

	for(int cb=0; cb < n_checkerboard; ++cb) {
		for(int cbsite=0; cbsite < _num_cbsites; ++cbsite) {
			IndexType index_in_block = cbsite + cb*_num_cbsites;

			IndexArray block_coords;
			CBIndexToCoords(cbsite,cb,_dimensions, block_coords);
			IndexArray local_lattice_coords;

			// Offset the block coords by the block origin in the local lattice
			for(int mu=0; mu < n_dim; ++mu) {
				local_lattice_coords[mu] = block_coords[mu] + _origin[mu];
			}

			CBSite fine_cbsite;
			CoordsToCBIndex(local_lattice_coords, local_lattice_dimensions, fine_cbsite.cb, fine_cbsite.site);
			_cbsite_list[ index_in_block ] = fine_cbsite;

		}
	}
#endif
#if 1
	// Looop through the sites in the blocks.
	for(IndexType t=0; t < _dimensions[T_DIR]; ++t) {
		for(IndexType z=0; z < _dimensions[Z_DIR]; ++z) {
			for(IndexType y=0; y < _dimensions[Y_DIR]; ++y) {
				for(IndexType x=0; x < _dimensions[X_DIR]; ++x) {


					// Convert the 'in-block' coordinates to fine coordinates by adding origin
					IndexArray in_block_coords = {{x,y,z,t}};
					IndexArray local_lattice_coords;

					// Offset the block coords by the block origin in the local lattice
					for(int mu=0; mu < n_dim; ++mu) {
						local_lattice_coords[mu] = in_block_coords[mu] + _origin[mu];
					}

					// There is no requirement for a checkerboard ordering in the block itself.
					// Only the coarse sites need to be checkerboarded. The fine sites in the block
					// will contain multile checkeboards  -- so just a simple CoordsToIndex here
					// to convert to a linear index.

					IndexType index_in_block = CoordsToIndex(in_block_coords, _dimensions);

					// But now, I want to convert the fine lattice site in local_lattice_coords
					// into a CB & CBSite pair

					CBSite fine_cbsite;
					CoordsToCBIndex(local_lattice_coords, local_lattice_dimensions, fine_cbsite.cb, fine_cbsite.site);
				    fine_cbsite.coords = local_lattice_coords;

#if 0
					std::cout << "Block Coordinates: ("<<_origin[0]<<","<<_origin[1]<<","<<_origin[2]<<","<<_origin[3]<<"): "
						<< "Coord: ("<<x<<","<<y<<","<<z<<","<<t <<")" << " LatticeCoord: (" << local_lattice_coords[0] <<","
							<< local_lattice_coords[1] <<"," << local_lattice_coords[2] << "," << local_lattice_coords[3] <<") "
							<< " cb=" << fine_cbsite.cb << " site=" << fine_cbsite.site << std::endl;
#endif
					_cbsite_list[ index_in_block ] = fine_cbsite;

					bool in_face[8]={false,false,false,false,false,false,false,false};
					in_face[0] = (x == _dimensions[X_DIR] - 1 );   // In forward X-face
					in_face[1] = (x == 0 ); // In backward X-face
					in_face[2] = (y == _dimensions[Y_DIR] - 1); // In forward Y-face
					in_face[3] = (y == 0 );
					in_face[4] = (z == _dimensions[Z_DIR] - 1 );   // In forward Z-face
					in_face[5] = (z == 0 ); // In backward X-face
					in_face[6] = (t == _dimensions[T_DIR] - 1); // In forward T-face
					in_face[7] = (t == 0 );

					bool in_a_face = false;
					for(int mu=0; mu < 8; ++mu ) {
						in_a_face |= in_face[mu];
					}

					// If this site is not in any faces -- add to inner bodgy
					if (! in_a_face ) {
						_inner_body.push_back(fine_cbsite);
					}


					for(int mu=0; mu < 8; ++mu) {
						// if the site is in the face add it to the face list
						if ( in_face[mu] ) {
							_face[mu].push_back(fine_cbsite);
						}
						else{
							_not_face[mu].push_back(fine_cbsite);  // This may be include sites also in with inner body.
						}
					}

				} // x
			} // y

		} // z
	}// t


#endif
	_created = true; // Mark it as done

}

// Create a list of blocks
void CreateBlockList(std::vector<Block>& blocklist, IndexArray& coarse_lattice_dimensions, const IndexArray& fine_lattice_dimensions, const IndexArray& block_dimensions,
		const IndexArray& local_lattice_origin)
{
	// Compute the dimensions of the blocked lattice. Check local lattice is divisible by block size
	for(int mu=0; mu < n_dim; mu++) {
		coarse_lattice_dimensions[mu] = fine_lattice_dimensions[mu] / block_dimensions[mu];
		if( fine_lattice_dimensions[mu] % block_dimensions[mu] != 0 ) {
			MasterLog(ERROR,"CreateBlockList: block_dimensions[%d]=%d does not divide local_lattice_dimensions[%d]=%d",
						mu,block_dimensions[mu], mu, fine_lattice_dimensions[mu]);
		}
	}

	// Compute the number of blocks
	IndexType num_coarse_sites = 1;
	for(IndexType mu=0; mu < n_dim; ++mu ) num_coarse_sites *= coarse_lattice_dimensions[mu];

	// The ordering of the blocks *IS* checkerboarded for now, since the coarse lattice
	// is checkerboarded. But within the blocks things are lexicographic still

	IndexArray coarse_lattice_cb_dims(coarse_lattice_dimensions);
	coarse_lattice_cb_dims[0]/=n_checkerboard;
	IndexType num_coarse_cb_sites = num_coarse_sites/n_checkerboard;

	// Now create the blocks: I can loop through the checkerboarded 'coarse sites'
	// to index the block list in checkerboarded order.

	// FIXME: This is a block_cb, coarse_cbsite loop
	//        However, block_idx is actually a lexicographic coordinate.???
	//

	// Storage order of block list is coarse cbsite fastest then, coarse cb.
	blocklist.resize(num_coarse_sites);
	for(int coarse_cb=0; coarse_cb < n_checkerboard; ++coarse_cb) {
		for(int coarse_cbsite=0; coarse_cbsite < num_coarse_cb_sites; ++coarse_cbsite ) {

			// Global index in the list of blocks. cbsite fastest
			unsigned int block_idx = coarse_cbsite + coarse_cb*num_coarse_cb_sites;

			/*! FIXME: is this strictly correct? How would I fold in a node checkerboard
			 *  later? Should I first create a site from the coarse_cbsite and then depending
			 *  then combine my node checkerboard with the one in the loop here to work out
			 *  the global checkerboard? It passes tests currently in a single node
			 *  setting so I will leave it.
			 */

			// convert cb/site to coordinates

			IndexArray coarse_coords;
			CBIndexToCoords(coarse_cbsite, coarse_cb, coarse_lattice_dimensions, coarse_coords);

			// Compute BlockOrigin
			IndexArray block_origin(coarse_coords);
			for(int mu=0; mu < n_dim; ++mu) block_origin[mu]*=block_dimensions[mu];

#if 0
			std::cout << "coarse_cb=" << coarse_cb << " coarse_cbsite="<< coarse_cbsite << " coarse_coords=(" << coarse_coords[0] << " , "
						<< coarse_coords[1] << " , " << coarse_coords[2] <<" , " << coarse_coords[3] << " )   block_origin = ( " << block_origin[0]<<" , "<< block_origin[1]<<" , "
											<< block_origin[2]<<" , "<<block_origin[3]<<" ) " <<std::endl;
#endif

			// Create Block structure
			blocklist[block_idx].create(fine_lattice_dimensions, block_origin, block_dimensions,local_lattice_origin);

		}
	}
}


}


