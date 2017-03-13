/*
blbl * lattice_utils.h
 *
 *  Created on: Oct 22, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_GEOMETRY_UTILS_H_
#define INCLUDE_LATTICE_GEOMETRY_UTILS_H_

#include <vector>
#include "lattice/constants.h"
#include "utils/print_utils.h"

namespace MG {


	/** Check that one lattice divides another
	 * @param IndexArray& dims_num        The dimensions to divide by dims_num
	 * @param IndexArray& dims_den        The dimensions of the dividing lattice
	 */
	inline
	void  AssertVectorsDivideEachOther( const IndexArray& dims_num,
			const IndexArray& dims_den)
	{
		 for(unsigned int mu=0; mu < n_dim; ++mu ) {
			 if ( dims_num[mu] % dims_den[mu] != 0 ) {
				 MG::MasterLog(MG::ERROR, "CheckVectorsDivideEachOther: Dimensions incompatible");
			 }
		 }
	}

	inline
	void  AssertEquals( const IndexType v1, const IndexType v2)
	{
		if ( v1 != v2 ) {
			MG::MasterLog(MG::ERROR, "IndexType values are unequal %u and %u",v1,v2);
		}
	}


	inline
	void IndexToCoords(unsigned int index,
			          const IndexArray& dims,
				  	  IndexArray& coords)
	{
		// Minimially we have something like  index = coords[0] + dims[0]*coords[1];
		//          but coult be more complex like:  index = coords[0] + dims[0]*(coords[1] + dims[1]*(... dims[n-2]*coords[n-1)...))

		unsigned int rem=index;
		for(unsigned int mu=0; mu <= n_dim-2; ++mu) {
			unsigned int tmp = rem/dims[mu];
			coords[mu]=rem-tmp*dims[mu];
			rem = tmp;
		}
		coords[n_dim-1] = rem;
	}

	inline
	unsigned int CoordsToIndex(const IndexArray& coords,
							   const IndexArray& dims)
	{
		int dimsize = n_dim;
		unsigned int ret_val = coords[ dimsize-1 ];
		for(int dim=dimsize-2; dim >= 0; --dim) {
			ret_val *= dims[dim];
			ret_val += coords[dim];
		}
		return ret_val;
	}

		inline
		void IndexToCoords3(unsigned int index,
				          const IndexArray& dims,
						  const int mu,
					  	  IndexArray& coords)
		{
		// I want to convert site index to
			int ind = index;

			// Get the other coordinates that we don't have
			for(int dir=0; dir < n_dim; ++dir ) {
				if( dir != mu ) {

					// On the last direction we'd look at r = coord[n]/dim[n] = 0
					// since coord[n] is always less than dim[n]
					// so the algorithm should terminate correctly
					int r = ind / dims[dir];
					coords[dir] = ind -dims[dir]*r;
					ind = r;
				}
			}

		}


		inline
		unsigned int CoordsToIndex3(const IndexArray& coords,
									const int mu,
									const IndexArray& dims)
		{

			int pos=0;
			int coords3[3];
			int dims3[3];
			for(int d=0; d < n_dim; ++d) {
				if( d != mu ) {
					coords3[pos] = coords[d];
					dims3[pos]=coords[d];
					pos++;
				}
			}
			return coords3[0]+dims3[0]*(coords3[1]+dims3[1]*coords3[2]);
		}

	inline
	void CBIndexToCoords(const int cbsite, const int cb, const IndexArray& lattice_size, IndexArray& coords)
	{
		IndexArray cb_lattice_size(lattice_size); cb_lattice_size[0] /= 2; // Size of checkberboarded lattice
		IndexToCoords( cbsite, cb_lattice_size, coords);
		coords[0] *= 2; // Convert to noncheckerboarded
#if 0
		// See if we need to offset this
		int tmpcb = (coords[0]+coords[1]+coords[2]+coords[3])&1;
		if (tmpcb != cb ) coords[0] += 1;
#endif
		coords[0] += (cb + coords[1]+coords[2]+coords[3])&1;

	}

	inline
	void CBIndexToCoords(const int cbsite, const int cb, const IndexArray& lattice_size, const IndexArray& origin, IndexArray& coords)
	{
		IndexArray cb_lattice_size(lattice_size); cb_lattice_size[0] /= 2; // Size of checkberboarded lattice
		IndexToCoords( cbsite, cb_lattice_size, coords);
		coords[0] *= 2; // Convert to noncheckerboarded
#if 0
		// See if we need to offset this
		int tmpcb = (coords[0]+coords[1]+coords[2]+coords[3])&1;
		if (tmpcb != cb ) coords[0] += 1;
#endif
		coords[0] += (cb + coords[1]+coords[2]+coords[3]+origin[0]+origin[1]+origin[2]+origin[3])&1;

	}
	// Coords and Dims are uncheckerboarded.
	inline
	void CoordsToCBIndex(const IndexArray& coords, const IndexArray& dims, int& cb, int &cbsite)
	{
		cb = (coords[0]+coords[1]+coords[2]+coords[3]) & 1;

		IndexArray cb_dims(dims); cb_dims[0] /= 2;
		IndexArray cb_coords(coords); cb_coords[0] /= 2;

		cbsite = CoordsToIndex(cb_coords, cb_dims);

	}

	// Coords and Dims are uncheckerboarded.
	inline
	void CoordsToCBIndex(const IndexArray& coords, const IndexArray& dims, const IndexArray& origin, int& cb, int &cbsite)
	{

		cb = (coords[0]+coords[1]+coords[2]+coords[3] + origin[0] + origin[1]+origin[2]+origin[3]) & 1;

		IndexArray cb_dims(dims); cb_dims[0] /= 2;
		IndexArray cb_coords(coords); cb_coords[0] /= 2;

		cbsite = CoordsToIndex(cb_coords, cb_dims);

	}

}



#endif /* INCLUDE_LATTICE_GEOMETRY_UTILS_H_ */
