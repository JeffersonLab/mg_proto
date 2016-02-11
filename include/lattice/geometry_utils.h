/*
 * lattice_utils.h
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



}



#endif /* INCLUDE_LATTICE_GEOMETRY_UTILS_H_ */
