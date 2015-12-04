/*
 * spinor.h
 *
 *  Created on: Oct 20, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_COMPACT_CB_AOS_SPINOR_LAYOUT_H_
#define INCLUDE_LATTICE_COMPACT_CB_AOS_SPINOR_LAYOUT_H_

#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/geometry_utils.h"

namespace MGGeometry {

  /** A Class to hold spinors
   *
   *  These are essentially vectors
   *  Data is needed to store Spin, Color, Real and Imaginary components
   *
   *  This class is just a 'container' and can do only rudimentary indexing
   */

// From playpen code example:

// Spinor Needs GetLatticeInfo function
// BlockSpinor needs Index(chirality_aggregate, block, leftover_spin, thin_color, reim, blocksite) -- accessor (read & write)
// Spinor Needs Index(spin, color, reim, fullsite) -- accessor(read & write)

// NB: At this level the BlockSpinor Does not need a ghost
// The block spinor will need block: norm2(), norm() innerProduct(), add(), sub(), madd(), msub() and scale() operations
// To perform the block orthogonalization.
// The add, sub are trivially parallel
// The mul, madd, msub involves manipulating complex components
// the norm2() and norm are simple reductions
// the innerProduct involves manipulating components



  /** This is a checkerboarded layout
   *  It is still compact as it will be site major
   *  I will always store 'checkerboard even' (followed by 'checkerboard odd')
   *
   *  The lattice info site tables are not really much use to me here.
   *
   */
   template<typename T>
   class CompactCBAOSSpinorLayout {
   public:
	   	  CompactCBAOSSpinorLayout(const CompactCBAOSSpinorLayout& to_copy) :
	   		  _info(to_copy._info), _n_color(to_copy._n_color),
			  _n_spin(to_copy._n_spin), _n_sites(to_copy._n_sites),
			  _n_cb_sites(to_copy._n_cb_sites) {



	   	  }

	   	  CompactCBAOSSpinorLayout() = delete;

 	  	  CompactCBAOSSpinorLayout(const LatticeInfo& info) : _info(info),
 		  	  _n_color(info.GetNumColors()),
 			  _n_spin(info.GetNumSpins()),
 			  _n_sites(info.GetNumSites()),
			  _n_cb_sites(info.GetNumCBSites()) {}
 	  	  ~CompactCBAOSSpinorLayout() {}


 	  	  const LatticeInfo& GetLatticeInfo(void) const { return _info; }


 	  	  /* Get the index of a site based on cb and site, with spin, color, etc indices */
 	  	  inline
 		  IndexType
 		  ContainerIndex(IndexType cb,
 				  	  	 IndexType cb_index,
 						 IndexType spin_index,
 						 IndexType color_index,
 						 IndexType reim) const {

 	  		  /* cb = 0 => cb_offset = 0
 	  		   * cb = 1 => cb_offset = numCBSites(0)
 	  		   */
 	  		  return reim + n_complex*(color_index+
 	  				  _n_color*(spin_index + _n_spin*(cb_index + _n_cb_sites*cb)));
 	  	  }


	  	  inline
 	  	  IndexType
 		  ContainerIndex(IndexType site_index,
 				  	  	 IndexType spin_index,
 						 IndexType color_index,
 						 IndexType reim) const {


 	  		  return reim + n_complex*(color_index + _n_color*(spin_index + _n_spin*site_index));

 	  	  }

 	  	  inline
 		  IndexType
 		  ContainerIndex(const IndexArray& coords,
 				  	  	 IndexType spin_index,
 						 IndexType color_index,
 						 IndexType reim) {

 	  		  IndexType sum_coords = coords[0];
 	  		  for(IndexType mu=1; mu < n_dim; ++mu) sum_coords += coords[mu];
 	  		  IndexType cb =  (sum_coords + _info.GetCBOrigin() ) & 1;

 	  		  IndexArray cb_coords(coords);
 	  		  cb_coords[0] /= 2;
 	  		  IndexType cb_index = CoordsToIndex(cb_coords,_info.GetCBLatticeDimensions());

 	  		return ContainerIndex(cb,cb_index, spin_index, color_index, reim);

 	  	  }

 		  size_t GetNumData() const {return  n_complex*_n_spin*_n_color*_n_sites;}

 		  size_t GetDataInBytes() const {return  GetNumData()*sizeof(T);}

 		  size_t GhostNumElem(IndexType cb,
 				  	  	  	  IndexType dir,
 							  IndexType forw_back) const {
 			  return n_complex*_n_spin*_n_color*_info.GetNumCBSurfaceSites(dir);
 		  }

 		  size_t GhostNumBytes(IndexType cb,
 				  	  	  	   IndexType dir,
 				  	  	  	   IndexType forw_back) const { return GhostNumElem(cb,dir,forw_back)*sizeof(T); }

   private:
 	  	const LatticeInfo _info;
 	  	IndexType _n_color;
 	  	IndexType _n_spin;
 	  	IndexType _n_sites;
 	  	IndexType _n_cb_sites;


   };





}




#endif /* INCLUDE_LATTICE_LATTICE_SPINOR_H_ */
