/*
 * spinor.h
 *
 *  Created on: Oct 20, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_CB_AOS_LAYOUT_H_
#define INCLUDE_LATTICE_GENERIC_CBLAYOUT_H_

#include "MG_config.h"
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

/** Checkerboarded, Layout to hold CB grids of Objects
  *  e.g. LatticeInfo Objects
  *
  *  The layout is:    cb x cbsites
  *
  *
  */
  template<typename T>
  class GenericCBLayout {
  public:
	   	  GenericCBLayout(const GenericCBLayout& to_copy) :
	   		  _info(to_copy._info),  _n_sites(to_copy._n_sites),
			  _n_cb_sites(to_copy._n_cb_sites) {}

	   	  GenericCBLayout() = delete;

	  	  GenericCBLayout(const LatticeInfo& info) : _info(info),
	  		  _n_sites(info.GetNumSites()),
			  _n_cb_sites(info.GetNumCBSites()) {}

	  	  ~GenericCBLayout() {}


	  	  const LatticeInfo& GetLatticeInfo(void) const { return _info; }


	  	  /* Get the index of a site based on cb and site, with spin, color, etc indices */
	  	  inline
		  IndexType
		  ContainerIndex(IndexType cb,
				  	  	 IndexType cb_index) const {

	  		  /* cb = 0 => cb_offset = 0
	  		   * cb = 1 => cb_offset = numCBSites(0)
	  		   */
	  		  return cb_index + _n_cb_sites*cb;

	  	  }


	  	  inline
	  	  IndexType
		  ContainerIndex(IndexType site_index) const {

	  		  IndexType cb = site_index/_n_cb_sites;
	  		  IndexType cb_index = site_index % _n_cb_sites;
	  		  return ContainerIndex(cb,cb_index);


	  	  }

	  	  inline
		  IndexType
		  ContainerIndex(const IndexArray& coords) {

	  		  IndexType sum_coords = coords[0];
	  		  for(int mu=1; mu < n_dim; ++mu) sum_coords += coords[mu];
	  		  IndexType cb =  (sum_coords + _info.GetCBOrigin() ) & 1;

	  		  IndexArray cb_coords(coords);
	  		  cb_coords[0] /= 2;
	  		  IndexType cb_index = CoordsToIndex(cb_coords,_info.GetCBLatticeDimensions());

	  		return ContainerIndex(cb,cb_index);

	  	  }

	  	  inline
		  size_t DataNumElem() const {return n_checkerboard*_n_cb_sites;}

	  	  inline
		  size_t DataInBytes() const {return DataNumElem()*sizeof(T);}

  private:
	  	const LatticeInfo _info;
	  	IndexType _n_sites;
	  	IndexType _n_cb_sites;


  };







}




#endif /* INCLUDE_LATTICE_LATTICE_SPINOR_H_ */
