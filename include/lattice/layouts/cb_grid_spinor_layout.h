/*
 * spinor.h
 *
 *  Created on: Oct 20, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_CB_SOA_SPINOR_LAYOUT_H_
#define INCLUDE_LATTICE_CB_SOA_SPINOR_LAYOUT_H_

#include "MG_config.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/geometry_utils.h"
#include "lattice/virtual_node.h"
#include "lattice/layout_traits.h"

#include "utils/print_utils.h"
namespace MG {

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



  /** Checkerboarded, SOA Spinor Layout
   *
   *  The layout is:   spins (slowest) x colors x cb x cbsites x complex
   *
   * NB: For now, let me forget about padding
   *
   */
   template<typename T, typename VNode = VN_Scalar<T>>
   class CBGridSpinorLayout {
   public:


	   	  /* Construct from a lattice info */
 	  	  CBGridSpinorLayout(const LatticeInfo& info) : _info(info),
 		  	  _n_color(info.GetNumColors()),
 			  _n_spin(info.GetNumSpins()),
 			  _n_sites(info.GetNumSites()),
			  _n_cb_sites(info.GetNumCBSites()) {

 	  		  /* Want to check that checkerboarding is not messed up:
 	  		   * Lattice Info x_dim has to be divisible by 4 (2 for checkerboarding and then even)
 	  		   *
 	  		   */
 	  		   _outer_dims = _info.GetCBLatticeDimensions();

 	  		   /* Lattice Info has already checked that checkerboarding restrictions
 	  		    * are met. Now I need to check also that where the VNode grid is not local
 	  		    * (i.e. any dimension < VNode::n_dim) is also even (even after checkerboarding)
 	  		    * to make sure the checkerboarding is not messed up.
 	  		    * In the scalar case, VNode::n_dim is 0 so this check will be just skipped over
 	  		    */
 	  		   for(IndexType mu = 0;  mu < VNode::n_dim; ++mu) {
 	  			   if ( _outer_dims[mu] % 2 != 0 ) {
 	  				   MG::MasterLog(MG::ERROR, "Dim %u needs to be even after checkerboarding for Grid Layout to work. It is %u", mu, dims[mu] );
 	  			   }
 	  			   else {
 	  				   _outer_dims[mu] /= 2;
 	  			   }
 	  		   }

 	  	  }


 	  	  /* Destruct */
 	  	  ~CBGridSpinorLayout() {}


 	  	  /* Query lattice info */
 	  	  const LatticeInfo& GetLatticeInfo(void) const { return _info; }

 	  	  const IndexArray& GetOuterDimensions(void) const { return _outer_dims; }

 	  	  /* Get the index of a site based on cb and site, with spin, color, etc indices */
 	  	  inline
 		  IndexType
 		  ContainerIndex(IndexType cb,
 				  	  	 IndexType cb_index,
 						 IndexType spin_index,
 						 IndexType color_index,
 						 IndexType reim) const {

 	  		  /* Running order
 	  		   * inner_site_re + VN::n_sites*(inner_site_im
 	  		   * 						     + n_complex*(spin
 	  		   * 						                  + n_spin*(color
 	  		   * 						                  	+ n_color*( outer_cbsite
 	  		   * 						                  	  + n_cbsites * cb )
 	  		   * 						                  	  )
 	  		   * 						                  	  )
 	  		   * 						                  	  )
 	  		   */



 	  		  	IndexType osite = cb_index >> VNode::n_dim;
 	  		  	IndexType isite = cb_index & VNode::mask;

 	  		  	return isite
 	  		  			+VNode::n_sites*(reim
 	  		  				+ n_complex*(spin_index
 	  		  					+ _n_spin*(color_index
 	  		  						+ _n_color*(osite +
										_n_cb_sites*cb))));

 	  	  }


	  	  inline
 	  	  IndexType
 		  ContainerIndex(IndexType site_index,
 				  	  	 IndexType spin_index,
 						 IndexType color_index,
 						 IndexType reim) const {

	  		  IndexType cb = site_index/_n_cb_sites;
	  		  IndexType cb_index = site_index % _n_cb_sites;
	  		  return ContainerIndex(cb,cb_index, spin_index, color_index, reim);


 	  	  }

#if 1
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
#endif

 	  	  // FIXME: DataNumElem can be useful for iterating, but NOT FOR MEMORY ALLOCATION
 	  	  //
 	  	  inline
 		  size_t GetNumData() const {return  n_complex*_n_spin*_n_color*n_checkerboard*_n_cb_sites_stride;}

 	  	  inline
 		  size_t GetDataInBytes() const {return GetNumData()*sizeof(T);}



   private:
 	  	const LatticeInfo _info;
 	  	IndexType _n_color;
 	  	IndexType _n_spin;
 	  	IndexType _n_sites;

 	  	IndexType _n_cb_sites;
 	  	IndexArray _outer_dims;
   };


   template<>
   struct LayoutTraits<CBSOASpinorLayout<float>>
   {
	   typedef float value_type;
	   const bool has_subviews=false;
	   typedef void subview_layout_type;

   };

   template<>
      struct LayoutTraits<CBSOASpinorLayout<double>>
      {
   	   typedef double value_type;
   	   const bool has_subviews=false;
	   typedef void subview_layout_type;

      };

   template<>
      struct LayoutTraits<CBSOASpinorLayout<IndexType>>
      {
   	   typedef float value_type;
   	   const bool has_subviews=false;
	   typedef void subview_layout_type;

      };



}




#endif /* INCLUDE_LATTICE_LATTICE_SPINOR_H_ */
