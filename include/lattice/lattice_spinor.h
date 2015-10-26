/*
 * spinor.h
 *
 *  Created on: Oct 20, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_LATTICE_SPINOR_H_
#define INCLUDE_LATTICE_LATTICE_SPINOR_H_

#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/geometry_utils.h"
#include "utils/memory.h"
#include <memory>

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


  // This is a simple whole lattice layout
  // Ordering is:  N_complex (fastest ) x NColor x NSpin x Site (slowest)
  //
  // Layout is not checkerboarded, but a checkerboarded accessor is provided.
  //  Nothing exciting happens: No padding, no vectorization, nothing
  //
  template<typename T>
  class CompactSOASpinorLayout {
  public:
	  	  CompactSOASpinorLayout(const LatticeInfo& info) : _info(info),
		  	  _n_color(info.GetNumColors()),
			  _n_spin(info.GetNumSpins()),
			  _n_sites(info.GetNumSites()),
			  _lat_dims(info.GetLatticeDimensions()){}
	  	  ~CompactSOASpinorLayout() {}


	  	  const LatticeInfo& GetLatticeInfo(void) const { return _info; }


	  	  inline
	  	  unsigned int
		  ContainerIndex(unsigned int site_index,
				  	  	 unsigned int spin_index,
						 unsigned int color_index,
						 unsigned int reim) const {

	  		  return reim + n_complex*(color_index + _n_color*(spin_index + _n_spin*site_index));

	  	  }

	  	  inline
		  unsigned int
		  ContainerIndex(unsigned int cb,
				  	  	 unsigned int cb_index,
						 unsigned int spin_index,
						 unsigned int color_index,
						 unsigned int reim) const {

	  		  // Find the site
	  		  unsigned int global_site = _info.GetCBSiteTable(cb)[cb_index];

	  		  return ContainerIndex(global_site, spin_index, color_index, reim);
	  	  }

	  	  inline
		  unsigned int
		  ContainerIndex(const std::vector<unsigned int>& coords,
				  	  	 unsigned int spin_index,
						 unsigned int color_index,
						 unsigned int reim) {
	  		  unsigned int global_site = CoordsToIndex(coords, _lat_dims);
	  		  return ContainerIndex(global_site, spin_index, color_index, reim);

	  	  }

		  size_t DataNumElem() const {return  n_complex*_n_spin*_n_color*_n_sites;}

		  size_t DataInBytes() const {return  DataNumElem()*sizeof(T);}

		  size_t GhostNumElem(unsigned int cb,
				  	  	  	  unsigned int dir,
							  unsigned int forw_back) const {
			  return n_complex*_n_spin*_n_color*_info.GetNumCBSurfaceSites(dir, forw_back, cb);
		  }

		  size_t GhostNumBytes(unsigned int cb,
				  	  	  	   unsigned int dir,
				  	  	  	   unsigned int forw_back) const { return GhostNumElem(cb,dir,forw_back)*sizeof(T); }

  private:
	  	const LatticeInfo& _info;
	  	unsigned int _n_color;
	  	unsigned int _n_spin;
	  	unsigned int _n_sites;
	  	const std::vector<unsigned int>& _lat_dims;

  };


  /** This is a checkerboarded layout
   *  It is still compact as it will be site major
   *  I will always store 'checkerboard even' (followed by 'checkerboard odd')
   *
   *  The lattice info site tables are not really much use to me here.
   *
   */
  template<typename T>
   class CompactCBSOASpinorLayout {
   public:
 	  	  CompactCBSOASpinorLayout(const LatticeInfo& info) : _info(info),
 		  	  _n_color(info.GetNumColors()),
 			  _n_spin(info.GetNumSpins()),
 			  _n_sites(info.GetNumSites()),
 			  _lat_dims(info.GetLatticeDimensions()){}
 	  	  ~CompactCBSOASpinorLayout() {}


 	  	  const LatticeInfo& GetLatticeInfo(void) const { return _info; }


 	  	  /* Get the index of a site based on cb and site, with spin, color, etc indices */
 	  	  inline
 		  unsigned int
 		  ContainerIndex(unsigned int cb,
 				  	  	 unsigned int cb_index,
 						 unsigned int spin_index,
 						 unsigned int color_index,
 						 unsigned int reim) const {

 	  		  /* cb = 0 => cb_offset = 0
 	  		   * cb = 1 => cb_offset = numCBSites(0)
 	  		   */
 	  		  unsigned int cb_offset = cb == 0 ? 0 : _info.GetNumCBSites(0);
 	  		  return reim + n_complex*(color_index+
 	  				  _n_color*(spin_index + _n_spin*(cb_index + cb_offset)));
 	  	  }


	  	  inline
 	  	  unsigned int
 		  ContainerIndex(unsigned int site_index,
 				  	  	 unsigned int spin_index,
 						 unsigned int color_index,
 						 unsigned int reim) const {


 	  		  return reim + n_complex*(color_index + _n_color*(spin_index + _n_spin*site_index));

 	  	  }

 	  	  inline
 		  unsigned int
 		  ContainerIndex(const std::vector<unsigned int>& coords,
 				  	  	 unsigned int spin_index,
 						 unsigned int color_index,
 						 unsigned int reim) {

 	  		  unsigned int sum_coords = coords[0];
 	  		  for(int mu=1; mu < n_dim; ++mu) sum_coords += coords[mu];
 	  		  unsigned int cb = ( (sum_coords & 1) + _info.GetCBOrigin() ) & 1;

 	  		  std::vector<unsigned int> cb_coords(coords);
 	  		  cb_coords[0] /= 2;
 	  		  unsigned int cb_index = CoordsToIndex(cb_coords,_info.cb_dims);

 	  		return ContainerIndex(cb,cb_index, spin_index, color_index, reim);

 	  	  }

 		  size_t DataNumElem() const {return  n_complex*_n_spin*_n_color*_n_sites;}

 		  size_t DataInBytes() const {return  DataNumElem()*sizeof(T);}

 		  size_t GhostNumElem(unsigned int cb,
 				  	  	  	  unsigned int dir,
 							  unsigned int forw_back) const {
 			  return n_complex*_n_spin*_n_color*_info.GetNumCBSurfaceSites(dir, forw_back, cb);
 		  }

 		  size_t GhostNumBytes(unsigned int cb,
 				  	  	  	   unsigned int dir,
 				  	  	  	   unsigned int forw_back) const { return GhostNumElem(cb,dir,forw_back)*sizeof(T); }

   private:
 	  	const LatticeInfo& _info;
 	  	unsigned int _n_color;
 	  	unsigned int _n_spin;
 	  	unsigned int _n_sites;
 	  	const std::vector<unsigned int>& _lat_dims;

   };


  template<typename T,        // Type in the body
  	  	   typename Layout = CompactSOASpinorLayout<T>,
		   const MGUtils::MemorySpace Space = MGUtils::REGULAR>
  class GeneralLatticeSpinor {
  public:

	GeneralLatticeSpinor(const LatticeSpinor<T, Layout, Space>& to_copy) :
			_layout(to_copy._layout), _info( to_copy._info), _data(to_copy._data)
  {
		for (int cb = 0; cb < n_checkerboard; ++cb) {
			for (int dim = 0; dim < n_dim; ++dim) {
				for (int dir = BACKWARD; dir <= FORWARD; ++dir) {
					_ghost_send[cb][dim][dir] =
							to_copy._ghost_send[cb][dim][dir];
					_ghost_recv[cb][dim][dir] =
							to_copy._ghost_recv[cb][dim][dir];
				}
			}
		}
	}

	 GeneralLatticeSpinor(const Layout& layout) : _layout(layout), _info(layout.GetLatticeInfo())
  	  {
#pragma omp master
		  {
			  // Master thread allocates -- using MGUtils Allocator
			  _data = reinterpret_cast<T*>(MGUtils::MemoryAllocate(layout.DataInBytes(),Space));

			  // Master thread allocates the Ghost Zones
			  for(int cb=0; cb < n_checkerboard; ++cb) {
				  for(int dim=0; dim < n_dim; ++dim) {
					  for(int dir=BACKWARD; dir <= FORWARD; ++dir) {
						  _ghost_send[cb][dim][dir]=reinterpret_cast<T*>(MGUtils::MemoryAllocate(layout.GhostNumBytes(cb,dim,dir),Space));
						  _ghost_recv[cb][dim][dir]=reinterpret_cast<T*>(MGUtils::MemoryAllocate(layout.GhostNumBytes(cb,dim,dir),Space));
					  }
				  }
			  }
		  }
#pragma omp barrier
  	  }


	  ~GeneralLatticeSpinor()
	  {
#pragma omp master
		  {
			  MGUtils::MemoryFree(_data);
		  	  for(int cb=0; cb < n_checkerboard; ++cb) {
			  	for(int dim=0; dim < n_dim; ++dim) {
				  for(int dir=BACKWARD; dir <= FORWARD; ++dir) {
					  MGUtils::MemoryFree(_ghost_send[cb][dir][dim]);
					  MGUtils::MemoryFree(_ghost_recv[cb][dir][dim]);
				  }
			  	}
		  	  }
		  }
#pragma omp barrier
	  }

	  // FIXME: In multi-threaded world this is dangerous
	  // as there is nothing to stop two threads from getting the same writeable
	  // reference.. For now it will be programmer specific
	  inline
	  T& operator()(int elem) __attribute__((always_inline)) {
		  return _data[ elem ];
	  }

	  const T& operator()(int elem) const __attribute((always_inline)) {
		  return _data[ elem ];
	  }

	  inline
	  T&  Index(unsigned int elem, unsigned int spin, unsigned int color, unsigned int reim) {
		  return _data[ _layout.ContainerIndex(elem,spin,color,reim) ];
	  }

	  inline
	  const T& Index(unsigned int elem, unsigned int spin, unsigned int color, unsigned int reim) const {
		  return _data[ _layout.ContainerIndex(elem,spin,color,reim) ];
	  }

	  const LatticeInfo& GetLatticeInfo(void) {
		  return _info;

	  }
  private:
	  const Layout<T>& _layout;
	  const LatticeInfo& _info; // Get from Layout
	  std::shared_ptr<T> _data;
	  std::shared_ptr<T> _ghost_send[2][n_dim][n_forw_back];
	  std::shared_ptr<T> _ghost_recv[2][n_dim][n_forw_back];

  };

  using LatticeSpinorF = GeneralLatticeSpinor<float,CompactSOASpinorLayout<float>, MGUtils::REGULAR>;



}




#endif /* INCLUDE_LATTICE_LATTICE_SPINOR_H_ */
