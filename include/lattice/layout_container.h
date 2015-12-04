/*
 * spinor.h
 *
 *  Created on: Oct 20, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_LAYOUT_CONTAINER_H_
#define INCLUDE_LATTICE_LAYOUT_CONTAINER_H_

#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/geometry_utils.h"
#include "lattice/buffer.h"
#include "lattice/layout_traits.h"
#include "utils/memory.h"
#include <memory>
#include <type_traits>

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



  template<typename T,        // Type in the body
  	  	   typename Layout>  // This allows layout to templated
  class GenericLayoutContainer {
  private:

	  const Layout _layout; // Layout
	  std::shared_ptr<Buffer<T>> _buffer; // We have shared ownership
	  T* _data; // Direct pointer into the data.

  public:
	  using layout_type  = Layout;
	  using base_type    = T;


	  /** Copy constructor. This will shallow copy a container, by simply copying all its members */
	  GenericLayoutContainer(const GenericLayoutContainer<T, Layout>& to_copy) :
			 _layout(to_copy._layout), _buffer(to_copy._buffer), _data(to_copy._data) {}

	  /** Allocating constructor */
	  GenericLayoutContainer(const Layout& layout, const MGUtils::MemorySpace Space=MGUtils::REGULAR)
	  : _layout(layout), _buffer(new Buffer<T>(layout.GetNumData(), Space)) {
		  _data = _buffer->GetData();
	  }

#if 0
	  /** Allocating constructor -- needs only lattice info */
	  GenericLayoutContainer(const LatticeInfo& info, const MGUtils::MemorySpace Space=MGUtils::REGULAR)
	  : _layout{info}, _buffer(new Buffer<T>(_layout.GetNumData(), Space)) {
		  _data = _buffer->GetData();
	  }
#endif

	  /* Used for creating a view over the container.
	   * The layout now refers to the layout of the view.
	   * We can pass an initial offset to where the view begins.
	   * NB: This would work in the situation where we have say 'blocks' of foo, and blocks run outermost
	   * It may not work if the blocks were to run innermost... So for recursive types only for now.
	   *
	   * Should this be a 'ContainerRef' to refer to a slice ?
	   */
	  GenericLayoutContainer(const Layout& layout, std::shared_ptr<Buffer<T>> buffer_in,
			  	  	 	 	 IndexType initial_offset) : _layout(layout), _buffer(buffer_in), _data(buffer_in->GetData(initial_offset)) {
	  }

	  virtual ~GenericLayoutContainer()
	  {

	  }

	  /** Using Variadic Template Args
	   *  To forward the args to the layout
	   *  NB: May need to take care of proper forwarding here using std::forward
	   *  NB: Layout has to support ContainerIndex with the forwarded Args
	   *      if it does not, that is a compilation error
	   */
	  template<typename ...Args>
	  inline
	  T&  Index(Args... args) {
		  return _data[ _layout.ContainerIndex(args...) ];
	  }

	  /** Using Variadic Template Args
	   *  To forward the args to the layout
	   *  NB: May need to take care of proper forwarding here using std::forward
	   *  NB: Layout has to support ContainerIndex with the forwarded Args
	   *      if it does not, that is a compilation error
	   */
	  template<typename ...Args>
	  inline
	  const T& Index(Args... args) const {
		  return _data[ _layout.ContainerIndex(args...) ];
	  }



	  inline
	  const  LatticeInfo& GetLatticeInfo() const {
		  return _layout.GetLatticeInfo();

	  }

	  inline
	  const Layout& GetLayout() const {
		  return _layout;
	  }

	  template<typename ...Args>
		  GenericLayoutContainer<T,	 typename LayoutTraits<Layout>::subview_layout_type>
		  GetSubview(Args... args)
		  {
			  IndexType offset = _layout.GetSubviewOffset(args...);
			  auto subview_layout = _layout.GetSubviewLayout(args...);
			  return GenericLayoutContainer<T,
					  	  typename LayoutTraits<Layout>::subview_layout_type>(subview_layout,
					  			  _buffer, offset);

		  }

  };


  /* Want specialized containers, where the user does not have to necessarily pass in the Layout.
   * We'd like the layout to be mostly hidden from the user
   */
  template<typename T, typename Layout>
  class LatticeLayoutContainer : public GenericLayoutContainer<T, Layout> {
  public:
	  LatticeLayoutContainer(const LatticeInfo& info, const MGUtils::MemorySpace Space=MGUtils::REGULAR) : GenericLayoutContainer<T,Layout>(Layout(info),Space) {}
	  ~LatticeLayoutContainer(){}
  };

  template<typename T, typename Layout>
  class AggregateLayoutContainer : public GenericLayoutContainer<T, Layout> {
   public:
 	  AggregateLayoutContainer(const LatticeInfo& info, const Aggregation& aggr,
 			  	  	  	  	  const MGUtils::MemorySpace Space=MGUtils::REGULAR) : GenericLayoutContainer<T,Layout>(Layout(info,aggr),Space) {}

 	  ~AggregateLayoutContainer(){};
   };

  template<typename T, typename Layout, typename Container>
  struct ContainerTraits {
	  typedef void base_type;
	  typedef void layout_type;
	  typedef void subview_container_type;
  };

  template<typename T, typename Layout>
  struct ContainerTraits< T, Layout, GenericLayoutContainer<T,Layout> > {
	  typedef T base_type;
	  typedef Layout layout_type;
	  typedef GenericLayoutContainer<T,typename LayoutTraits<Layout>::subview_layout_type> subview_container_type;
  };

}




#endif /* INCLUDE_LATTICE_LATTICE_SPINOR_H_ */
