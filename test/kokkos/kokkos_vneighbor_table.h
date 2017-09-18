#ifndef KOKKOS_VNEIGHBOR_TABLE
#define KOKKOS_VNEIGHBOR_TABLE

#include "MG_config.h"
#include "Kokkos_Core.hpp"

namespace MG { 

#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
   template<typename VN>
   void ComputeSiteTable(int _n_xh, int _n_x, int _n_y, int _n_z, int _n_t,  Kokkos::View<Kokkos::pair<int,typename VN::MaskType>*[2][8],NeighLayout, MemorySpace> _table) {
		int num_sites =  _n_xh*_n_y*_n_z*_n_t;
			Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,num_sites), KOKKOS_LAMBDA(int site) {
		        for(int target_cb=0; target_cb < 2; ++target_cb) {
			     // Break down site index into xcb, y,z and t
			     IndexType tmp_yzt = site / _n_xh;
			     IndexType xcb = site - _n_xh * tmp_yzt;
			     IndexType tmp_zt = tmp_yzt / _n_y;
			     IndexType y = tmp_yzt - _n_y * tmp_zt;
			     IndexType t = tmp_zt / _n_z;
			     IndexType z = tmp_zt - _n_z * t;

			     // Global, uncheckerboarded x, assumes cb = (x + y + z + t ) & 1
			     IndexType x = 2*xcb + ((target_cb+y+z+t)&0x1);

			     if( t > 0 ) {
			       _table(site,target_cb,T_MINUS) = Kokkos::make_pair(xcb + _n_xh*(y + _n_y*(z + _n_z*(t-1))),VN::NoPermuteMask);
			     }
			     else {
			       _table(site,target_cb,T_MINUS) = Kokkos::make_pair(xcb + _n_xh*(y + _n_y*(z + _n_z*(_n_t-1))),VN::TPermuteMask);
			     }

			     if( z > 0 ) {
			       _table(site,target_cb,Z_MINUS) = Kokkos::make_pair( xcb + _n_xh*(y + _n_y*((z-1) + _n_z*t)), VN::NoPermuteMask);
			     }
			     else {
			       _table(site,target_cb,Z_MINUS) = Kokkos::make_pair( xcb + _n_xh*(y + _n_y*((_n_z-1) + _n_z*t)),VN::ZPermuteMask);
			     }

			     if( y > 0 ) {
			       _table(site,target_cb,Y_MINUS) = Kokkos::make_pair( xcb + _n_xh*((y-1) + _n_y*(z + _n_z*t)), VN::NoPermuteMask);
			     }
			     else {
			       _table(site,target_cb,Y_MINUS) = Kokkos::make_pair( xcb + _n_xh*((_n_y-1) + _n_y*(z + _n_z*t)),VN::YPermuteMask);
			     }

			     if ( x > 0 ) {
			       _table(site,target_cb,X_MINUS)= Kokkos::make_pair(((x-1)/2) + _n_xh*(y + _n_y*(z + _n_z*t)), VN::NoPermuteMask);
			     }
			     else {
			       _table(site,target_cb,X_MINUS)= Kokkos::make_pair(((_n_x-1)/2) + _n_xh*(y + _n_y*(z + _n_z*t)),VN::XPermuteMask);
			     }

			     if ( x < _n_x - 1 ) {
			       _table(site,target_cb,X_PLUS) = Kokkos::make_pair(((x+1)/2)  + _n_xh*(y + _n_y*(z + _n_z*t)),VN::NoPermuteMask);
			     }
			     else {
			       _table(site,target_cb,X_PLUS) = Kokkos::make_pair(0 + _n_xh*(y + _n_y*(z + _n_z*t)),VN::XPermuteMask);
			     }

			     if( y < _n_y-1 ) {
			       _table(site,target_cb,Y_PLUS) = Kokkos::make_pair(xcb + _n_xh*((y+1) + _n_y*(z + _n_z*t)),VN::NoPermuteMask);
			     }
			     else {
			       _table(site,target_cb,Y_PLUS) = Kokkos::make_pair(xcb + _n_xh*(0 + _n_y*(z + _n_z*t)),VN::YPermuteMask);
			     }

			     if( z < _n_z-1 ) {
			       _table(site,target_cb,Z_PLUS) = Kokkos::make_pair(xcb + _n_xh*(y + _n_y*((z+1) + _n_z*t)),VN::NoPermuteMask);
			     }
			     else {
			       _table(site,target_cb,Z_PLUS) = Kokkos::make_pair(xcb + _n_xh*(y + _n_y*(0 + _n_z*t)), VN::ZPermuteMask);
			     }

			     if( t < _n_t-1 ) {
			       _table(site,target_cb,T_PLUS) = Kokkos::make_pair(xcb + _n_xh*(y + _n_y*(z + _n_z*(t+1))),VN::NoPermuteMask);
			     }
			     else {
			       _table(site,target_cb,T_PLUS) = Kokkos::make_pair(xcb + _n_xh*(y + _n_y*(z + _n_z*(0))),VN::TPermuteMask);
			     }
			    } // target CB
		        });

	}
#endif


template<typename VN>
class SiteTable {
public:


	  SiteTable(int n_xh,
		    int n_y,
		    int n_z,
		    int n_t) : 
	 _n_x(2*n_xh),
	 _n_xh(n_xh),
	 _n_y(n_y),
	 _n_z(n_z),
	 _n_t(n_t) {

#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
	   _table = Kokkos::View<Kokkos::pair<int,typename VN::MaskType>*[2][8],NeighLayout,MemorySpace>("table", n_xh*n_y*n_z*n_t);
	   ComputeSiteTable<VN>(n_xh, 2*n_xh, n_y, n_z, n_t, _table);
#endif
	}

	  KOKKOS_FORCEINLINE_FUNCTION
	  	int  coords_to_idx(const int& xcb, const int& y, const int& z, const int& t) const
	  	{
	  	  return xcb+_n_xh*(y + _n_y*(z + _n_z*t));
	  	}


#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborTMinus(int site, int target_cb, int& n_idx, typename VN::MaskType& mask) const {
		const Kokkos::pair<int,typename VN::MaskType>& lookup = _table(site,target_cb,T_MINUS);
		n_idx = lookup.first;
		mask = lookup.second;
	}

	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborTPlus(int site, int target_cb, int& n_idx, typename VN::MaskType& mask) const {
		const Kokkos::pair<int,typename VN::MaskType>& lookup = _table(site,target_cb,T_PLUS);
		n_idx = lookup.first;
		mask = lookup.second;
	}

	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborZMinus(int site, int target_cb, int& n_idx, typename VN::MaskType& mask) const {
		const Kokkos::pair<int,typename VN::MaskType>& lookup = _table(site,target_cb,Z_MINUS);
		n_idx = lookup.first;
		mask = lookup.second;
	}

	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborZPlus(int site, int target_cb, int& n_idx, typename VN::MaskType& mask) const {
		const Kokkos::pair<int,typename VN::MaskType>& lookup = _table(site,target_cb,Z_PLUS);
		n_idx = lookup.first;
		mask = lookup.second;
	}

	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborYMinus(int site, int target_cb, int& n_idx, typename VN::MaskType& mask) const {
		const Kokkos::pair<int,typename VN::MaskType>& lookup = _table(site,target_cb,Y_MINUS);
		n_idx = lookup.first;
		mask = lookup.second;
	}

	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborYPlus(int site, int target_cb, int& n_idx, typename VN::MaskType& mask) const {
		const Kokkos::pair<int,typename VN::MaskType>& lookup = _table(site,target_cb,Y_PLUS);
		n_idx = lookup.first;
		mask = lookup.second;
	}

	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborXMinus(int site, int target_cb, int& n_idx, typename VN::MaskType& mask) const {
		const Kokkos::pair<int,typename VN::MaskType>& lookup = _table(site,target_cb,X_MINUS);
		n_idx = lookup.first;
		mask = lookup.second;
	}

	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborXPlus(int site, int target_cb, int& n_idx, typename VN::MaskType& mask) const {
		const Kokkos::pair<int,typename VN::MaskType>& lookup = _table(site,target_cb,X_PLUS);
		n_idx = lookup.first;
		mask = lookup.second;
	}



#else


	KOKKOS_FORCEINLINE_FUNCTION
	void idx_to_coords(int site, int& xcb, int& y, int& z, int& t) const
	{
		IndexType tmp_yzt = site / _n_xh;
		xcb = site - _n_xh * tmp_yzt;
		IndexType tmp_zt = tmp_yzt / _n_y;
		y = tmp_yzt - _n_y * tmp_zt;
		t = tmp_zt / _n_z;
		z = tmp_zt - _n_z * t;
	}

	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborTMinus(int xcb, int y, int z, int t, int& n_idx, typename VN::MaskType& mask) const {
		if( t >  0) {
			n_idx=xcb + _n_xh*(y + _n_y*(z + _n_z*(t-1)));
			mask=VN::NoPermuteMask;
		}
		else {
			n_idx = xcb + _n_xh*(y + _n_y*(z + _n_z*(_n_t-1)));
			mask=VN::TPermuteMask;
		}
	}

	KOKKOS_FORCEINLINE_FUNCTION
	 void NeighborZMinus(int xcb, int y, int z, int t, int& n_idx, typename VN::MaskType& mask) const {
		if( z >  0 ) {
			n_idx=xcb + _n_xh*(y + _n_y*((z-1) + _n_z*t));
		    mask=VN::NoPermuteMask;
		}
		else {
			n_idx=xcb + _n_xh*(y + _n_y*((_n_z-1) + _n_z*t));
			mask=VN::ZPermuteMask;
		}
	}


	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborYMinus(int xcb, int y, int z, int t, int& n_idx, typename VN::MaskType& mask) const {
		if ( y > 0 ) {
			n_idx = xcb + _n_xh*((y-1) + _n_y*(z + _n_z*t));
			mask  = VN::NoPermuteMask;
		}
		else {
			n_idx = xcb + _n_xh*((_n_y-1) + _n_y*(z + _n_z*t));
			mask  = VN::YPermuteMask;
		}
	}


	KOKKOS_FORCEINLINE_FUNCTION
	void NeighborXMinus(int xcb, int y, int z, int t, int target_cb, int& n_idx, typename VN::MaskType& mask) const {
		int x = 2*xcb + ((target_cb+y+z+t)&0x1);
		if ( x > 0 ) {
				n_idx = ((x-1)/2) + _n_xh*(y + _n_y*(z + _n_z*t));
				mask  = VN::NoPermuteMask;
		}
		else {
				n_idx = ((_n_x-1)/2) + _n_xh*(y + _n_y*(z + _n_z*t));
				mask = VN::XPermuteMask;
		}
	}


	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborXPlus(int xcb, int y, int z, int t, int target_cb, int& n_idx, typename VN::MaskType& mask) const {
		int x = 2*xcb + ((target_cb+y+z+t)&0x1);
		if ( x < _n_x - 1) {
			n_idx = ((x+1)/2)  + _n_xh*(y + _n_y*(z + _n_z*t));
			mask = VN::NoPermuteMask;
		}
		else {
			n_idx = _n_xh*(y + _n_y*(z + _n_z*t));
			mask = VN::XPermuteMask;

		}
	}



	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborYPlus(int xcb, int y, int z, int t, int& n_idx, typename VN::MaskType& mask) const {
		if (y < _n_y - 1) {
			n_idx = xcb + _n_xh*((y+1) + _n_y*(z + _n_z*t));
			mask = VN::NoPermuteMask;
		}
		else {
			n_idx = xcb + _n_xh*(0 + _n_y*(z + _n_z*t));
			mask = VN::YPermuteMask;
		}
	}


	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborZPlus(int xcb, int y, int z, int t, int& n_idx, typename VN::MaskType& mask) const {


		if(z < _n_z - 1) {
			n_idx = xcb + _n_xh*(y + _n_y*((z+1) + _n_z*t));
			mask = VN::NoPermuteMask;
		}
		else {
		    n_idx = xcb + _n_xh*(y + _n_y*(0 + _n_z*t));
		    mask = VN::ZPermuteMask;
		}
	}


	KOKKOS_FORCEINLINE_FUNCTION
	  void NeighborTPlus(int xcb, int y, int z, int t, int& n_idx, typename VN::MaskType& mask) const {

		if (t < _n_t - 1) {
			n_idx = xcb + _n_xh*(y + _n_y*(z + _n_z*(t+1)));
			mask = VN::NoPermuteMask;
		}
		else {
			n_idx = xcb + _n_xh*(y + _n_y*(z + _n_z*(0)));
			mask = VN::TPermuteMask;
		}
	}
#endif


	KOKKOS_INLINE_FUNCTION
	  SiteTable( const SiteTable& st):
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
	  _table(st._table),
#endif
	  _n_x(st._n_x),
	  _n_xh(st._n_xh),
	  _n_y(st._n_y),
	  _n_z(st._n_z),
	  _n_t(st._n_t) {}

	KOKKOS_INLINE_FUNCTION
	  SiteTable& operator=(const  SiteTable& st) {
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
	  _table = st._table;
#endif
	  _n_x = st._n_x;
	  _n_xh = st._n_xh;
	  _n_y = st._n_y;
	  _n_z = st._n_z;
	  _n_t = st._n_t;

	  return *this;
	}

private:
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
	Kokkos::View<Kokkos::pair<int,typename VN::MaskType>*[2][8], NeighLayout,MemorySpace > _table;
#endif
       int _n_x;
       int _n_xh;
       int _n_y;
       int _n_z;
       int _n_t;

};

} // Namespace MG

#endif

