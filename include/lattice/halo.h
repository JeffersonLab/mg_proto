/*
 * spinor_halo.h
 *
 *  Created on: Mar 7, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_HALO_H_
#define INCLUDE_LATTICE_HALO_H_

#include "MG_config.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/lattice_info.h"
#include "lattice/geometry_utils.h"
#include <omp.h>
#if defined(MG_QMP_COMMS)
#include "lattice/halo_container_qmp.h"
#else
#include "lattice/halo_container_single.h"
#endif
namespace MG  {

using SpinorHaloCB = HaloContainer<CoarseSpinor>;
using CoarseGaugeHaloCB = HaloContainer<CoarseSpinor>;

template<typename T>
struct CoarseAccessor
{
  inline
  static
  const float* get(const T& in, int cb, int cbsite, int dir, int fb);
};

template<>
inline
const float*
CoarseAccessor<CoarseSpinor>::get(const CoarseSpinor& in, int cb, int cbsite, int dir, int fb)
{
	return in.GetSiteDataPtr(cb,cbsite);
}

#if 0
template<>
inline
const float*
CoarseAccessor<CoarseGauge>::get(const CoarseGauge& in, int cb, int cbsite, int dir, int fb)
{
    int mu = 8;
    return in.GetSiteDirADDataPtr(cb,cbsite,mu);
}
#endif

template<typename T, template <typename> class Accessor>
inline
void
packFace( HaloContainer<T>& halo, const T& in, IndexType cb,  IndexType dir, IndexType fb)
{
	const LatticeInfo& info = in.GetInfo();
	const IndexArray& latt_dims = info.GetLatticeDimensions();
	const IndexArray& latt_cb_dims = info.GetCBLatticeDimensions();
	IndexArray coords;



	// Grab the buffer from the Halo
	float* buffer = halo.GetSendToDirBuf(2*dir + fb);

	//int num_color_spins = info.GetNumColorSpins();
	int buffer_site_offset = halo.GetDataTypeSize();
	int buffer_sites = halo.NumSitesInFace(dir);

	// Loop through the sites in the buffer
#pragma omp for
	for(int site =0; site < buffer_sites; ++site) {
		int local_cb = (cb  + info.GetCBOrigin())&1;

		// I need to convert the face site index
		// into a body site index in the required checkerboard.
		coords[dir]= (fb == MG_BACKWARD ) ? 0 : latt_dims[dir]-1;
		if( dir == 0 ) {
			// X direction is special
			IndexArray x_cb_dims(latt_cb_dims); x_cb_dims[Y_DIR]/=2;

			IndexToCoords3(site,x_cb_dims,X_DIR,coords);
			coords[Y_DIR] *= 2;
			coords[Y_DIR] += ((local_cb + coords[X_DIR]+coords[Z_DIR] + coords[T_DIR])&1);
			coords[X_DIR] /=2; // Convert back to checkerboarded X_coord
		}
		else {
			// The Muth coordinate is eithe 0, or the last coordinate
			IndexToCoords3(site,latt_cb_dims,dir,coords);
		}
		int body_site = CoordsToIndex(coords,latt_cb_dims);
		float* buffersite = &buffer[site*buffer_site_offset];
		// Grab the body site
		const float* bodysite = Accessor<T>::get(in,cb,body_site,dir,fb);

		// Copy body site into buffer site
		// This is likely to be done in a thread, so
		// use SIMD if you can.
#pragma omp simd
		for(int cspin_idx=0; cspin_idx < halo.GetDataTypeSize(); ++cspin_idx) {
			buffersite[cspin_idx] = bodysite[cspin_idx];
		} // Finish copying

	} // finish loop over sites.

}



template<typename T, template <typename> class Accessor>
inline
void
CommunicateHaloSyncInOMPParallel(HaloContainer<T>& halo, const T& in, const int target_cb)
{

	halo.setNCols(in.GetNCol());
	if( halo.NumNonLocalDirs() > 0 ) {
		for(int mu=0; mu < n_dim; ++mu) {
			// Pack face usese omp for internally
			if ( ! halo.LocalDir(mu) ) {
				packFace<T,Accessor>(halo,in,1-target_cb,mu,MG_BACKWARD);
				packFace<T,Accessor>(halo,in,1-target_cb,mu,MG_FORWARD);
			}
		}

	// Make sure faces are packed
#pragma omp barrier

	// master thread does the comms -- do this better later
#pragma omp master
		{
			halo.StartAllRecvs();
			halo.StartAllSends();
			halo.FinishAllSends();
			halo.FinishAllRecvs();
		}

	// Barrier after comms to sync master with other threads
#pragma omp barrier
	}
}


template<typename T, template <typename> class Accessor>
inline
void
CommunicateHaloSync(HaloContainer<T>& halo, const T& in, const int target_cb)
{
	halo.setNCols(in.GetNCol());
	if( halo.NumNonLocalDirs() > 0 ) {
		for(int mu=0; mu < n_dim; ++mu) {
			// Pack face usese omp for internally
			if ( ! halo.LocalDir(mu) ) {
				packFace<T,Accessor>(halo,in,1-target_cb,mu,MG_BACKWARD);
				packFace<T,Accessor>(halo,in,1-target_cb,mu,MG_FORWARD);
			}
		}

		halo.StartAllRecvs();
		halo.StartAllSends();
		halo.FinishAllSends();
		halo.FinishAllRecvs();
	}
}


template<typename T, template <typename> class Accessor>
inline
const float*
GetNeighborXPlus(const HaloContainer<T>& halo, const T& in, int x, int y, int z, int t, int source_cb)
{
	const IndexType dir = X_DIR;
	const IndexType fb = MG_FORWARD;

	if ( x < in.GetNx() - 1 ) {

		return Accessor<T>::get(in,source_cb, ((x+1)/2)  + in.GetNxh()*(y + in.GetNy()*(z + in.GetNz()*t)), dir,fb);
	}
	else {

		if(halo.LocalDir(X_DIR) )  {
			return Accessor<T>::get(in,source_cb, 0 + in.GetNxh()*(y + in.GetNy()*(z + in.GetNz()*t)), dir, fb);
		}
		else {

			int index = halo.GetDataTypeSize()*((y + in.GetNy()*(z + in.GetNz()*t))/2);
			return  &( halo.GetRecvFromDirBuf(2*X_DIR + MG_FORWARD)[index]);
		}
	}
}

template<typename T, template <typename> class Accessor>
inline
const float*
GetNeighborXMinus(const HaloContainer<T>& halo, const T& in, int x, int y, int z, int t, int source_cb)
{
	const IndexType dir = X_DIR;
	const IndexType fb = MG_BACKWARD;

	if ( x > 0 ) {
		return Accessor<T>::get(in,source_cb, ((x-1)/2) + in.GetNxh()*(y + in.GetNy()*(z + in.GetNz()*t)), dir, fb);
	}
	else {
		if ( halo.LocalDir(X_DIR) ) {
			return Accessor<T>::get(in,source_cb, ((in.GetNx()-1)/2) + in.GetNxh()*(y + in.GetNy()*(z + in.GetNz()*t)), dir,fb);
		}
		else {
			// Get the buffer

			int index = halo.GetDataTypeSize()*((y + in.GetNy()*(z + in.GetNz()*t))/2);

			return  &(halo.GetRecvFromDirBuf(2*X_DIR + MG_BACKWARD)[index]);
		}
	}
}

template<typename T, template <typename> class Accessor>
inline
const float*
GetNeighborYPlus(const HaloContainer<T>& halo, const T& in, int x_cb, int y, int z, int t, int source_cb)
{
	const IndexType dir = Y_DIR;
	const IndexType fb = MG_FORWARD;

	if ( y < in.GetNy() - 1 ) {

		return Accessor<T>::get(in,source_cb, x_cb+ in.GetNxh()*((y+1) + in.GetNy()*(z + in.GetNz()*t)), dir, fb);
	}
	else {

		if(halo.LocalDir(Y_DIR) )  {
			return Accessor<T>::get(in,source_cb, x_cb + in.GetNxh()*(0 + in.GetNy()*(z + in.GetNz()*t)), dir, fb);
		}
		else {

			int index = halo.GetDataTypeSize()*(x_cb + in.GetNxh()*(z + in.GetNz()*t));
			return  &( halo.GetRecvFromDirBuf(2*Y_DIR + MG_FORWARD)[index]);
		}
	}
}

template<typename T, template <typename> class Accessor>
inline
const float*
GetNeighborYMinus(const HaloContainer<T>& halo, const T& in, int x_cb, int y, int z, int t, int source_cb)
{
	const IndexType dir = Y_DIR;
	const IndexType fb = MG_BACKWARD;

	if ( y > 0  ) {

		return Accessor<T>::get(in,source_cb, x_cb+ in.GetNxh()*((y-1) + in.GetNy()*(z + in.GetNz()*t)), dir, fb);
	}
	else {

		if(halo.LocalDir(Y_DIR) )  {
			return Accessor<T>::get(in,source_cb, x_cb + in.GetNxh()*((in.GetNy()-1) + in.GetNy()*(z + in.GetNz()*t)),dir,fb);
		}
		else {

			int index = halo.GetDataTypeSize()*(x_cb + in.GetNxh()*(z + in.GetNz()*t));
			return  &( halo.GetRecvFromDirBuf(2*Y_DIR + MG_BACKWARD)[index]);
		}
	}
}

template<typename T, template <typename> class Accessor>
inline
const float*
GetNeighborZPlus(const HaloContainer<T>& halo, const T& in, int x_cb, int y, int z, int t, int source_cb)
{
	const IndexType dir = Z_DIR;
	const IndexType fb = MG_FORWARD;

	if ( z < in.GetNz() - 1 ) {

		return Accessor<T>::get(in,source_cb, x_cb+ in.GetNxh()*(y + in.GetNy()*((z+1) + in.GetNz()*t)), dir, fb);
	}
	else {

		if(halo.LocalDir(Z_DIR) )  {
			return Accessor<T>::get(in,source_cb, x_cb + in.GetNxh()*(y + in.GetNy()*(0 + in.GetNz()*t)),dir,fb);
		}
		else {

			int index = halo.GetDataTypeSize()*(x_cb + in.GetNxh()*(y + in.GetNy()*t));
			return  &( halo.GetRecvFromDirBuf(2*Z_DIR + MG_FORWARD)[index]);
		}
	}
}

template<typename T, template <typename> class Accessor>
inline
const float*
GetNeighborZMinus(const HaloContainer<T>& halo, const T& in, int x_cb, int y, int z, int t, int source_cb)
{
	const IndexType dir = Z_DIR;
	const IndexType fb = MG_BACKWARD;

	if ( z > 0  ) {

		return Accessor<T>::get(in,source_cb, x_cb+ in.GetNxh()*(y + in.GetNy()*((z-1) + in.GetNz()*t)), dir, fb);
	}
	else {

		if(halo.LocalDir(Z_DIR) )  {
			return Accessor<T>::get(in,source_cb, x_cb + in.GetNxh()*(y + in.GetNy()*((in.GetNz()-1) + in.GetNz()*t)), dir, fb);
		}
		else {

			int index = halo.GetDataTypeSize()*(x_cb + in.GetNxh()*(y + in.GetNy()*t));
			return  &( halo.GetRecvFromDirBuf(2*Z_DIR + MG_BACKWARD)[index]);
		}
	}
}

template<typename T, template <typename> class Accessor>
inline
const float*
GetNeighborTPlus(const HaloContainer<T>& halo, const T& in, int x_cb, int y, int z, int t, int source_cb)
{
	const IndexType dir = T_DIR;
	const IndexType fb = MG_FORWARD;

	if ( t < in.GetNt() - 1 ) {

		return Accessor<T>::get(in,source_cb, x_cb+ in.GetNxh()*(y + in.GetNy()*(z + in.GetNz()*(t+1))), dir, fb);
	}
	else {

		if(halo.LocalDir(T_DIR) )  {
			return Accessor<T>::get(in,source_cb, x_cb + in.GetNxh()*(y + in.GetNy()*z), dir, fb);
		}
		else {

			int index = halo.GetDataTypeSize()*(x_cb + in.GetNxh()*(y + in.GetNy()*z));

			return  &( halo.GetRecvFromDirBuf(2*T_DIR + MG_FORWARD)[index]);
		}
	}
}


template<typename T, template <typename> class Accessor>
inline
const float*
GetNeighborTMinus(const HaloContainer<T>& halo, const T& in, int x_cb, int y, int z, int t, int source_cb)

{
  const IndexType dir = T_DIR; const IndexType fb = MG_BACKWARD;

  if ( t > 0  ) {
	  	  return Accessor<T>::get(in,source_cb, x_cb+ in.GetNxh()*(y + in.GetNy()*(z + in.GetNz()*(t-1))),dir,fb);
	}
	else {

		if(halo.LocalDir(T_DIR) )  {
			return Accessor<T>::get(in,source_cb, x_cb + in.GetNxh()*(y + in.GetNy()*(z + in.GetNz()*(in.GetNt()-1))),dir,fb);
		}
		else {
			int index = halo.GetDataTypeSize()*(x_cb + in.GetNxh()*(y + in.GetNy()*z));
			return  &( halo.GetRecvFromDirBuf(2*T_DIR + MG_BACKWARD)[index]);
		}
	}
}

template<typename T, template <typename> class Accessor>
inline const float*
GetNeighborDir(const HaloContainer<T>& halo, const T& in, int dir, int target_cb, int cbsite )
{
	int source_cb = 1-target_cb;
	int tmp_yzt = cbsite / in.GetNxh();
	int xcb = cbsite - in.GetNxh() * tmp_yzt;
	int tmp_zt = tmp_yzt / in.GetNy();
	int y = tmp_yzt - in.GetNy() * tmp_zt;
	int t = tmp_zt / in.GetNz();
	int z = tmp_zt - in.GetNz() * t;
	int x = 2*xcb + ((target_cb+y+z+t)&0x1);  // Global X


	switch(dir) {
	case 0:
		return GetNeighborXPlus<T,Accessor>(halo,in,x,y,z,t,source_cb);
		break;
	case 1:
		return GetNeighborXMinus<T,Accessor>(halo,in,x,y,z,t,source_cb);
		break;
	case 2:
		return GetNeighborYPlus<T,Accessor>(halo,in,xcb,y,z,t,source_cb);
		break;
	case 3:
		return GetNeighborYMinus<T,Accessor>(halo,in,xcb,y,z,t,source_cb);
		break;
	case 4:
		return GetNeighborZPlus<T,Accessor>(halo,in,xcb,y,z,t,source_cb);
		break;
	case 5:
		return GetNeighborZMinus<T,Accessor>(halo,in,xcb,y,z,t,source_cb);
		break;
	case 6:
		return GetNeighborTPlus<T,Accessor>(halo,in,xcb,y,z,t,source_cb);
		break;
	case 7:
		return GetNeighborTMinus<T,Accessor>(halo,in,xcb,y,z,t,source_cb);
		break;
	default:
		MasterLog(ERROR, "Dir %d > 7 in GetNeighborDir. This ought to never happen", dir);
		break;
	}
	// Wot no return...
	return nullptr; // never get here...
}


} // namespace


#endif /* INCLUDE_LATTICE_SPINOR_HALO_H_ */
