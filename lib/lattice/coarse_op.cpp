
#include <omp.h>
#include <cstdio>
#include <iostream>

#include "lattice/coarse/coarse_op.h"
#include "lattice/cmat_mult.h"
#include "utils/memory.h"
#include "utils/print_utils.h"
#include <complex>

// #include <immintrin.h>

//#include "../../include/lattice/thread_info.h.bak"
#include "lattice/geometry_utils.h"
namespace MG {

template<int N_colorspin, typename InitOp>
void genericSiteOffDiagXPay(float *output,
		const float* gauge_links[9],
		float coeff,
		const float* neigh_spinors[8],
		const IndexType dagger,
		const InitOp& op);

void CoarseDiracOp::unprecOp(CoarseSpinor& spinor_out,
			const CoarseGauge& gauge_clov_in,
			const CoarseSpinor& spinor_in,
			const IndexType target_cb,
			const IndexType dagger,
			const IndexType tid) const
{

	IndexType min_site = _thread_limits[tid].min_site;
	IndexType max_site = _thread_limits[tid].max_site;

// 	Synchronous for now -- maybe change to comms compute overlap later
	// We are in an OMP region.

	if( _halo.NumNonLocalDirs() > 0 ) {


		for(int mu=0; mu < n_dim; ++mu) {
			// Pack face usese omp for internally
			if ( ! _halo.LocalDir(mu) ) {
				packFace(spinor_in,1-target_cb,mu,MG_BACKWARD);
				packFace(spinor_in,1-target_cb,mu,MG_FORWARD);
			}
		}

		// Make sure faces are packed
#pragma omp barrier
#pragma omp master
		{
			_halo.StartAllRecvs();
			_halo.StartAllSends();
			_halo.FinishAllSends();
			_halo.FinishAllRecvs();
		}
#pragma omp barrier
	}


	// Site is output site
	for(IndexType site=min_site; site < max_site;++site) {

		// Turn site into x,y,z,t coords assuming we run as
		//  site = x_cb + Nxh*( y + Ny*( z + Nz*t ) ) )

		IndexType tmp_yzt = site / _n_xh;
		IndexType xcb = site - _n_xh * tmp_yzt;
		IndexType tmp_zt = tmp_yzt / _n_y;
		IndexType y = tmp_yzt - _n_y * tmp_zt;
		IndexType t = tmp_zt / _n_z;
		IndexType z = tmp_zt - _n_z * t;



		float* output = spinor_out.GetSiteDataPtr(target_cb, site);
		const float* gauge_base = gauge_clov_in.GetSiteDataPtr(target_cb,site);
		const float* spinor_cb = spinor_in.GetSiteDataPtr(target_cb,site);
		const IndexType gdir_offset = gauge_clov_in.GetLinkOffset();

		const float *gauge_links[9]={ gauge_base,                    // X forward
							gauge_base+gdir_offset,        // X backward
							gauge_base+2*gdir_offset,      // Y forward
							gauge_base+3*gdir_offset,      // Y backward
							gauge_base+4*gdir_offset,      // Z forward
							gauge_base+5*gdir_offset,      // Z backward
							gauge_base+6*gdir_offset,      // T forward
							gauge_base+7*gdir_offset,       // T backward
							gauge_base+8*gdir_offset
		};

		// Neighbouring spinors
		IndexType x = 2*xcb + ((target_cb+y+z+t)&0x1);  // Global X

		// Boundaries -- we can indirect here to
		// some face buffers if needs be
		IndexType x_plus = (x < _n_x-1 ) ? (x + 1) : 0;
		IndexType x_minus = ( x > 0 ) ?  (x - 1) : _n_x-1;



		IndexType y_plus = ( y < _n_y - 1) ? y+1 : 0;
		IndexType y_minus = ( y > 0 ) ? y-1 : _n_y - 1;

		IndexType z_plus = ( z < _n_z - 1) ? z+1 : 0;
		IndexType z_minus = ( z > 0 ) ? z-1 : _n_z - 1;

		IndexType t_plus = ( t < _n_t - 1) ? t+1 : 0;
		IndexType t_minus = ( t > 0 ) ? t-1 : _n_t - 1;


		x_plus /= 2; // Convert to checkerboard
		x_minus /=2; // Covert to checkerboard
		const IndexType source_cb = 1 - target_cb;
		const float *neigh_spinors[8] = {
				GetNeighborXPlus(x,y,z,t,source_cb,spinor_in),
				GetNeighborXMinus(x,y,z,t,source_cb,spinor_in),
				GetNeighborYPlus(xcb,y,z,t,source_cb,spinor_in),
				GetNeighborYMinus(xcb,y,z,t,source_cb,spinor_in),
				GetNeighborZPlus(xcb,y,z,t, source_cb,spinor_in),
				GetNeighborZMinus(xcb,y,z,t,source_cb,spinor_in),
				GetNeighborTPlus(xcb,y,z,t,source_cb,spinor_in),
				GetNeighborTMinus(xcb,y,z,t,source_cb,spinor_in)
		};


		siteApplyClover(output,gauge_links[8],spinor_cb,dagger);
		siteApplyDslash_xpmy(output, 1.0, gauge_links, neigh_spinors, dagger);
	}

}


// Acts on the full spinor
void CoarseDiracOp::L_inv_matrix(CoarseSpinor& spinor_out,
			const CoarseGauge& gauge_clov_in,
			const CoarseSpinor& spinor_in,
			const IndexType dagger) const
{
	int target_cb = 1;

#pragma omp parallel
	{

		int tid=omp_get_thread_num();

		// Parity 0 - is easy it is just M^{-1}_ee
		M_diagInv(spinor_out,gauge_clov_in, spinor_in,0,LINOP_OP,tid);

#pragma omp barrier



	IndexType min_site = _thread_limits[tid].min_site;
	IndexType max_site = _thread_limits[tid].max_site;

// 	Synchronous for now -- maybe change to comms compute overlap later
	// We are in an OMP region.

	if( _halo.NumNonLocalDirs() > 0 ) {


		for(int mu=0; mu < n_dim; ++mu) {
			// Pack face usese omp for internally
			if ( ! _halo.LocalDir(mu) ) {
				packFace(spinor_in,1-target_cb,mu,MG_BACKWARD);
				packFace(spinor_in,1-target_cb,mu,MG_FORWARD);
			}
		}

		// Make sure faces are packed
#pragma omp barrier
#pragma omp master
		{
			_halo.StartAllRecvs();
			_halo.StartAllSends();
			_halo.FinishAllSends();
			_halo.FinishAllRecvs();
		}
#pragma omp barrier
	}


	// Site is output site
	for(IndexType site=min_site; site < max_site;++site) {

		// Turn site into x,y,z,t coords assuming we run as
		//  site = x_cb + Nxh*( y + Ny*( z + Nz*t ) ) )

		IndexType tmp_yzt = site / _n_xh;
		IndexType xcb = site - _n_xh * tmp_yzt;
		IndexType tmp_zt = tmp_yzt / _n_y;
		IndexType y = tmp_yzt - _n_y * tmp_zt;
		IndexType t = tmp_zt / _n_z;
		IndexType z = tmp_zt - _n_z * t;



		float* output = spinor_out.GetSiteDataPtr(target_cb, site);
		const float* gauge_base = gauge_clov_in.GetSiteADDataPtr(target_cb,site);
		const float* spinor_cb = spinor_in.GetSiteDataPtr(target_cb,site);
		const IndexType gdir_offset = gauge_clov_in.GetLinkOffset();

		const float *gauge_links[9]={ gauge_base,                    // X forward
							gauge_base+gdir_offset,        // X backward
							gauge_base+2*gdir_offset,      // Y forward
							gauge_base+3*gdir_offset,      // Y backward
							gauge_base+4*gdir_offset,      // Z forward
							gauge_base+5*gdir_offset,      // Z backward
							gauge_base+6*gdir_offset,      // T forward
							gauge_base+7*gdir_offset,       // T backward
							gauge_base+8*gdir_offset
		};

		// Neighbouring spinors
		IndexType x = 2*xcb + ((target_cb+y+z+t)&0x1);  // Global X

		// Boundaries -- we can indirect here to
		// some face buffers if needs be
		IndexType x_plus = (x < _n_x-1 ) ? (x + 1) : 0;
		IndexType x_minus = ( x > 0 ) ?  (x - 1) : _n_x-1;



		IndexType y_plus = ( y < _n_y - 1) ? y+1 : 0;
		IndexType y_minus = ( y > 0 ) ? y-1 : _n_y - 1;

		IndexType z_plus = ( z < _n_z - 1) ? z+1 : 0;
		IndexType z_minus = ( z > 0 ) ? z-1 : _n_z - 1;

		IndexType t_plus = ( t < _n_t - 1) ? t+1 : 0;
		IndexType t_minus = ( t > 0 ) ? t-1 : _n_t - 1;


		x_plus /= 2; // Convert to checkerboard
		x_minus /=2; // Covert to checkerboard
		const IndexType source_cb = 1 - target_cb;
		const float *neigh_spinors[8] = {
				GetNeighborXPlus(x,y,z,t,source_cb,spinor_out),
				GetNeighborXMinus(x,y,z,t,source_cb,spinor_out),
				GetNeighborYPlus(xcb,y,z,t,source_cb,spinor_out),
				GetNeighborYMinus(xcb,y,z,t,source_cb,spinor_out),
				GetNeighborZPlus(xcb,y,z,t, source_cb,spinor_out),
				GetNeighborZMinus(xcb,y,z,t,source_cb,spinor_out),
				GetNeighborTPlus(xcb,y,z,t,source_cb,spinor_out),
				GetNeighborTMinus(xcb,y,z,t,source_cb,spinor_out)
		};


		siteApplyClover(output,gauge_links[8],spinor_cb,dagger);
		siteApplyDslash_xpmy(output, -1.0, gauge_links, neigh_spinors, dagger);
	}
	} // omp parallel
}


void CoarseDiracOp::M_diag(CoarseSpinor& spinor_out,
			const CoarseGauge& gauge_clov_in,
			const CoarseSpinor& spinor_in,
			const IndexType target_cb,
			const IndexType dagger,
			const IndexType tid) const
{
	IndexType min_site = _thread_limits[tid].min_site;
	IndexType max_site = _thread_limits[tid].max_site;

	// Site is output site
	for(IndexType site=min_site; site < max_site;++site) {

		float* output = spinor_out.GetSiteDataPtr(target_cb, site);
		const float* clover = gauge_clov_in.GetSiteDirDataPtr(target_cb,site,8);
		const float* input = spinor_in.GetSiteDataPtr(target_cb,site);

		siteApplyClover(output, clover, input, dagger);
	}

}

void CoarseDiracOp::M_diagInv(CoarseSpinor& spinor_out,
			const CoarseGauge& gauge_clov_in,
			const CoarseSpinor& spinor_in,
			const IndexType target_cb,
			const IndexType dagger,
			const IndexType tid) const
{
	IndexType min_site = _thread_limits[tid].min_site;
	IndexType max_site = _thread_limits[tid].max_site;

	// Site is output site
	for(IndexType site=min_site; site < max_site;++site) {

		float* output = spinor_out.GetSiteDataPtr(target_cb, site);
		const float* clover = gauge_clov_in.GetSiteDirADDataPtr(target_cb,site,8);
		const float* input = spinor_in.GetSiteDataPtr(target_cb,site);

		siteApplyClover(output, clover, input, dagger);
	}

}


void CoarseDiracOp::M_offDiag_xpay(CoarseSpinor& spinor_out,
			const float alpha,
			const CoarseGauge& gauge_clov_in,
			const CoarseSpinor& spinor_in,
			const IndexType target_cb,
			const IndexType dagger,
			const IndexType tid) const
{
	const int N_colorspin = spinor_in.GetNumColorSpin();
	IndexType min_site = _thread_limits[tid].min_site;
	IndexType max_site = _thread_limits[tid].max_site;

// 	Synchronous for now -- maybe change to comms compute overlap later
	// We are in an OMP region.

	if( _halo.NumNonLocalDirs() > 0 ) {


		for(int mu=0; mu < n_dim; ++mu) {
			// Pack face usese omp for internally
			if ( ! _halo.LocalDir(mu) ) {
				packFace(spinor_in,1-target_cb,mu,MG_BACKWARD);
				packFace(spinor_in,1-target_cb,mu,MG_FORWARD);
			}
		}

		// Make sure faces are packed
#pragma omp barrier
#pragma omp master
		{
			_halo.StartAllRecvs();
			_halo.StartAllSends();
			_halo.FinishAllSends();
			_halo.FinishAllRecvs();
		}
#pragma omp barrier
	}


	// Site is output site
	for(IndexType site=min_site; site < max_site;++site) {

		// Turn site into x,y,z,t coords assuming we run as
		//  site = x_cb + Nxh*( y + Ny*( z + Nz*t ) ) )

		IndexType tmp_yzt = site / _n_xh;
		IndexType xcb = site - _n_xh * tmp_yzt;
		IndexType tmp_zt = tmp_yzt / _n_y;
		IndexType y = tmp_yzt - _n_y * tmp_zt;
		IndexType t = tmp_zt / _n_z;
		IndexType z = tmp_zt - _n_z * t;



		float* output = spinor_out.GetSiteDataPtr(target_cb, site);
		const float* gauge_base = gauge_clov_in.GetSiteDataPtr(target_cb,site);
		const float* spinor_cb = spinor_in.GetSiteDataPtr(target_cb,site);
		const IndexType gdir_offset = gauge_clov_in.GetLinkOffset();

		const float *gauge_links[9]={ gauge_base,                    // X forward
							gauge_base+gdir_offset,        // X backward
							gauge_base+2*gdir_offset,      // Y forward
							gauge_base+3*gdir_offset,      // Y backward
							gauge_base+4*gdir_offset,      // Z forward
							gauge_base+5*gdir_offset,      // Z backward
							gauge_base+6*gdir_offset,      // T forward
							gauge_base+7*gdir_offset,       // T backward
							gauge_base+8*gdir_offset
		};

		// Neighbouring spinors
		IndexType x = 2*xcb + ((target_cb+y+z+t)&0x1);  // Global X

		// Boundaries -- we can indirect here to
		// some face buffers if needs be
		IndexType x_plus = (x < _n_x-1 ) ? (x + 1) : 0;
		IndexType x_minus = ( x > 0 ) ?  (x - 1) : _n_x-1;



		IndexType y_plus = ( y < _n_y - 1) ? y+1 : 0;
		IndexType y_minus = ( y > 0 ) ? y-1 : _n_y - 1;

		IndexType z_plus = ( z < _n_z - 1) ? z+1 : 0;
		IndexType z_minus = ( z > 0 ) ? z-1 : _n_z - 1;

		IndexType t_plus = ( t < _n_t - 1) ? t+1 : 0;
		IndexType t_minus = ( t > 0 ) ? t-1 : _n_t - 1;


		x_plus /= 2; // Convert to checkerboard
		x_minus /=2; // Covert to checkerboard
		const IndexType source_cb = 1 - target_cb;
		const float *neigh_spinors[8] = {
				GetNeighborXPlus(x,y,z,t,source_cb,spinor_in),
				GetNeighborXMinus(x,y,z,t,source_cb,spinor_in),
				GetNeighborYPlus(xcb,y,z,t,source_cb,spinor_in),
				GetNeighborYMinus(xcb,y,z,t,source_cb,spinor_in),
				GetNeighborZPlus(xcb,y,z,t, source_cb,spinor_in),
				GetNeighborZMinus(xcb,y,z,t,source_cb,spinor_in),
				GetNeighborTPlus(xcb,y,z,t,source_cb,spinor_in),
				GetNeighborTMinus(xcb,y,z,t,source_cb,spinor_in)
		};



		siteApplyDslash_xpmy(output, 1.0, gauge_links, neigh_spinors, dagger);
	}

}

void CoarseDiracOp::M_invOffDiag_xpay(CoarseSpinor& spinor_out,
			const float alpha,
			const CoarseGauge& gauge_clov_in,
			const CoarseSpinor& spinor_in,
			const IndexType target_cb,
			const IndexType dagger,
			const IndexType tid) const
{
	const int N_colorspin = spinor_in.GetNumColorSpin();
	IndexType min_site = _thread_limits[tid].min_site;
	IndexType max_site = _thread_limits[tid].max_site;

// 	Synchronous for now -- maybe change to comms compute overlap later
	// We are in an OMP region.

	if( _halo.NumNonLocalDirs() > 0 ) {


		for(int mu=0; mu < n_dim; ++mu) {
			// Pack face usese omp for internally
			if ( ! _halo.LocalDir(mu) ) {
				packFace(spinor_in,1-target_cb,mu,MG_BACKWARD);
				packFace(spinor_in,1-target_cb,mu,MG_FORWARD);
			}
		}

		// Make sure faces are packed
#pragma omp barrier
#pragma omp master
		{
			_halo.StartAllRecvs();
			_halo.StartAllSends();
			_halo.FinishAllSends();
			_halo.FinishAllRecvs();
		}
#pragma omp barrier
	}


	// Site is output site
	for(IndexType site=min_site; site < max_site;++site) {

		// Turn site into x,y,z,t coords assuming we run as
		//  site = x_cb + Nxh*( y + Ny*( z + Nz*t ) ) )

		IndexType tmp_yzt = site / _n_xh;
		IndexType xcb = site - _n_xh * tmp_yzt;
		IndexType tmp_zt = tmp_yzt / _n_y;
		IndexType y = tmp_yzt - _n_y * tmp_zt;
		IndexType t = tmp_zt / _n_z;
		IndexType z = tmp_zt - _n_z * t;



		float* output = spinor_out.GetSiteDataPtr(target_cb, site);
		const float* gauge_base = gauge_clov_in.GetSiteADDataPtr(target_cb,site);
		const float* spinor_cb = spinor_in.GetSiteDataPtr(target_cb,site);
		const IndexType gdir_offset = gauge_clov_in.GetLinkOffset();

		const float *gauge_links[9]={ gauge_base,                    // X forward
							gauge_base+gdir_offset,        // X backward
							gauge_base+2*gdir_offset,      // Y forward
							gauge_base+3*gdir_offset,      // Y backward
							gauge_base+4*gdir_offset,      // Z forward
							gauge_base+5*gdir_offset,      // Z backward
							gauge_base+6*gdir_offset,      // T forward
							gauge_base+7*gdir_offset,       // T backward
							gauge_base+8*gdir_offset
		};

		// Neighbouring spinors
		IndexType x = 2*xcb + ((target_cb+y+z+t)&0x1);  // Global X

		// Boundaries -- we can indirect here to
		// some face buffers if needs be
		IndexType x_plus = (x < _n_x-1 ) ? (x + 1) : 0;
		IndexType x_minus = ( x > 0 ) ?  (x - 1) : _n_x-1;



		IndexType y_plus = ( y < _n_y - 1) ? y+1 : 0;
		IndexType y_minus = ( y > 0 ) ? y-1 : _n_y - 1;

		IndexType z_plus = ( z < _n_z - 1) ? z+1 : 0;
		IndexType z_minus = ( z > 0 ) ? z-1 : _n_z - 1;

		IndexType t_plus = ( t < _n_t - 1) ? t+1 : 0;
		IndexType t_minus = ( t > 0 ) ? t-1 : _n_t - 1;


		x_plus /= 2; // Convert to checkerboard
		x_minus /=2; // Covert to checkerboard
		const IndexType source_cb = 1 - target_cb;
		const float *neigh_spinors[8] = {
				GetNeighborXPlus(x,y,z,t,source_cb,spinor_in),
				GetNeighborXMinus(x,y,z,t,source_cb,spinor_in),
				GetNeighborYPlus(xcb,y,z,t,source_cb,spinor_in),
				GetNeighborYMinus(xcb,y,z,t,source_cb,spinor_in),
				GetNeighborZPlus(xcb,y,z,t, source_cb,spinor_in),
				GetNeighborZMinus(xcb,y,z,t,source_cb,spinor_in),
				GetNeighborTPlus(xcb,y,z,t,source_cb,spinor_in),
				GetNeighborTMinus(xcb,y,z,t,source_cb,spinor_in)
		};



		siteApplyDslash_xpmy(output, alpha, gauge_links, neigh_spinors, dagger);
	}

}

void CoarseDiracOp::M_invOffDiag_xpayz(CoarseSpinor& spinor_out,
			const float alpha,
			const CoarseGauge& gauge_clov_in,
			const CoarseSpinor& spinor_cb,
			const CoarseSpinor& spinor_in,
			const IndexType target_cb,
			const IndexType dagger,
			const IndexType tid) const
{
	const int N_colorspin = spinor_cb.GetNumColorSpin();
	IndexType min_site = _thread_limits[tid].min_site;
	IndexType max_site = _thread_limits[tid].max_site;

// 	Synchronous for now -- maybe change to comms compute overlap later
	// We are in an OMP region.

	if( _halo.NumNonLocalDirs() > 0 ) {


		for(int mu=0; mu < n_dim; ++mu) {
			// Pack face usese omp for internally
			if ( ! _halo.LocalDir(mu) ) {
				packFace(spinor_in,1-target_cb,mu,MG_BACKWARD);
				packFace(spinor_in,1-target_cb,mu,MG_FORWARD);
			}
		}

		// Make sure faces are packed
#pragma omp barrier
#pragma omp master
		{
			_halo.StartAllRecvs();
			_halo.StartAllSends();
			_halo.FinishAllSends();
			_halo.FinishAllRecvs();
		}
#pragma omp barrier
	}


	// Site is output site
	for(IndexType site=min_site; site < max_site;++site) {

		// Turn site into x,y,z,t coords assuming we run as
		//  site = x_cb + Nxh*( y + Ny*( z + Nz*t ) ) )

		IndexType tmp_yzt = site / _n_xh;
		IndexType xcb = site - _n_xh * tmp_yzt;
		IndexType tmp_zt = tmp_yzt / _n_y;
		IndexType y = tmp_yzt - _n_y * tmp_zt;
		IndexType t = tmp_zt / _n_z;
		IndexType z = tmp_zt - _n_z * t;



		float* output = spinor_out.GetSiteDataPtr(target_cb, site);
		const float* gauge_base = gauge_clov_in.GetSiteADDataPtr(target_cb,site);
		const float* in_cb = spinor_cb.GetSiteDataPtr(target_cb,site);
		const IndexType gdir_offset = gauge_clov_in.GetLinkOffset();

		const float *gauge_links[9]={ gauge_base,                    // X forward
							gauge_base+gdir_offset,        // X backward
							gauge_base+2*gdir_offset,      // Y forward
							gauge_base+3*gdir_offset,      // Y backward
							gauge_base+4*gdir_offset,      // Z forward
							gauge_base+5*gdir_offset,      // Z backward
							gauge_base+6*gdir_offset,      // T forward
							gauge_base+7*gdir_offset,       // T backward
							gauge_base+8*gdir_offset
		};

		// Neighbouring spinors
		IndexType x = 2*xcb + ((target_cb+y+z+t)&0x1);  // Global X

		// Boundaries -- we can indirect here to
		// some face buffers if needs be
		IndexType x_plus = (x < _n_x-1 ) ? (x + 1) : 0;
		IndexType x_minus = ( x > 0 ) ?  (x - 1) : _n_x-1;



		IndexType y_plus = ( y < _n_y - 1) ? y+1 : 0;
		IndexType y_minus = ( y > 0 ) ? y-1 : _n_y - 1;

		IndexType z_plus = ( z < _n_z - 1) ? z+1 : 0;
		IndexType z_minus = ( z > 0 ) ? z-1 : _n_z - 1;

		IndexType t_plus = ( t < _n_t - 1) ? t+1 : 0;
		IndexType t_minus = ( t > 0 ) ? t-1 : _n_t - 1;


		x_plus /= 2; // Convert to checkerboard
		x_minus /=2; // Covert to checkerboard
		const IndexType source_cb = 1 - target_cb;
		const float *neigh_spinors[8] = {
				GetNeighborXPlus(x,y,z,t,source_cb,spinor_in),
				GetNeighborXMinus(x,y,z,t,source_cb,spinor_in),
				GetNeighborYPlus(xcb,y,z,t,source_cb,spinor_in),
				GetNeighborYMinus(xcb,y,z,t,source_cb,spinor_in),
				GetNeighborZPlus(xcb,y,z,t, source_cb,spinor_in),
				GetNeighborZMinus(xcb,y,z,t,source_cb,spinor_in),
				GetNeighborTPlus(xcb,y,z,t,source_cb,spinor_in),
				GetNeighborTMinus(xcb,y,z,t,source_cb,spinor_in)
		};



		siteApplyDslash_xpayz(output, alpha, gauge_links,in_cb,
											neigh_spinors, dagger);
	}

}


void CoarseDiracOp::M_invOffDiag(CoarseSpinor& spinor_out,
			const CoarseGauge& gauge_clov_in,
			const CoarseSpinor& spinor_in,
			const IndexType target_cb,
			const IndexType dagger,
			const IndexType tid) const
{
	const int N_colorspin = spinor_in.GetNumColorSpin();
	IndexType min_site = _thread_limits[tid].min_site;
	IndexType max_site = _thread_limits[tid].max_site;

// 	Synchronous for now -- maybe change to comms compute overlap later
	// We are in an OMP region.

	if( _halo.NumNonLocalDirs() > 0 ) {


		for(int mu=0; mu < n_dim; ++mu) {
			// Pack face usese omp for internally
			if ( ! _halo.LocalDir(mu) ) {
				packFace(spinor_in,1-target_cb,mu,MG_BACKWARD);
				packFace(spinor_in,1-target_cb,mu,MG_FORWARD);
			}
		}

		// Make sure faces are packed
#pragma omp barrier
#pragma omp master
		{
			_halo.StartAllRecvs();
			_halo.StartAllSends();
			_halo.FinishAllSends();
			_halo.FinishAllRecvs();
		}
#pragma omp barrier
	}


	// Site is output site
	for(IndexType site=min_site; site < max_site;++site) {

		// Turn site into x,y,z,t coords assuming we run as
		//  site = x_cb + Nxh*( y + Ny*( z + Nz*t ) ) )

		IndexType tmp_yzt = site / _n_xh;
		IndexType xcb = site - _n_xh * tmp_yzt;
		IndexType tmp_zt = tmp_yzt / _n_y;
		IndexType y = tmp_yzt - _n_y * tmp_zt;
		IndexType t = tmp_zt / _n_z;
		IndexType z = tmp_zt - _n_z * t;



		float* output = spinor_out.GetSiteDataPtr(target_cb, site);
		const float* gauge_base = gauge_clov_in.GetSiteADDataPtr(target_cb,site);
		const float* spinor_cb = spinor_in.GetSiteDataPtr(target_cb,site);
		const IndexType gdir_offset = gauge_clov_in.GetLinkOffset();

		const float *gauge_links[9]={ gauge_base,                    // X forward
							gauge_base+gdir_offset,        // X backward
							gauge_base+2*gdir_offset,      // Y forward
							gauge_base+3*gdir_offset,      // Y backward
							gauge_base+4*gdir_offset,      // Z forward
							gauge_base+5*gdir_offset,      // Z backward
							gauge_base+6*gdir_offset,      // T forward
							gauge_base+7*gdir_offset,       // T backward
							gauge_base+8*gdir_offset
		};

		// Neighbouring spinors
		IndexType x = 2*xcb + ((target_cb+y+z+t)&0x1);  // Global X

		// Boundaries -- we can indirect here to
		// some face buffers if needs be
		IndexType x_plus = (x < _n_x-1 ) ? (x + 1) : 0;
		IndexType x_minus = ( x > 0 ) ?  (x - 1) : _n_x-1;



		IndexType y_plus = ( y < _n_y - 1) ? y+1 : 0;
		IndexType y_minus = ( y > 0 ) ? y-1 : _n_y - 1;

		IndexType z_plus = ( z < _n_z - 1) ? z+1 : 0;
		IndexType z_minus = ( z > 0 ) ? z-1 : _n_z - 1;

		IndexType t_plus = ( t < _n_t - 1) ? t+1 : 0;
		IndexType t_minus = ( t > 0 ) ? t-1 : _n_t - 1;


		x_plus /= 2; // Convert to checkerboard
		x_minus /=2; // Covert to checkerboard
		const IndexType source_cb = 1 - target_cb;
		const float *neigh_spinors[8] = {
				GetNeighborXPlus(x,y,z,t,source_cb,spinor_in),
				GetNeighborXMinus(x,y,z,t,source_cb,spinor_in),
				GetNeighborYPlus(xcb,y,z,t,source_cb,spinor_in),
				GetNeighborYMinus(xcb,y,z,t,source_cb,spinor_in),
				GetNeighborZPlus(xcb,y,z,t, source_cb,spinor_in),
				GetNeighborZMinus(xcb,y,z,t,source_cb,spinor_in),
				GetNeighborTPlus(xcb,y,z,t,source_cb,spinor_in),
				GetNeighborTMinus(xcb,y,z,t,source_cb,spinor_in)
		};



		siteApplyDslash(output, gauge_links, neigh_spinors, dagger);
	}

}

class ZeroOutput {
public:
	inline
	void operator()(float *output, int i)  const {  output[i] = 0; }
};

class NopOutput {
public:
	inline
	void operator()(float *output, int i)  const { }
};

template<int N_colorspin, typename InitOp>
void genericSiteOffDiagXPay(float *output,
		const float* gauge_links[9],
		float coeff,
		const float* neigh_spinors[8],
		const IndexType dagger,
		const InitOp& op)
{
	constexpr int N_color = N_colorspin/2;

#pragma omp simd
	for(int i=0; i <  N_colorspin * n_complex; ++i) {
		op(output,i);
	}


	if( dagger == LINOP_OP ) {
		for(int mu=0; mu < 8; ++mu) {
			CMatMultNaiveCoeffAdd(output, coeff, gauge_links[mu], neigh_spinors[mu], N_colorspin);
		}
	}
	else {

		// A temporary so I can apply Gamma
		float in_spinor[N_colorspin*n_complex] __attribute__((aligned(64)));
		// A temporary so I can apply Gamma
		float tmpvec[N_colorspin*n_complex] __attribute__((aligned(64)));

		// Apply (1,1,1,..1,-1,-1,...-1);
		for(int j=0; j < N_color*n_complex; ++j) {
			in_spinor[j] = neigh_spinors[0][j];
		}
		for(int j=N_color*n_complex; j < N_colorspin*n_complex; ++j) {
			in_spinor[j] = -neigh_spinors[0][j];
		}
		CMatMultNaive(tmpvec, gauge_links[0], in_spinor, N_colorspin);

		// Apply the Dslash term.
		for(int mu=1; mu < 8; ++mu) {

			// Apply (1,1,1,..1,-1,-1,...-1);
			for(int j=0; j < N_color*n_complex; ++j) {
				in_spinor[j] = neigh_spinors[mu][j];
			}
			for(int j=N_color*n_complex; j < N_colorspin*n_complex; ++j) {
				in_spinor[j] = -neigh_spinors[mu][j];
			}

			CMatMultNaiveAdd(tmpvec, gauge_links[mu], in_spinor, N_colorspin);
		} // mu

		for(int j=0; j < N_color*n_complex; ++j) {
			output[j] += coeff * tmpvec[j];
		}

		for(int j=N_color*n_complex; j < N_colorspin*n_complex; ++j) {
			output[j] -= coeff * tmpvec[j];
		}

	}


}
template<int N_colorspin, typename InitOp>
void genericSiteOffDiagXPayz(float *output,
		const float* gauge_links[9],
		float coeff,
		const float* spinor_cb,
		const float* neigh_spinors[8],
		const IndexType dagger,
		const InitOp& op)
{
	constexpr int N_color = N_colorspin/2;


		// This is the same as for the dagger because we have G_5 I G_5 = G_5 G_5 I = I
		// D is the diagona
#pragma omp simd
		for(int i=0; i < 2*N_colorspin; ++i) {
			output[i] = spinor_cb[i];
		}


	if( dagger == LINOP_OP ) {
		// Dslash the offdiag
		for(int mu=0; mu < 8; ++mu) {
			CMatMultNaiveCoeffAdd(output, coeff, gauge_links[mu], neigh_spinors[mu], N_colorspin);
		}
	}
	else {
		// A temporary so I can apply Gamma
		float in_spinor[N_colorspin*n_complex] __attribute__((aligned(64)));
		// A temporary so I can apply Gamma
		float tmpvec[N_colorspin*n_complex] __attribute__((aligned(64)));

		// Apply (1,1,1,..1,-1,-1,...-1);
		for(int j=0; j < N_color*n_complex; ++j) {
			in_spinor[j] = neigh_spinors[0][j];
		}
		for(int j=N_color*n_complex; j < N_colorspin*n_complex; ++j) {
			in_spinor[j] = -neigh_spinors[0][j];
		}
		CMatMultNaive(tmpvec, gauge_links[0], in_spinor, N_colorspin);

		// Apply the Dslash term.
		for(int mu=1; mu < 8; ++mu) {

			// Apply (1,1,1,..1,-1,-1,...-1);
			for(int j=0; j < N_color*n_complex; ++j) {
				in_spinor[j] = neigh_spinors[mu][j];
			}
			for(int j=N_color*n_complex; j < N_colorspin*n_complex; ++j) {
				in_spinor[j] = -neigh_spinors[mu][j];
			}

			CMatMultNaiveAdd(tmpvec, gauge_links[mu], in_spinor, N_colorspin);
		} // mu

		for(int j=0; j < N_color*n_complex; ++j) {
			output[j] += coeff * tmpvec[j];
		}

		for(int j=N_color*n_complex; j < N_colorspin*n_complex; ++j) {
			output[j] -= coeff * tmpvec[j];
		}
	}
}

inline
void CoarseDiracOp::siteApplyDslash( float *output,
		  	  	  	  	 	 const float* gauge_links[9],
							 const float* neigh_spinors[8],
							 const IndexType dagger) const
{
	const int N_colorspin = GetNumColorSpin();
	const float coeff = 1;
	const ZeroOutput op;

	if (N_colorspin == 12 ) {
		genericSiteOffDiagXPay<12,ZeroOutput>(output, gauge_links, coeff, neigh_spinors, dagger, op);
	}
	else if( N_colorspin == 16 ) {
		genericSiteOffDiagXPay<16,ZeroOutput>(output, gauge_links, coeff, neigh_spinors, dagger, op);
	}
	else if( N_colorspin == 24 ) {
		genericSiteOffDiagXPay<24,ZeroOutput>(output, gauge_links, coeff, neigh_spinors, dagger, op);
	}
	else if ( N_colorspin == 32 ) {
		genericSiteOffDiagXPay<32,ZeroOutput>(output, gauge_links, coeff, neigh_spinors, dagger, op);
	}
	else if (N_colorspin == 48 ) {
		genericSiteOffDiagXPay<48,ZeroOutput>(output, gauge_links, coeff, neigh_spinors, dagger, op);
	}
	else if (N_colorspin == 64 ) {
		genericSiteOffDiagXPay<64,ZeroOutput>(output, gauge_links, coeff, neigh_spinors, dagger, op);
	}
	else if (N_colorspin == 96 ) {
		genericSiteOffDiagXPay<96,ZeroOutput>(output, gauge_links, coeff, neigh_spinors, dagger, op);
	}
	else if (N_colorspin == 128 ) {
		genericSiteOffDiagXPay<128,ZeroOutput>(output, gauge_links, coeff, neigh_spinors, dagger, op);
	}
	else {
		MasterLog(ERROR, "N_colorspin = %d not supported in siteApplyDslash" , N_colorspin );
	}
}




inline
void CoarseDiracOp::siteApplyDslash_xpmy( float *output,
							 const float coeff,
		  	  	  	  	 	 const float* gauge_links[9],
							 const float* neigh_spinors[8],
							 const IndexType dagger) const
{
	const int N_colorspin = GetNumColorSpin();
	const NopOutput op;

	if (N_colorspin == 12 ) {
		genericSiteOffDiagXPay<12,NopOutput>(output, gauge_links, coeff, neigh_spinors, dagger, op);
	}
	else if( N_colorspin == 16 ) {
		genericSiteOffDiagXPay<16,NopOutput>(output, gauge_links, coeff, neigh_spinors, dagger, op);
	}
	else if( N_colorspin == 24 ) {
		genericSiteOffDiagXPay<24,NopOutput>(output, gauge_links, coeff, neigh_spinors, dagger, op);
	}
	else if ( N_colorspin == 32 ) {
		genericSiteOffDiagXPay<32,NopOutput>(output, gauge_links, coeff, neigh_spinors, dagger, op);
	}
	else if (N_colorspin == 48 ) {
		genericSiteOffDiagXPay<48,NopOutput>(output, gauge_links, coeff, neigh_spinors, dagger, op);
	}
	else if (N_colorspin == 64 ) {
		genericSiteOffDiagXPay<64,NopOutput>(output, gauge_links, coeff, neigh_spinors, dagger, op);
	}
	else if (N_colorspin == 96 ) {
		genericSiteOffDiagXPay<96,NopOutput>(output, gauge_links, coeff, neigh_spinors, dagger, op);
	}
	else if (N_colorspin == 128 ) {
		genericSiteOffDiagXPay<128,NopOutput>(output, gauge_links, coeff, neigh_spinors, dagger, op);
	}
	else {
		MasterLog(ERROR, "N_colorspin = %d not supported in siteApplyDslash" , N_colorspin );
	}
}

inline
void CoarseDiracOp::siteApplyDslash_xpayz( float *output,
							 const float coeff,
		  	  	  	  	 	 const float* gauge_links[9],
							 const float* in_spinor_cb,
							 const float* neigh_spinors[8],
							 const IndexType dagger) const
{
	const int N_colorspin = GetNumColorSpin();
	const NopOutput op;

	if (N_colorspin == 12 ) {
		genericSiteOffDiagXPayz<12,NopOutput>(output, gauge_links, coeff, in_spinor_cb, neigh_spinors, dagger, op);
	}
	else if( N_colorspin == 16 ) {
		genericSiteOffDiagXPayz<16,NopOutput>(output, gauge_links, coeff, in_spinor_cb, neigh_spinors, dagger, op);
	}
	else if( N_colorspin == 24 ) {
		genericSiteOffDiagXPayz<24,NopOutput>(output, gauge_links, coeff, in_spinor_cb, neigh_spinors, dagger, op);
	}
	else if ( N_colorspin == 32 ) {
		genericSiteOffDiagXPayz<32,NopOutput>(output, gauge_links, coeff, in_spinor_cb, neigh_spinors, dagger, op);
	}
	else if (N_colorspin == 48 ) {
		genericSiteOffDiagXPayz<48,NopOutput>(output, gauge_links, coeff, in_spinor_cb, neigh_spinors, dagger, op);
	}
	else if (N_colorspin == 64 ) {
		genericSiteOffDiagXPayz<64,NopOutput>(output, gauge_links, coeff, in_spinor_cb, neigh_spinors, dagger, op);
	}
	else if (N_colorspin == 96 ) {
		genericSiteOffDiagXPayz<96,NopOutput>(output, gauge_links, coeff, in_spinor_cb, neigh_spinors, dagger, op);
	}
	else if (N_colorspin == 128 ) {
		genericSiteOffDiagXPayz<128,NopOutput>(output, gauge_links, coeff, in_spinor_cb, neigh_spinors, dagger, op);
	}
	else {
		MasterLog(ERROR, "N_colorspin = %d not supported in siteApplyDslash" , N_colorspin );
	}
}
// Lost site apply clover...
inline
void CoarseDiracOp::siteApplyClover( float* output,
					  const float* clover,
					  const float* input,
					  const IndexType dagger) const
{
	const int N_color = GetNumColor();
	const int N_colorspin = GetNumColorSpin();

	// NB: For = 6 input spinor may not be aligned!!!! BEWARE when testing optimized
	// CMatMult-s.
	if( dagger == LINOP_OP) {
		CMatMultNaive(output, clover, input, N_colorspin);
	}
	else {

		CMatAdjMultNaive(output, clover, input, N_colorspin);
	}

}


// Apply a single direction of Dslash -- used for coarsening
void CoarseDiracOp::DslashDir(CoarseSpinor& spinor_out,
			const CoarseGauge& gauge_in,
			const CoarseSpinor& spinor_in,
			const IndexType target_cb,
			const IndexType dir,
			const IndexType tid) const
{

	// This needs to be figured out.

	IndexType min_site = _thread_limits[tid].min_site;
	IndexType max_site = _thread_limits[tid].max_site;
	const int N_colorspin = GetNumColorSpin();

	int dir_4 = dir/2;
	int fb = (dir %2 == 0) ? MG_BACKWARD : MG_FORWARD;
	int bf = ( fb == MG_BACKWARD ) ? MG_FORWARD : MG_BACKWARD;
	if( ! _halo.LocalDir(dir_4) ) {
		// Prepost receive
#pragma omp master
		{
			// Start recv from e.g. back
			_halo.StartRecvFromDir(2*dir_4+bf);
		}
		// No need for barrier here
		// Pack face forward
		packFace(spinor_in,1-target_cb,dir_4,fb);

		/// Need barrier to make sure all threads finished packing
#pragma omp barrier

		// Master calls MPI stuff
#pragma omp master
		{
			// Start send to forwartd
			_halo.StartSendToDir(2*dir_4+fb);
			_halo.FinishSendToDir(2*dir_4+fb);
			_halo.FinishRecvFromDir(2*dir_4+bf);
		}
		// Threads oughtnt start until finish is complete
#pragma omp barrier
	}



	// Site is output site
	for(IndexType site=min_site; site < max_site;++site) {

		// Turn site into x,y,z,t coords assuming we run as
		//  site = x_cb + Nxh*( y + Ny*( z + Nz*t ) ) )

		IndexType tmp_yzt = site / _n_xh;
		IndexType xcb = site - _n_xh * tmp_yzt;
		IndexType tmp_zt = tmp_yzt / _n_y;
		IndexType y = tmp_yzt - _n_y * tmp_zt;
		IndexType t = tmp_zt / _n_z;
		IndexType z = tmp_zt - _n_z * t;

		// x coordinate: mult by 2 and add on offset to turn into
		// uncheckerboarded index
		IndexType x = 2*xcb + ((target_cb+y+z+t)&0x1);  // Global X

		float* output = spinor_out.GetSiteDataPtr(target_cb, site);
		const float* gauge_base = gauge_in.GetSiteDataPtr(target_cb,site);
		const IndexType gdir_offset = gauge_in.GetLinkOffset();

		const float* gauge_link_dir = gauge_base + dir*gdir_offset;

		/* The following case statement selects neighbors.
		 *  It is culled from the full Dslash
		 *  It of course would get complicated if some of the neighbors were in a halo
		 */

		const IndexType source_cb = 1 - target_cb;
		const float *neigh_spinor = nullptr;

		switch( dir ) {
		case 0:
		{
			neigh_spinor = GetNeighborXPlus(x,y,z,t,source_cb,spinor_in);
		}
		break;
		case 1:
		{
			neigh_spinor = GetNeighborXMinus(x,y,z,t,source_cb,spinor_in);
			break;
		case 2:
		{
			neigh_spinor = GetNeighborYPlus(xcb,y,z,t,source_cb,spinor_in);
		}
			break;
		case 3:
		{
			neigh_spinor = GetNeighborYMinus(xcb,y,z,t,source_cb,spinor_in);
		}
		}
			break;
		case 4:
		{
			neigh_spinor = GetNeighborZPlus(xcb,y,z,t,source_cb,spinor_in);
		}

			break;
		case 5:
		{
			neigh_spinor = GetNeighborZMinus(xcb,y,z,t,source_cb,spinor_in);
		}
			break;
		case 6:
		{
			neigh_spinor = GetNeighborTPlus(xcb,y,z,t,source_cb,spinor_in);
		}

			break;
		case 7:
		{
			neigh_spinor = GetNeighborTMinus(xcb,y,z,t,source_cb,spinor_in);
		}
			break;
		default:
			MasterLog(ERROR,"Invalid direction %d specified in DslashDir", dir);
			break;
		}

		// Multiply the link with the neighbor. EasyPeasy?
		CMatMultNaive(output, gauge_link_dir, neigh_spinor, N_colorspin);
	} // Loop over sites
}




CoarseDiracOp::CoarseDiracOp(const LatticeInfo& l_info, IndexType n_smt)
	: _lattice_info(l_info),
	  _n_color(l_info.GetNumColors()),
	  _n_spin(l_info.GetNumSpins()),
	  _n_colorspin(_n_color*_n_spin),
	  _n_smt(n_smt),
	  _n_vrows(2*_n_colorspin/VECLEN),
	  _n_xh( l_info.GetCBLatticeDimensions()[0] ),
	  _n_x( l_info.GetLatticeDimensions()[0] ),
	  _n_y( l_info.GetLatticeDimensions()[1] ),
	  _n_z( l_info.GetLatticeDimensions()[2] ),
	  _n_t( l_info.GetLatticeDimensions()[3] ),
	  _halo( l_info ),
	  _tmpvec( l_info )
{
#pragma omp parallel
	{
#pragma omp master
		{
			// Set the number of threads
			_n_threads = omp_get_num_threads();

			// ThreadLimits give iteration bounds for a specific thread with a tid.
			// These are things like min_site, max_site, min_row, max_row etc.
			// So here I allocate one for each thread.
			_thread_limits = (ThreadLimits*) MG::MemoryAllocate(_n_threads*sizeof(ThreadLimits), MG::REGULAR);
		} // omp master: barrier implied

#pragma omp barrier

		// Set the number of threads and break down into SIMD ID and Core ID
		// This requires knowledge about the order in which threads are assigned.
		//
		const int tid = omp_get_thread_num();
		const int n_cores = _n_threads/_n_smt;

		// Decompose tid into site_par_id (parallelism over sites)
		// and mv_par_id ( parallelism over rows of the matvec )
		// Order is: mv_par_id + _n_mv_parallel*site_par_id
        // Same as   smt_id + n_smt * core_id

		const int core_id = tid/_n_smt;
		const int smt_id = tid - _n_smt*core_id;
		const int n_floats_per_cacheline =MG_DEFAULT_CACHE_LINE_SIZE/sizeof(float);
		int n_cachelines = _n_vrows*VECLEN/n_floats_per_cacheline;
		int cl_per_smt = n_cachelines/_n_smt;
		if( n_cachelines % _n_smt != 0 ) cl_per_smt++;
		int min_cl = smt_id*cl_per_smt;
		int max_cl = MinInt((smt_id+1)*cl_per_smt, n_cachelines);
		int min_vrow = (min_cl*n_floats_per_cacheline)/VECLEN;
		int max_vrow = (max_cl*n_floats_per_cacheline)/VECLEN;

#if 1
		_thread_limits[tid].min_vrow = min_vrow;
		_thread_limits[tid].max_vrow = max_vrow;
#else
		// Hack so that we get 1 thread per core running even in SMT mode
		if ( smt_id == 0 ) {
			_thread_limits[tid].min_vrow = 0;
			_thread_limits[tid].max_vrow = (n_complex*_n_colorspin)/VECLEN;
		}
		else {
			// Non SMT threads will idle (loop limits too high)
			_thread_limits[tid].min_vrow =1+ (n_complex*_n_colorspin)/VECLEN;
			_thread_limits[tid].max_vrow =1+ (n_complex*_n_colorspin)/VECLEN;
		}
#endif

		// Find minimum and maximum site -- assume
		// small lattice so no blocking at this point
		// just linearly divide the sites
		const int n_sites_cb = _lattice_info.GetNumCBSites();
		int sites_per_core = n_sites_cb/n_cores;
		if( n_sites_cb % n_cores != 0 ) sites_per_core++;
		int min_site = core_id*sites_per_core;
		int max_site = MinInt((core_id+1)*sites_per_core, n_sites_cb);
		_thread_limits[tid].min_site = min_site;
		_thread_limits[tid].max_site = max_site;

//#pragma omp critical
	//	{
	//		std::printf("Thread=%d smtid=%d n_sites_cb=%d n_vrows=%d min_vrow=%d max_vrow=%d min_site=%d max_site=%d\n",
	//				tid,smt_id,n_sites_cb, _n_vrows, _thread_limits[tid].min_vrow, _thread_limits[tid].max_vrow, min_site, max_site);
	//	}
	} // omp parallel

}

inline
const float*
CoarseDiracOp::GetNeighborXPlus(int x, int y, int z, int t, int source_cb, const CoarseSpinor& spinor_in) const
{

	if ( x < _n_x - 1 ) {

		return spinor_in.GetSiteDataPtr(source_cb, ((x+1)/2)  + _n_xh*(y + _n_y*(z + _n_z*t)));
	}
	else {

		if(_halo.LocalDir(X_DIR) )  {
			return spinor_in.GetSiteDataPtr(source_cb, 0 + _n_xh*(y + _n_y*(z + _n_z*t)));
		}
		else {
			int n_colorspin  = spinor_in.GetNumColorSpin();
			int index = n_colorspin*n_complex*((y + _n_y*(z + _n_z*t))/2);
			return  &( _halo.GetRecvFromDirBuf(2*X_DIR + MG_FORWARD)[index]);
		}
	}
}

inline
const float*
CoarseDiracOp::GetNeighborXMinus(int x, int y, int z, int t, int source_cb, const CoarseSpinor& spinor_in) const
{
	if ( x > 0 ) {
		return spinor_in.GetSiteDataPtr(source_cb, ((x-1)/2) + _n_xh*(y + _n_y*(z + _n_z*t)));
	}
	else {
		if ( _halo.LocalDir(X_DIR) ) {
			return spinor_in.GetSiteDataPtr(source_cb, ((_n_x-1)/2) + _n_xh*(y + _n_y*(z + _n_z*t)));
		}
		else {
			// Get the buffer
			int n_colorspin = spinor_in.GetNumColorSpin();
			int index = n_complex*n_colorspin*((y + _n_y*(z + _n_z*t))/2);

			return  &(_halo.GetRecvFromDirBuf(2*X_DIR + MG_BACKWARD)[index]);
		}
	}
}

inline
const float*
CoarseDiracOp::GetNeighborYPlus(int x_cb, int y, int z, int t, int source_cb, const CoarseSpinor& spinor_in) const
{

	if ( y < _n_y - 1 ) {

		return spinor_in.GetSiteDataPtr(source_cb, x_cb+ _n_xh*((y+1) + _n_y*(z + _n_z*t)));
	}
	else {

		if(_halo.LocalDir(Y_DIR) )  {
			return spinor_in.GetSiteDataPtr(source_cb, x_cb + _n_xh*(0 + _n_y*(z + _n_z*t)));
		}
		else {
			int n_colorspin  = spinor_in.GetNumColorSpin();
			int index = n_colorspin*n_complex*(x_cb + _n_xh*(z + _n_z*t));
			return  &( _halo.GetRecvFromDirBuf(2*Y_DIR + MG_FORWARD)[index]);
		}
	}
}

inline
const float*
CoarseDiracOp::GetNeighborYMinus(int x_cb, int y, int z, int t, int source_cb, const CoarseSpinor& spinor_in) const
{

	if ( y > 0  ) {

		return spinor_in.GetSiteDataPtr(source_cb, x_cb+ _n_xh*((y-1) + _n_y*(z + _n_z*t)));
	}
	else {

		if(_halo.LocalDir(Y_DIR) )  {
			return spinor_in.GetSiteDataPtr(source_cb, x_cb + _n_xh*((_n_y-1) + _n_y*(z + _n_z*t)));
		}
		else {
			int n_colorspin  = spinor_in.GetNumColorSpin();
			int index = n_colorspin*n_complex*(x_cb + _n_xh*(z + _n_z*t));
			return  &( _halo.GetRecvFromDirBuf(2*Y_DIR + MG_BACKWARD)[index]);
		}
	}
}

inline
const float*
CoarseDiracOp::GetNeighborZPlus(int x_cb, int y, int z, int t, int source_cb, const CoarseSpinor& spinor_in) const
{

	if ( z < _n_z - 1 ) {

		return spinor_in.GetSiteDataPtr(source_cb, x_cb+ _n_xh*(y + _n_y*((z+1) + _n_z*t)));
	}
	else {

		if(_halo.LocalDir(Z_DIR) )  {
			return spinor_in.GetSiteDataPtr(source_cb, x_cb + _n_xh*(y + _n_y*(0 + _n_z*t)));
		}
		else {
			int n_colorspin  = spinor_in.GetNumColorSpin();
			int index = n_colorspin*n_complex*(x_cb + _n_xh*(y + _n_y*t));
			return  &( _halo.GetRecvFromDirBuf(2*Z_DIR + MG_FORWARD)[index]);
		}
	}
}

inline
const float*
CoarseDiracOp::GetNeighborZMinus(int x_cb, int y, int z, int t, int source_cb, const CoarseSpinor& spinor_in) const
{

	if ( z > 0  ) {

		return spinor_in.GetSiteDataPtr(source_cb, x_cb+ _n_xh*(y + _n_y*((z-1) + _n_z*t)));
	}
	else {

		if(_halo.LocalDir(Z_DIR) )  {
			return spinor_in.GetSiteDataPtr(source_cb, x_cb + _n_xh*(y + _n_y*((_n_z-1) + _n_z*t)));
		}
		else {
			int n_colorspin  = spinor_in.GetNumColorSpin();
			int index = n_colorspin*n_complex*(x_cb + _n_xh*(y + _n_y*t));
			return  &( _halo.GetRecvFromDirBuf(2*Z_DIR + MG_BACKWARD)[index]);
		}
	}
}
inline
const float*
CoarseDiracOp::GetNeighborTPlus(int x_cb, int y, int z, int t, int source_cb, const CoarseSpinor& spinor_in) const
{

	if ( t < _n_t - 1 ) {

		return spinor_in.GetSiteDataPtr(source_cb, x_cb+ _n_xh*(y + _n_y*(z + _n_z*(t+1))));
	}
	else {

		if(_halo.LocalDir(T_DIR) )  {
			return spinor_in.GetSiteDataPtr(source_cb, x_cb + _n_xh*(y + _n_y*z));
		}
		else {
			int n_colorspin  = spinor_in.GetNumColorSpin();
			int index = n_colorspin*n_complex*(x_cb + _n_xh*(y + _n_y*z));
			return  &( _halo.GetRecvFromDirBuf(2*T_DIR + MG_FORWARD)[index]);
		}
	}
}

inline
const float*
CoarseDiracOp::GetNeighborTMinus(int x_cb, int y, int z, int t, int source_cb, const CoarseSpinor& spinor_in) const
{

	if ( t > 0  ) {

		return spinor_in.GetSiteDataPtr(source_cb, x_cb+ _n_xh*(y + _n_y*(z + _n_z*(t-1))));
	}
	else {

		if(_halo.LocalDir(T_DIR) )  {
			return spinor_in.GetSiteDataPtr(source_cb, x_cb + _n_xh*(y + _n_y*(z + _n_z*(_n_t-1))));
		}
		else {
			int n_colorspin  = spinor_in.GetNumColorSpin();
			int index = n_colorspin*n_complex*(x_cb + _n_xh*(y + _n_y*z));
			return  &( _halo.GetRecvFromDirBuf(2*T_DIR + MG_BACKWARD)[index]);
		}
	}
}


void
CoarseDiracOp::packFace(const CoarseSpinor& spinor,
						IndexType cb, //CB to back
						IndexType mu,
						IndexType fb) const
{
	const IndexArray& latt_dims = _lattice_info.GetLatticeDimensions();
	const IndexArray& latt_cb_dims = _lattice_info.GetCBLatticeDimensions();
	IndexArray coords;



	// Grab the buffer from the Halo
	float* buffer = _halo.GetSendToDirBuf(2*mu + fb);

	int num_color_spins = _lattice_info.GetNumColorSpins();
	int buffer_site_offset = n_complex*num_color_spins;
	int buffer_sites = _halo.NumSitesInFace(mu);

	// Loop through the sites in the buffer
#pragma omp for
	for(int site =0; site < buffer_sites; ++site) {
		int local_cb = (cb  + _lattice_info.GetCBOrigin())&1;

		// I need to convert the face site index
		// into a body site index in the required checkerboard.
		coords[mu]= (fb == MG_BACKWARD ) ? 0 : latt_dims[mu]-1;
		if( mu == 0 ) {
			// X direction is special
			IndexArray x_cb_dims(latt_cb_dims); x_cb_dims[Y_DIR]/=2;

			IndexToCoords3(site,x_cb_dims,X_DIR,coords);
			coords[Y_DIR] *= 2;
			coords[Y_DIR] += ((local_cb + coords[X_DIR]+coords[Z_DIR] + coords[T_DIR])&1);
			coords[X_DIR] /=2; // Convert back to checkerboarded X_coord
		}
		else {
			// The Muth coordinate is eithe 0, or the last coordinate
			IndexToCoords3(site,latt_cb_dims,mu,coords);
		}
		int body_site = CoordsToIndex(coords,latt_cb_dims);
		float* buffersite = &buffer[site*buffer_site_offset];
		// Grab the body site
		const float* bodysite = spinor.GetSiteDataPtr(cb,body_site);

		// Copy body site into buffer site
		// This is likely to be done in a thread, so
		// use SIMD if you can.
#pragma omp simd
		for(int cspin_idx=0; cspin_idx < n_complex*num_color_spins; ++cspin_idx) {
			buffersite[cspin_idx] = bodysite[cspin_idx];
		} // Finish copying

	} // finish loop over sites.

}

} // Namespace

