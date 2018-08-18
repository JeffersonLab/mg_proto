
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
void genericSiteOffDiagXPayz(float *output,
		const float alpha,
		const float* gauge_links[8],
		const float* spinor_cb,
		const float* neigh_spinors[8])
{
	constexpr int N_color = N_colorspin/2;


		// This is the same as for the dagger because we have G_5 I G_5 = G_5 G_5 I = I
		// D is the diagona
#pragma omp simd
		for(int i=0; i < 2*N_colorspin; ++i) {
			output[i] = InitOp::op(spinor_cb,i);
		}



		// Dslash the offdiag
		for(int mu=0; mu < 8; ++mu) {
			CMatMultNaiveCoeffAdd(output, alpha, gauge_links[mu], neigh_spinors[mu], N_colorspin);
		}

}

template<int N_colorspin, typename InitOp>
void genericSiteGcOffDiagGcXPayz(float *output,
		const float alpha,
		const float* gauge_links[8],
		const float* spinor_cb,
		const float* neigh_spinors[8])
{
	constexpr int N_color = N_colorspin/2;


		// This is the same as for the dagger because we have G_5 I G_5 = G_5 G_5 I = I
		// D is the diagona
#pragma omp simd
		for(int i=0; i < 2*N_colorspin; ++i) {
			output[i] = InitOp::op(spinor_cb,i);
		}



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
			output[j] += alpha * tmpvec[j];
		}

		for(int j=N_color*n_complex; j < N_colorspin*n_complex; ++j) {
			output[j] -= alpha * tmpvec[j];
		}
}


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
	CommunicateHaloSyncInOMPParallel<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,target_cb);


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
		const float* gauge_base = gauge_clov_in.GetSiteDirDataPtr(target_cb,site,0);

		const float* spinor_cb = spinor_in.GetSiteDataPtr(target_cb,site);
		const IndexType gdir_offset = gauge_clov_in.GetLinkOffset();

		const float* clov = gauge_clov_in.GetSiteDiagDataPtr(target_cb,site);

		const float *gauge_links[8]={ gauge_base,                    // X forward
							gauge_base+gdir_offset,        // X backward
							gauge_base+2*gdir_offset,      // Y forward
							gauge_base+3*gdir_offset,      // Y backward
							gauge_base+4*gdir_offset,      // Z forward
							gauge_base+5*gdir_offset,      // Z backward
							gauge_base+6*gdir_offset,      // T forward
							gauge_base+7*gdir_offset };       // T backward

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
				GetNeighborXPlus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,x,y,z,t,source_cb),
				GetNeighborXMinus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,x,y,z,t,source_cb),
				GetNeighborYPlus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb),
				GetNeighborYMinus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb),
				GetNeighborZPlus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb),
				GetNeighborZMinus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb),
				GetNeighborTPlus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb),
				GetNeighborTMinus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb)
		};


		siteApplyClover(output,clov,spinor_cb,dagger);
		if( dagger == LINOP_OP) {
			siteApplyDslash_xpayz(output, 1.0, gauge_links,output, neigh_spinors);
		}
		else {
			siteApplyGcDslashGc_xpayz(output, 1.0, gauge_links,output, neigh_spinors);
		}
	}

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
		const float* clover = gauge_clov_in.GetSiteDiagDataPtr(target_cb,site);
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
		const float* clover = gauge_clov_in.GetSiteInvDiagDataPtr(target_cb,site);
		const float* input = spinor_in.GetSiteDataPtr(target_cb,site);

		siteApplyClover(output, clover, input, dagger);
	}

}


void CoarseDiracOp::M_D_xpay(CoarseSpinor& spinor_out,
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
	CommunicateHaloSyncInOMPParallel<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,target_cb);

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
		const float* gauge_base = gauge_clov_in.GetSiteDirDataPtr(target_cb,site,0);
		const float* spinor_cb = spinor_in.GetSiteDataPtr(target_cb,site);
		const IndexType gdir_offset = gauge_clov_in.GetLinkOffset();

		const float *gauge_links[8]={ gauge_base,                    // X forward
							gauge_base+gdir_offset,        // X backward
							gauge_base+2*gdir_offset,      // Y forward
							gauge_base+3*gdir_offset,      // Y backward
							gauge_base+4*gdir_offset,      // Z forward
							gauge_base+5*gdir_offset,      // Z backward
							gauge_base+6*gdir_offset,      // T forward
							gauge_base+7*gdir_offset };      // T backward


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
				GetNeighborXPlus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,x,y,z,t,source_cb),
				GetNeighborXMinus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,x,y,z,t,source_cb),
				GetNeighborYPlus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb),
				GetNeighborYMinus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb),
				GetNeighborZPlus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb),
				GetNeighborZMinus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb),
				GetNeighborTPlus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb),
				GetNeighborTMinus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb)
		};


		if( dagger == LINOP_OP ) {
			siteApplyDslash_xpayz(output, 1.0, gauge_links, output, neigh_spinors);
		}
		else {
			siteApplyGcDslashGc_xpayz(output, 1.0, gauge_links, output, neigh_spinors);
		}
	}

}

void CoarseDiracOp::M_AD_xpayz(CoarseSpinor& spinor_out,
			const float alpha,
			const CoarseGauge& gauge_in,
			const CoarseSpinor& spinor_in_cb,
			const CoarseSpinor& spinor_in_od,
			const IndexType target_cb,
			const IndexType dagger,
			const IndexType tid) const
{
	const int N_colorspin = spinor_in_cb.GetNumColorSpin();
	IndexType min_site = _thread_limits[tid].min_site;
	IndexType max_site = _thread_limits[tid].max_site;

	// 	Synchronous for now -- maybe change to comms compute overlap later
	CommunicateHaloSyncInOMPParallel<CoarseSpinor,CoarseAccessor>(_halo,spinor_in_od,target_cb);

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
		const float* gauge_base = ((dagger == LINOP_OP) ?
					gauge_in.GetSiteDirADDataPtr(target_cb,site,0)
					: gauge_in.GetSiteDirDADataPtr(target_cb,site,0)) ;

		const float* spinor_cb = spinor_in_cb.GetSiteDataPtr(target_cb,site);
		const IndexType gdir_offset = gauge_in.GetLinkOffset();

		const float *gauge_links[8]={ gauge_base,                    // X forward
							gauge_base+gdir_offset,        // X backward
							gauge_base+2*gdir_offset,      // Y forward
							gauge_base+3*gdir_offset,      // Y backward
							gauge_base+4*gdir_offset,      // Z forward
							gauge_base+5*gdir_offset,      // Z backward
							gauge_base+6*gdir_offset,      // T forward
							gauge_base+7*gdir_offset };      // T backward


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
				GetNeighborXPlus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in_od,x,y,z,t,source_cb),
				GetNeighborXMinus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in_od,x,y,z,t,source_cb),
				GetNeighborYPlus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in_od,xcb,y,z,t,source_cb),
				GetNeighborYMinus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in_od,xcb,y,z,t,source_cb),
				GetNeighborZPlus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in_od,xcb,y,z,t,source_cb),
				GetNeighborZMinus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in_od,xcb,y,z,t,source_cb),
				GetNeighborTPlus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in_od,xcb,y,z,t,source_cb),
				GetNeighborTMinus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in_od,xcb,y,z,t,source_cb)
		};


		if ( dagger == LINOP_OP ) {
			siteApplyDslash_xpayz(output, alpha, gauge_links, spinor_cb, neigh_spinors);
		}
		else {
			siteApplyGcDslashGc_xpayz(output, alpha, gauge_links, spinor_cb, neigh_spinors);
		}
	}

}

void CoarseDiracOp::M_DA_xpayz(CoarseSpinor& spinor_out,
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
	CommunicateHaloSyncInOMPParallel<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,target_cb);

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
		const float* gauge_base = (dagger == LINOP_OP ) ? gauge_clov_in.GetSiteDirDADataPtr(target_cb,site,0) :
				gauge_clov_in.GetSiteDirADDataPtr(target_cb,site,0);

		const float* in_cb = spinor_cb.GetSiteDataPtr(target_cb,site);
		const IndexType gdir_offset = gauge_clov_in.GetLinkOffset();

		const float *gauge_links[8]={ gauge_base,                    // X forward
							gauge_base+gdir_offset,        // X backward
							gauge_base+2*gdir_offset,      // Y forward
							gauge_base+3*gdir_offset,      // Y backward
							gauge_base+4*gdir_offset,      // Z forward
							gauge_base+5*gdir_offset,      // Z backward
							gauge_base+6*gdir_offset,      // T forward
							gauge_base+7*gdir_offset };       // T backward

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
				GetNeighborXPlus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,x,y,z,t,source_cb),
				GetNeighborXMinus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,x,y,z,t,source_cb),
				GetNeighborYPlus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb),
				GetNeighborYMinus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb),
				GetNeighborZPlus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb),
				GetNeighborZMinus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb),
				GetNeighborTPlus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb),
				GetNeighborTMinus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb)
		};


		if( dagger == LINOP_OP ) {
			siteApplyDslash_xpayz(output, alpha, gauge_links,in_cb,
				neigh_spinors);
		}
		else {
			siteApplyGcDslashGc_xpayz(output, alpha, gauge_links,in_cb,
							neigh_spinors);
		}
	}

}


void CoarseDiracOp::M_AD(CoarseSpinor& spinor_out,
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
	CommunicateHaloSyncInOMPParallel<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,target_cb);

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
		const float* gauge_base =(dagger == LINOP_OP)? gauge_clov_in.GetSiteDirADDataPtr(target_cb,site,0)
				: gauge_clov_in.GetSiteDirDADataPtr(target_cb,site,0);
		const float* spinor_cb = spinor_in.GetSiteDataPtr(target_cb,site);
		const IndexType gdir_offset = gauge_clov_in.GetLinkOffset();

		const float *gauge_links[8]={ gauge_base,                    // X forward
							gauge_base+gdir_offset,        // X backward
							gauge_base+2*gdir_offset,      // Y forward
							gauge_base+3*gdir_offset,      // Y backward
							gauge_base+4*gdir_offset,      // Z forward
							gauge_base+5*gdir_offset,      // Z backward
							gauge_base+6*gdir_offset,      // T forward
							gauge_base+7*gdir_offset };     // T backward

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
				GetNeighborXPlus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,x,y,z,t,source_cb),
				GetNeighborXMinus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,x,y,z,t,source_cb),
				GetNeighborYPlus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb),
				GetNeighborYMinus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb),
				GetNeighborZPlus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb),
				GetNeighborZMinus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb),
				GetNeighborTPlus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb),
				GetNeighborTMinus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb)
		};


		if( dagger == LINOP_OP ) {
			siteApplyDslash(output, gauge_links, neigh_spinors);
		}
		else {
			siteApplyGcDslashGc(output, gauge_links, neigh_spinors);
		}
	}

}


void CoarseDiracOp::M_DA(CoarseSpinor& spinor_out,
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
	CommunicateHaloSyncInOMPParallel<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,target_cb);

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
		const float* gauge_base = (dagger == LINOP_OP) ? gauge_clov_in.GetSiteDirDADataPtr(target_cb,site,0)
					: gauge_clov_in.GetSiteDirADDataPtr(target_cb,site,0);
		const float* spinor_cb = spinor_in.GetSiteDataPtr(target_cb,site);
		const IndexType gdir_offset = gauge_clov_in.GetLinkOffset();

		const float *gauge_links[8]={ gauge_base,                    // X forward
							gauge_base+gdir_offset,        // X backward
							gauge_base+2*gdir_offset,      // Y forward
							gauge_base+3*gdir_offset,      // Y backward
							gauge_base+4*gdir_offset,      // Z forward
							gauge_base+5*gdir_offset,      // Z backward
							gauge_base+6*gdir_offset,      // T forward
							gauge_base+7*gdir_offset };      // T backward


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
				GetNeighborXPlus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,x,y,z,t,source_cb),
				GetNeighborXMinus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,x,y,z,t,source_cb),
				GetNeighborYPlus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb),
				GetNeighborYMinus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb),
				GetNeighborZPlus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb),
				GetNeighborZMinus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb),
				GetNeighborTPlus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb),
				GetNeighborTMinus<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,xcb,y,z,t,source_cb)
		};


		if( dagger == LINOP_OP ) {
			siteApplyDslash(output, gauge_links, neigh_spinors);
		}
		else {
			siteApplyGcDslashGc(output, gauge_links, neigh_spinors);
		}
	}

}

class ZeroOutput {
public:
	inline
	static
	float op(const float *input, int i) {  return 0; }
};

class NopOutput {
public:
	inline
	static
	float op(const float *input, int i)  { return input[i]; }
};



inline
void CoarseDiracOp::siteApplyDslash( float *output,
		  	  	  	  	 	 const float* gauge_links[8],
							 const float* neigh_spinors[8]) const
{
	const int N_colorspin = GetNumColorSpin();
	const float coeff = 1;

	if (N_colorspin == 12 ) {
		genericSiteOffDiagXPayz<12,ZeroOutput>(output, coeff, gauge_links, output, neigh_spinors);
	}
	else if( N_colorspin == 16 ) {
		genericSiteOffDiagXPayz<16,ZeroOutput>(output, coeff, gauge_links, output, neigh_spinors);

	}
	else if( N_colorspin == 24 ) {
		genericSiteOffDiagXPayz<24,ZeroOutput>(output, coeff, gauge_links, output, neigh_spinors);

	}
	else if ( N_colorspin == 32 ) {
		genericSiteOffDiagXPayz<32,ZeroOutput>(output, coeff, gauge_links, output, neigh_spinors);

	}
	else if (N_colorspin == 48 ) {
		genericSiteOffDiagXPayz<48,ZeroOutput>(output, coeff, gauge_links, output, neigh_spinors);

	}
	else if (N_colorspin == 64 ) {
		genericSiteOffDiagXPayz<64,ZeroOutput>(output, coeff, gauge_links, output, neigh_spinors);

	}
	else if (N_colorspin == 96 ) {
		genericSiteOffDiagXPayz<96,ZeroOutput>(output, coeff, gauge_links, output, neigh_spinors);
	}
	else {
		MasterLog(ERROR, "N_colorspin = %d not supported in siteApplyDslash" , N_colorspin );
	}
}


inline
void CoarseDiracOp::siteApplyGcDslashGc( float *output,
		  	  	  	  	 	 const float* gauge_links[8],
							 const float* neigh_spinors[8]) const
{
	const int N_colorspin = GetNumColorSpin();
	const float coeff = 1;


	if (N_colorspin == 12 ) {
		genericSiteGcOffDiagGcXPayz<12,ZeroOutput>(output, coeff, gauge_links, output, neigh_spinors);
	}
	else if( N_colorspin == 16 ) {
		genericSiteGcOffDiagGcXPayz<16,ZeroOutput>(output, coeff, gauge_links, output, neigh_spinors);

	}
	else if( N_colorspin == 24 ) {
		genericSiteGcOffDiagGcXPayz<24,ZeroOutput>(output, coeff, gauge_links, output, neigh_spinors);

	}
	else if ( N_colorspin == 32 ) {
		genericSiteGcOffDiagGcXPayz<32,ZeroOutput>(output, coeff, gauge_links, output, neigh_spinors);

	}
	else if (N_colorspin == 48 ) {
		genericSiteGcOffDiagGcXPayz<48,ZeroOutput>(output, coeff, gauge_links, output, neigh_spinors);

	}
	else if (N_colorspin == 64 ) {
		genericSiteGcOffDiagGcXPayz<64,ZeroOutput>(output, coeff, gauge_links, output, neigh_spinors);

	}
	else if (N_colorspin == 96 ) {
		genericSiteGcOffDiagGcXPayz<96,ZeroOutput>(output, coeff, gauge_links, output, neigh_spinors);
	}
	else {
		MasterLog(ERROR, "N_colorspin = %d not supported in siteApplyDslash" , N_colorspin );
	}
}


inline
void CoarseDiracOp::siteApplyDslash_xpayz( float *output,
							 const float coeff,
		  	  	  	  	 	 const float* gauge_links[8],
							 const float* in_spinor_cb,
							 const float* neigh_spinors[8]) const
{
	const int N_colorspin = GetNumColorSpin();


	if (N_colorspin == 12 ) {
		genericSiteOffDiagXPayz<12,NopOutput>(output, coeff, gauge_links, in_spinor_cb, neigh_spinors);
	}
	else if( N_colorspin == 16 ) {
		genericSiteOffDiagXPayz<16,NopOutput>(output, coeff, gauge_links, in_spinor_cb, neigh_spinors);
	}
	else if( N_colorspin == 24 ) {
		genericSiteOffDiagXPayz<24,NopOutput>(output, coeff, gauge_links, in_spinor_cb, neigh_spinors);
	}
	else if ( N_colorspin == 32 ) {
		genericSiteOffDiagXPayz<32,NopOutput>(output, coeff, gauge_links, in_spinor_cb, neigh_spinors);
	}
	else if (N_colorspin == 48 ) {
		genericSiteOffDiagXPayz<48,NopOutput>(output, coeff, gauge_links, in_spinor_cb, neigh_spinors);
	}
	else if (N_colorspin == 64 ) {
		genericSiteOffDiagXPayz<64,NopOutput>(output, coeff, gauge_links, in_spinor_cb, neigh_spinors);
	}
	else if (N_colorspin == 96 ) {
		genericSiteOffDiagXPayz<96,NopOutput>(output, coeff, gauge_links, in_spinor_cb, neigh_spinors);
	}

	else {
		MasterLog(ERROR, "N_colorspin = %d not supported in siteApplyDslash" , N_colorspin );
	}
}


inline
void CoarseDiracOp::siteApplyGcDslashGc_xpayz( float *output,
							 const float coeff,
		  	  	  	  	 	 const float* gauge_links[8],
							 const float* in_spinor_cb,
							 const float* neigh_spinors[8]) const
{
	const int N_colorspin = GetNumColorSpin();


	if (N_colorspin == 12 ) {
		genericSiteGcOffDiagGcXPayz<12,NopOutput>(output, coeff, gauge_links, in_spinor_cb, neigh_spinors);
	}
	else if( N_colorspin == 16 ) {
		genericSiteGcOffDiagGcXPayz<16,NopOutput>(output, coeff, gauge_links, in_spinor_cb, neigh_spinors);
	}
	else if( N_colorspin == 24 ) {
		genericSiteGcOffDiagGcXPayz<24,NopOutput>(output, coeff, gauge_links, in_spinor_cb, neigh_spinors);
	}
	else if ( N_colorspin == 32 ) {
		genericSiteGcOffDiagGcXPayz<32,NopOutput>(output, coeff, gauge_links, in_spinor_cb, neigh_spinors);
	}
	else if (N_colorspin == 48 ) {
		genericSiteGcOffDiagGcXPayz<48,NopOutput>(output, coeff, gauge_links, in_spinor_cb, neigh_spinors);
	}
	else if (N_colorspin == 64 ) {
		genericSiteGcOffDiagGcXPayz<64,NopOutput>(output, coeff, gauge_links, in_spinor_cb, neigh_spinors);
	}
	else if (N_colorspin == 96 ) {
		genericSiteGcOffDiagGcXPayz<96,NopOutput>(output, coeff, gauge_links, in_spinor_cb, neigh_spinors);
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
		packFace<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,1-target_cb,dir_4,fb);

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


		float* output = spinor_out.GetSiteDataPtr(target_cb, site);
		const float* gauge_link_dir = gauge_in.GetSiteDirDataPtr(target_cb,site,dir);

		/* The following case statement selects neighbors.
		 *  It is culled from the full Dslash
		 *  It of course would get complicated if some of the neighbors were in a halo
		 */

		const float *neigh_spinor = GetNeighborDir<CoarseSpinor,CoarseAccessor>(_halo, spinor_in, dir, target_cb, site);

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


	} // omp parallel

}




} // Namespace

