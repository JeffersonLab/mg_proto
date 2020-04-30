
#include <omp.h>
#include <cstdio>
#include <iostream>

#include "lattice/coarse/coarse_op.h"
#include "lattice/cmat_mult.h"
#include "utils/memory.h"
#include "utils/print_utils.h"
#include "MG_config.h"
#include <complex>

// #include <immintrin.h>

//#include "../../include/lattice/thread_info.h.bak"
#include "lattice/geometry_utils.h"
namespace MG {

namespace {
	enum InitOp { zero, add };
	
	void genericSiteOffDiagXPayz(int N_colorspin,
			InitOp initop,
			float *output,
			const float alpha,
			const float* gauge_links[8],
			IndexType dagger, 
			const float* spinor_cb,
			const float* neigh_spinors[8],
			IndexType ncol=1)
	{
		int N_color = N_colorspin/2;


		// This is the same as for the dagger because we have G_5 I G_5 = G_5 G_5 I = I
		// D is the diagonal
		if (initop == add) {
			for(int i=0; i < 2*N_colorspin*ncol; ++i) {
				output[i] = spinor_cb[i];
			}
		}

		// Dslash the offdiag
		for(int mu=0; mu < 8; ++mu) {
			if (dagger == LINOP_OP) {
				CMatMultCoeffAddNaive(initop == zero && mu == 0 ? 0.0 : 1.0, output, alpha, gauge_links[mu], neigh_spinors[mu], N_colorspin, ncol);
			} else {
				GcCMatMultGcCoeffAddNaive(initop == zero && mu == 0 ? 0.0 : 1.0, output, alpha, gauge_links[mu], neigh_spinors[mu], N_colorspin, ncol);
			}
		}
	}

	// Lost site apply clover...
	void siteApplyClover(int N_colorspin,
			float* output,
			const float* clover,
			const float* input,
			const IndexType dagger,
			IndexType ncol)
	{
		// CMatMult-s.
		if( dagger == LINOP_OP) {
			CMatMultNaive(output, clover, input, N_colorspin, ncol);
		}
		else {
			// Slow: CMatAdjMultNaive(output, clover, input, N_colorspin);
	
			// Use Cc Hermiticity for faster operation
			GcCMatMultGcNaive(output,clover,input, N_colorspin, ncol);
		}
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

	IndexType ncol = spinor_in.GetNCol();

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



		float* output = spinor_out.GetSiteDataPtr(0, target_cb, site);
		const float* gauge_base = gauge_clov_in.GetSiteDirDataPtr(target_cb,site,0);

		const float* spinor_cb = spinor_in.GetSiteDataPtr(0, target_cb,site);
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


		siteApplyClover(GetNumColorSpin(), output,clov,spinor_cb,dagger, ncol);
		genericSiteOffDiagXPayz(GetNumColorSpin(), InitOp::add, output, 1.0, gauge_links, dagger, output, neigh_spinors, ncol);
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

	IndexType ncol = spinor_in.GetNCol();

	// Site is output site
	for(IndexType site=min_site; site < max_site;++site) {

		float* output = spinor_out.GetSiteDataPtr(0, target_cb, site);
		const float* clover = gauge_clov_in.GetSiteDiagDataPtr(target_cb,site);
		const float* input = spinor_in.GetSiteDataPtr(0, target_cb,site);

		siteApplyClover(GetNumColorSpin(), output, clover, input, dagger, ncol);
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

	IndexType ncol = spinor_in.GetNCol();

	// Site is output site
	for(IndexType site=min_site; site < max_site;++site) {

		float* output = spinor_out.GetSiteDataPtr(0, target_cb, site);
		const float* clover = gauge_clov_in.GetSiteInvDiagDataPtr(target_cb,site);
		const float* input = spinor_in.GetSiteDataPtr(0, target_cb,site);

		siteApplyClover(GetNumColorSpin(), output, clover, input, dagger, ncol);
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

	IndexType ncol = spinor_in.GetNCol();

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



		float* output = spinor_out.GetSiteDataPtr(0, target_cb, site);
		const float* gauge_base = gauge_clov_in.GetSiteDirDataPtr(target_cb,site,0);
		const float* spinor_cb = spinor_in.GetSiteDataPtr(0, target_cb,site);
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


		genericSiteOffDiagXPayz(GetNumColorSpin(), InitOp::add, output, 1.0, gauge_links, dagger, output, neigh_spinors, ncol);
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

	IndexType ncol = spinor_in_cb.GetNCol();

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



		float* output = spinor_out.GetSiteDataPtr(0, target_cb, site);
		const float* gauge_base = ((dagger == LINOP_OP) ?
					gauge_in.GetSiteDirADDataPtr(target_cb,site,0)
					: gauge_in.GetSiteDirDADataPtr(target_cb,site,0)) ;

		const float* spinor_cb = spinor_in_cb.GetSiteDataPtr(0, target_cb,site);
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


		genericSiteOffDiagXPayz(GetNumColorSpin(), InitOp::add, output, alpha, gauge_links, dagger, spinor_cb, neigh_spinors, ncol);
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

	IndexType ncol = spinor_in.GetNCol();

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



		float* output = spinor_out.GetSiteDataPtr(0, target_cb, site);
		const float* gauge_base = (dagger == LINOP_OP ) ? gauge_clov_in.GetSiteDirDADataPtr(target_cb,site,0) :
				gauge_clov_in.GetSiteDirADDataPtr(target_cb,site,0);

		const float* in_cb = spinor_cb.GetSiteDataPtr(0, target_cb,site);
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


		genericSiteOffDiagXPayz(GetNumColorSpin(), InitOp::add, output, alpha, gauge_links, dagger, in_cb, neigh_spinors, ncol);
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

	IndexType ncol = spinor_in.GetNCol();

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



		float* output = spinor_out.GetSiteDataPtr(0, target_cb, site);
		const float* gauge_base =(dagger == LINOP_OP)? gauge_clov_in.GetSiteDirADDataPtr(target_cb,site,0)
				: gauge_clov_in.GetSiteDirDADataPtr(target_cb,site,0);
		const float* spinor_cb = spinor_in.GetSiteDataPtr(0, target_cb,site);
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


		genericSiteOffDiagXPayz(GetNumColorSpin(), InitOp::zero, output, 1.0, gauge_links, dagger, output, neigh_spinors, ncol);
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

	IndexType ncol = spinor_in.GetNCol();

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



		float* output = spinor_out.GetSiteDataPtr(0, target_cb, site);
		const float* gauge_base = (dagger == LINOP_OP) ? gauge_clov_in.GetSiteDirDADataPtr(target_cb,site,0)
					: gauge_clov_in.GetSiteDirADDataPtr(target_cb,site,0);
		const float* spinor_cb = spinor_in.GetSiteDataPtr(0, target_cb,site);
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


		genericSiteOffDiagXPayz(GetNumColorSpin(), InitOp::zero, output, 1.0, gauge_links, dagger, output, neigh_spinors, ncol);
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


		float* output = spinor_out.GetSiteDataPtr(0, target_cb, site);
		const float* gauge_link_dir = gauge_in.GetSiteDirDataPtr(target_cb,site,dir);

		/* The following case statement selects neighbors.
		 *  It is culled from the full Dslash
		 *  It of course would get complicated if some of the neighbors were in a halo
		 */

		const float *neigh_spinor = GetNeighborDir<CoarseSpinor,CoarseAccessor>(_halo, spinor_in, dir, target_cb, site);

		// Multiply the link with the neighbor. EasyPeasy?
		CMatMultNaive(output, gauge_link_dir, neigh_spinor, N_colorspin, spinor_in.GetNCol());
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
	  _halo( l_info )
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

#ifdef MG_WRITE_COARSE
#include <mpi.h>

void CoarseDiracOp::write(const CoarseGauge& gauge, std::string& filename)
{
	IndexType n_colorspin = gauge.GetNumColorSpin();
	IndexArray lattice_dims;
	gauge.GetInfo().LocalDimsToGlobalDims(lattice_dims, gauge.GetInfo().GetLatticeDimensions());
	IndexType nxh = gauge.GetNxh();
	IndexType nx = gauge.GetNx();
	IndexType ny = gauge.GetNy();
	IndexType nz = gauge.GetNz();
	IndexType nt = gauge.GetNt();
	unsigned long num_sites = gauge.GetInfo().GetNumSites(), offset;
	MPI_Scan(&num_sites, &offset, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
	offset -= num_sites;
	MPI_File fh;
	MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
	if (offset == 0) {
		float header[6] = {4, (float)lattice_dims[0], (float)lattice_dims[1], (float)lattice_dims[2], (float)lattice_dims[3], (float)n_colorspin};
		MPI_Status status;
		MPI_File_write(fh, header, n_dim+2, MPI_FLOAT, &status);
	}
	MPI_File_set_view(fh, sizeof(float)*(n_dim+2 + (n_complex*n_colorspin*n_colorspin + n_dim*2)*(2*n_dim+1)*offset), MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);

	// Site is output site
	const int n_sites_cb = gauge.GetInfo().GetNumCBSites();
	for(IndexType site_cb=0; site_cb < n_sites_cb;++site_cb) {
		for(int target_cb=0; target_cb < 2;++target_cb) {

			const float* gauge_base = gauge.GetSiteDirDataPtr(target_cb,site_cb,0);
			const IndexType gdir_offset = gauge.GetLinkOffset();
			const float* clov = gauge.GetSiteDiagDataPtr(target_cb,site_cb);

			const float *gauge_links[9]={
				clov,                          // Diag
				gauge_base,                    // X forward
				gauge_base+gdir_offset,        // X backward
				gauge_base+2*gdir_offset,      // Y forward
				gauge_base+3*gdir_offset,      // Y backward
				gauge_base+4*gdir_offset,      // Z forward
				gauge_base+5*gdir_offset,      // Z backward
				gauge_base+6*gdir_offset,      // T forward
				gauge_base+7*gdir_offset };    // T backward

			// Turn site into x,y,z,t coords assuming we run as
			//  site = x_cb + Nxh*( y + Ny*( z + Nz*t ) ) )

			IndexType tmp_yzt = site_cb / nxh;
			IndexType xcb = site_cb - nxh * tmp_yzt;
			IndexType tmp_zt = tmp_yzt / ny;
			IndexType y = tmp_yzt - ny * tmp_zt;
			IndexType t = tmp_zt / nz;
			IndexType z = tmp_zt - nz * t;

			// Neighbouring spinors
			IndexType x = 2*xcb + ((target_cb+y+z+t)&0x1);  // Global X

			// Compute global coordinate
			const IndexArray local_site_coor = {x, y, z, t};
			IndexArray global_site_coor;
			gauge.GetInfo().LocalCoordToGlobalCoord(global_site_coor, local_site_coor);
			
			IndexArray coors[9];
			coors[0] = global_site_coor;
			for(int i=0,j=1; i<4; i++) {
				// Forward
				global_site_coor[i] = (global_site_coor[i] + 1) % lattice_dims[i];
				coors[j++] = global_site_coor;

				// Backward
				global_site_coor[i] = (global_site_coor[i] + lattice_dims[i] - 2) % lattice_dims[i];
				coors[j++] = global_site_coor;

				// Restore
				global_site_coor[i] = (global_site_coor[i] + 1) % lattice_dims[i];
			}

			for (int i=0; i<9; i++) {
				float coords[8] = {(float)coors[0][0], (float)coors[0][1], (float)coors[0][2], (float)coors[0][3], (float)coors[i][0], (float)coors[i][1], (float)coors[i][2], (float)coors[i][3]};
				MPI_Status status;
				MPI_File_write(fh, coords, 8, MPI_FLOAT, &status);
				MPI_File_write(fh, gauge_links[i], n_complex*n_colorspin*n_colorspin, MPI_FLOAT, &status);
			}
		}
	}

	MPI_File_close(&fh);
}
#else
void CoarseDiracOp::write(const CoarseGauge& gauge, std::string& filename)
{
}
#endif // MG_WRITE

} // Namespace

