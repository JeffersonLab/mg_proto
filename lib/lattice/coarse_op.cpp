
#include <omp.h>
#include <cstdio>
#include <iostream>

#include "lattice/coarse/coarse_op.h"
#include "lattice/cmat_mult.h"
#include "utils/memory.h"
#include "utils/print_utils.h"
#include <complex>

#include <immintrin.h>

//#include "../../include/lattice/thread_info.h.bak"
#include "lattice/geometry_utils.h"
namespace MG {


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
	  _n_t( l_info.GetLatticeDimensions()[3] )
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

#pragma omp critical
		{
			std::printf("Thread=%d smtid=%d n_sites_cb=%d n_vrows=%d min_vrow=%d max_vrow=%d min_site=%d max_site=%d\n",
					tid,smt_id,n_sites_cb, _n_vrows, _thread_limits[tid].min_vrow, _thread_limits[tid].max_vrow, min_site, max_site);
		}
	} // omp parallel

}


void CoarseDiracOp::operator()(CoarseSpinor& spinor_out,
			const CoarseGauge& gauge_clov_in,
			const CoarseSpinor& spinor_in,
			const IndexType target_cb,
			const IndexType dagger,
			const IndexType tid) const
{
	IndexType min_site = _thread_limits[tid].min_site;
	IndexType max_site = _thread_limits[tid].max_site;

//	std::cout << "target_cb =" << target_cb << " min_site=" << min_site << " max_site=" << max_site << std::endl;

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



#if 1
		x_plus /= 2; // Convert to checkerboard
		x_minus /=2; // Covert to checkerboard
		const IndexType source_cb = 1 - target_cb;
		const float *neigh_spinors[8] = {
			spinor_in.GetSiteDataPtr(source_cb, x_plus  + _n_xh*(y + _n_y*(z + _n_z*t))),
			spinor_in.GetSiteDataPtr(source_cb, x_minus + _n_xh*(y + _n_y*(z + _n_z*t))),
			spinor_in.GetSiteDataPtr(source_cb, xcb + _n_xh*(y_plus + _n_y*(z + _n_z*t))),
			spinor_in.GetSiteDataPtr(source_cb, xcb + _n_xh*(y_minus + _n_y*(z + _n_z*t))),
			spinor_in.GetSiteDataPtr(source_cb, xcb + _n_xh*(y + _n_y*(z_plus + _n_z*t))),
			spinor_in.GetSiteDataPtr(source_cb, xcb + _n_xh*(y + _n_y*(z_minus + _n_z*t))),
			spinor_in.GetSiteDataPtr(source_cb, xcb + _n_xh*(y + _n_y*(z + _n_z*t_plus))),
			spinor_in.GetSiteDataPtr(source_cb, xcb + _n_xh*(y + _n_y*(z + _n_z*t_minus))),
		};
#else
		IndexArray latdims=spinor_in.GetInfo().GetLatticeDimensions();
		int xf_cb, xf_site; CoordsToCBIndex({{x_plus,y,z,t}}, latdims, xf_cb, xf_site);
		int xb_cb, xb_site; CoordsToCBIndex({{x_minus,y,z,t}}, latdims, xb_cb, xb_site);
		int yf_cb, yf_site; CoordsToCBIndex({{x,y_plus,z,t}}, latdims, yf_cb, yf_site);
		int yb_cb, yb_site; CoordsToCBIndex({{x,y_minus,z,t}}, latdims, yb_cb, yb_site);
		int zf_cb, zf_site; CoordsToCBIndex({{x,y, z_plus,t}}, latdims, zf_cb, zf_site);
		int zb_cb, zb_site; CoordsToCBIndex({{x,y, z_minus,t}}, latdims, zb_cb, zb_site);
		int tf_cb, tf_site; CoordsToCBIndex({{x,y, z,t_plus}}, latdims, tf_cb, tf_site);
		int tb_cb, tb_site; CoordsToCBIndex({{x,y, z,t_minus}}, latdims, tb_cb, tb_site);

		const float *neigh_spinors[8] = {
			spinor_in.GetSiteDataPtr(xf_cb, xf_site),
			spinor_in.GetSiteDataPtr(xb_cb, xb_site),
			spinor_in.GetSiteDataPtr(yf_cb, yf_site),
			spinor_in.GetSiteDataPtr(yb_cb, yb_site),
			spinor_in.GetSiteDataPtr(zf_cb, zf_site),
			spinor_in.GetSiteDataPtr(zb_cb, zb_site),
			spinor_in.GetSiteDataPtr(tf_cb, tf_site),
			spinor_in.GetSiteDataPtr(tb_cb, tb_site)
		};
#endif
		siteApplyDslash(output, gauge_links, spinor_cb, neigh_spinors, dagger);
	}

}


inline
void CoarseDiracOp::siteApplyDslash( float *output,
		  	  	  	  	 	 const float* gauge_links[9],
							 const float* spinor_cb,
							 const float* neigh_spinors[8],
							 const IndexType dagger) const
{
	const int N_color = GetNumColor();
	const int N_colorspin = GetNumColorSpin();


	for(int i=0; i < N_colorspin*n_complex; ++i) {
		output[i] = 0;
	}

#if 1
	if( dagger == LINOP_OP ) {

		// Central Piece:
		CMatMultNaive(output,gauge_links[8],spinor_cb, N_colorspin);

		// Leaf pieces
		for(int mu=0; mu < 8; ++mu) {
			CMatMultNaiveAdd(output, gauge_links[mu], neigh_spinors[mu], N_colorspin);
		}
	}
	else {

		// Central Piece:
		// Apply (1,1,1,..1,-1,-1,...-1);
		float in_spinor[N_colorspin*n_complex];
		for(int j=0; j < N_color*n_complex; ++j) {
			in_spinor[j] = spinor_cb[j];
		}
		for(int j=N_color*n_complex; j < N_colorspin*n_complex; ++j) {
			in_spinor[j] = -spinor_cb[j];
		}
		// Central Piece:
		CMatMultNaive(output,gauge_links[8],in_spinor, N_colorspin);

		// Apply the Dslash term.
		for(int mu=0; mu < 8; ++mu) {

			// Apply (1,1,1,..1,-1,-1,...-1);
			for(int j=0; j < N_color*n_complex; ++j) {
				in_spinor[j] = neigh_spinors[mu][j];
			}
			for(int j=N_color*n_complex; j < N_colorspin*n_complex; ++j) {
				in_spinor[j] = -neigh_spinors[mu][j];
			}

			CMatMultNaiveAdd(output, gauge_links[mu], in_spinor, N_colorspin);
		} // mu

		for(int j=N_color*n_complex; j < N_colorspin*n_complex; ++j) {
			output[j] = -output[j];
		}
	}
#endif

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
			IndexType x_plus = (x < _n_x-1 ) ? (x + 1) : 0;
			x_plus /= 2; // Convert to checkerboard
			neigh_spinor = spinor_in.GetSiteDataPtr(source_cb, x_plus  + _n_xh*(y + _n_y*(z + _n_z*t)));
		}
		break;
		case 1:
		{
			IndexType x_minus = ( x > 0 ) ?  (x - 1) : _n_x-1;
			x_minus /=2; // Covert to checkerboard
			neigh_spinor = spinor_in.GetSiteDataPtr(source_cb, x_minus + _n_xh*(y + _n_y*(z + _n_z*t)));
		}
			break;
		case 2:
		{
			IndexType y_plus = ( y < _n_y - 1) ? y+1 : 0;
			neigh_spinor = spinor_in.GetSiteDataPtr(source_cb, xcb + _n_xh*(y_plus + _n_y*(z + _n_z*t)));
		}
			break;
		case 3:
		{
			IndexType y_minus = ( y > 0 ) ? y-1 : _n_y - 1;
			neigh_spinor = spinor_in.GetSiteDataPtr(source_cb, xcb + _n_xh*(y_minus + _n_y*(z + _n_z*t)));
		}
			break;
		case 4:
		{
			IndexType z_plus = ( z < _n_z - 1) ? z+1 : 0;
			neigh_spinor = spinor_in.GetSiteDataPtr(source_cb, xcb + _n_xh*(y + _n_y*(z_plus + _n_z*t)));
		}

			break;
		case 5:
		{
			IndexType z_minus = ( z > 0 ) ? z-1 : _n_z - 1;
			neigh_spinor = spinor_in.GetSiteDataPtr(source_cb, xcb + _n_xh*(y + _n_y*(z_minus + _n_z*t)));
		}
			break;
		case 6:
		{
			IndexType t_plus = ( t < _n_t - 1) ? t+1 : 0;
			neigh_spinor = spinor_in.GetSiteDataPtr(source_cb, xcb + _n_xh*(y + _n_y*(z + _n_z*t_plus)));
		}

			break;
		case 7:
		{
			IndexType t_minus = ( t > 0 ) ? t-1 : _n_t - 1;
			neigh_spinor = spinor_in.GetSiteDataPtr(source_cb, xcb + _n_xh*(y + _n_y*(z + _n_z*t_minus)));
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



// Run in thread
void CoarseDiracOp::CloverApply(CoarseSpinor& spinor_out,
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
		float in_spinor[N_colorspin*n_complex];

		// Apply (1,1,1,..1,-1,-1,...-1); to spinor_cb
		for(int j=0; j < N_color*n_complex; ++j) {
			in_spinor[j] = input[j];
		}
		for(int j=N_color*n_complex; j < N_colorspin*n_complex; ++j) {
			in_spinor[j] = -input[j];
		}

		CMatMultNaive(output, clover, in_spinor, N_colorspin);

		// Apply (1,1,1,..1,-1,-1,...-1); to output
		for(int j=N_color*n_complex; j < N_colorspin*n_complex; ++j) {
			output[j] = -output[j];
		}

	}

}

} // Namespace

