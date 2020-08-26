
#include <cstdio>
#include <iostream>
#include <omp.h>

#include "MG_config.h"
#include "lattice/cmat_mult.h"
#include "lattice/coarse/coarse_op.h"
#include "utils/memory.h"
#include "utils/print_utils.h"
#include <complex>

// #include <immintrin.h>

//#include "../../include/lattice/thread_info.h.bak"
#include "lattice/geometry_utils.h"
namespace MG {

    namespace {
        enum InitOp { zero, add };

        typedef std::array<const float *, 8> Neigh_spinors;
        typedef std::array<const float *, 8> Gauge_links;

        Neigh_spinors get_neigh_spinors(const HaloContainer<CoarseSpinor> &halo,
                                        const CoarseSpinor &in, int target_cb, int cbsite) {
            return GetNeighborDirs<CoarseSpinor, CoarseAccessor>(halo, in, target_cb, cbsite);
        }

        Gauge_links get_gauge_links(const CoarseGauge &in, int target_cb, int cbsite) {
            const float *gauge_base = in.GetSiteDirDataPtr(target_cb, cbsite, 0);
            const IndexType gdir_offset = in.GetLinkOffset();
            return Gauge_links({{gauge_base,                      // X forward
                                 gauge_base + gdir_offset,        // X backward
                                 gauge_base + 2 * gdir_offset,    // Y forward
                                 gauge_base + 3 * gdir_offset,    // Y backward
                                 gauge_base + 4 * gdir_offset,    // Z forward
                                 gauge_base + 5 * gdir_offset,    // Z backward
                                 gauge_base + 6 * gdir_offset,    // T forward
                                 gauge_base + 7 * gdir_offset}}); // T backward
        }

        Gauge_links get_gauge_ad_links(const CoarseGauge &in, int target_cb, int cbsite,
                                       int dagger) {
            const float *gauge_base =
                ((dagger == LINOP_OP) ? in.GetSiteDirADDataPtr(target_cb, cbsite, 0)
                                      : in.GetSiteDirDADataPtr(target_cb, cbsite, 0));
            const IndexType gdir_offset = in.GetLinkOffset();
            return Gauge_links({{gauge_base,                      // X forward
                                 gauge_base + gdir_offset,        // X backward
                                 gauge_base + 2 * gdir_offset,    // Y forward
                                 gauge_base + 3 * gdir_offset,    // Y backward
                                 gauge_base + 4 * gdir_offset,    // Z forward
                                 gauge_base + 5 * gdir_offset,    // Z backward
                                 gauge_base + 6 * gdir_offset,    // T forward
                                 gauge_base + 7 * gdir_offset}}); // T backward
        }

        void genericSiteOffDiagXPayz(int N_colorspin, InitOp initop, float *output,
                                     const float alpha, const Gauge_links &gauge_links,
                                     IndexType dagger, const float *spinor_cb,
                                     const Neigh_spinors &neigh_spinors, IndexType ncol = 1) {
            // This is the same as for the dagger because we have G_5 I G_5 = G_5 G_5 I = I
            // D is the diagonal
            if (initop == add && output != spinor_cb) {
                for (int i = 0; i < 2 * N_colorspin * ncol; ++i) { output[i] = spinor_cb[i]; }
            }

            // Dslash the offdiag
            for (int mu = 0; mu < 8; ++mu) {
                if (dagger == LINOP_OP) {
                    CMatMultCoeffAddNaive(initop == zero && mu == 0 ? 0.0 : 1.0, output, alpha,
                                          gauge_links[mu], neigh_spinors[mu], N_colorspin, ncol);
                } else {
                    GcCMatMultGcCoeffAddNaive(initop == zero && mu == 0 ? 0.0 : 1.0, output, alpha,
                                              gauge_links[mu], neigh_spinors[mu], N_colorspin,
                                              ncol);
                }
            }
        }

        // Lost site apply clover...
        void siteApplyClover(int N_colorspin, float *output, const float *clover,
                             const float *input, const IndexType dagger, IndexType ncol) {
            // CMatMult-s.
            if (dagger == LINOP_OP) {
                CMatMultNaive(output, clover, input, N_colorspin, ncol);
            } else {
                // Slow: CMatAdjMultNaive(output, clover, input, N_colorspin);

                // Use Cc Hermiticity for faster operation
                GcCMatMultGcNaive(output, clover, input, N_colorspin, ncol);
            }
        }
    }

    void CoarseDiracOp::unprecOp(CoarseSpinor &spinor_out, const CoarseGauge &gauge_clov_in,
                                 const CoarseSpinor &spinor_in, const IndexType target_cb,
                                 const IndexType dagger, const IndexType tid) const {
        IndexType min_site = _thread_limits[tid].min_site;
        IndexType max_site = _thread_limits[tid].max_site;

        // 	Synchronous for now -- maybe change to comms compute overlap later
        // We are in an OMP region.
        CommunicateHaloSyncInOMPParallel<CoarseSpinor, CoarseAccessor>(_halo, spinor_in, target_cb);

        IndexType ncol = spinor_in.GetNCol();

        // Site is output site
        for (IndexType site = min_site; site < max_site; ++site) {

            float *output = spinor_out.GetSiteDataPtr(0, target_cb, site);

            const float *spinor_cb = spinor_in.GetSiteDataPtr(0, target_cb, site);
            const float *clov = gauge_clov_in.GetSiteDiagDataPtr(target_cb, site);
            siteApplyClover(GetNumColorSpin(), output, clov, spinor_cb, dagger, ncol);

            const Gauge_links gauge_links = get_gauge_links(gauge_clov_in, target_cb, site);
            const Neigh_spinors neigh_spinors =
                get_neigh_spinors(_halo, spinor_in, target_cb, site);
            genericSiteOffDiagXPayz(GetNumColorSpin(), InitOp::add, output, 1.0, gauge_links,
                                    dagger, output, neigh_spinors, ncol);
        }
    }

    void CoarseDiracOp::M_diag(CoarseSpinor &spinor_out, const CoarseGauge &gauge_clov_in,
                               const CoarseSpinor &spinor_in, const IndexType target_cb,
                               const IndexType dagger, const IndexType tid) const {
        IndexType min_site = _thread_limits[tid].min_site;
        IndexType max_site = _thread_limits[tid].max_site;

        IndexType ncol = spinor_in.GetNCol();

        // Site is output site
        for (IndexType site = min_site; site < max_site; ++site) {

            float *output = spinor_out.GetSiteDataPtr(0, target_cb, site);
            const float *clover = gauge_clov_in.GetSiteDiagDataPtr(target_cb, site);
            const float *input = spinor_in.GetSiteDataPtr(0, target_cb, site);

            siteApplyClover(GetNumColorSpin(), output, clover, input, dagger, ncol);
        }
    }

    void CoarseDiracOp::M_diagInv(CoarseSpinor &spinor_out, const CoarseGauge &gauge_clov_in,
                                  const CoarseSpinor &spinor_in, const IndexType target_cb,
                                  const IndexType dagger, const IndexType tid) const {
        IndexType min_site = _thread_limits[tid].min_site;
        IndexType max_site = _thread_limits[tid].max_site;

        IndexType ncol = spinor_in.GetNCol();

        // Site is output site
        for (IndexType site = min_site; site < max_site; ++site) {

            float *output = spinor_out.GetSiteDataPtr(0, target_cb, site);
            const float *clover = gauge_clov_in.GetSiteInvDiagDataPtr(target_cb, site);
            const float *input = spinor_in.GetSiteDataPtr(0, target_cb, site);

            siteApplyClover(GetNumColorSpin(), output, clover, input, dagger, ncol);
        }
    }

    void CoarseDiracOp::M_D_xpay(CoarseSpinor &spinor_out, const float alpha,
                                 const CoarseGauge &gauge_clov_in, const CoarseSpinor &spinor_in,
                                 const IndexType target_cb, const IndexType dagger,
                                 const IndexType tid) const {
        IndexType min_site = _thread_limits[tid].min_site;
        IndexType max_site = _thread_limits[tid].max_site;

        // 	Synchronous for now -- maybe change to comms compute overlap later
        CommunicateHaloSyncInOMPParallel<CoarseSpinor, CoarseAccessor>(_halo, spinor_in, target_cb);

        IndexType ncol = spinor_in.GetNCol();

        // Site is output site
        for (IndexType site = min_site; site < max_site; ++site) {

            float *output = spinor_out.GetSiteDataPtr(0, target_cb, site);

            const Gauge_links gauge_links = get_gauge_links(gauge_clov_in, target_cb, site);
            const Neigh_spinors neigh_spinors =
                get_neigh_spinors(_halo, spinor_in, target_cb, site);
            genericSiteOffDiagXPayz(GetNumColorSpin(), InitOp::add, output, alpha, gauge_links,
                                    dagger, output, neigh_spinors, ncol);
        }
    }

    void CoarseDiracOp::M_AD_xpayz(CoarseSpinor &spinor_out, const float alpha,
                                   const CoarseGauge &gauge_in, const CoarseSpinor &spinor_in_cb,
                                   const CoarseSpinor &spinor_in_od, const IndexType target_cb,
                                   const IndexType dagger, const IndexType tid) const {
        IndexType min_site = _thread_limits[tid].min_site;
        IndexType max_site = _thread_limits[tid].max_site;

        // 	Synchronous for now -- maybe change to comms compute overlap later
        CommunicateHaloSyncInOMPParallel<CoarseSpinor, CoarseAccessor>(_halo, spinor_in_od,
                                                                       target_cb);

        IndexType ncol = spinor_in_cb.GetNCol();

        // Site is output site
        for (IndexType site = min_site; site < max_site; ++site) {

            float *output = spinor_out.GetSiteDataPtr(0, target_cb, site);
            const float *spinor_cb = spinor_in_cb.GetSiteDataPtr(0, target_cb, site);
            const Gauge_links gauge_links = get_gauge_ad_links(gauge_in, target_cb, site, dagger);
            const Neigh_spinors neigh_spinors =
                get_neigh_spinors(_halo, spinor_in_od, target_cb, site);
            genericSiteOffDiagXPayz(GetNumColorSpin(), InitOp::add, output, alpha, gauge_links,
                                    dagger, spinor_cb, neigh_spinors, ncol);
        }
    }

    void CoarseDiracOp::M_D_xpay_Mz(CoarseSpinor &spinor_out, const float alpha,
                                    const CoarseGauge &gauge_in, const CoarseSpinor &spinor_in_cb,
                                    const CoarseSpinor &spinor_in_od, const IndexType target_cb,
                                    const IndexType dagger, const IndexType tid) const {
        IndexType min_site = _thread_limits[tid].min_site;
        IndexType max_site = _thread_limits[tid].max_site;

        // 	Synchronous for now -- maybe change to comms compute overlap later
        CommunicateHaloSyncInOMPParallel<CoarseSpinor, CoarseAccessor>(_halo, spinor_in_od,
                                                                       target_cb);

        IndexType ncol = spinor_in_cb.GetNCol();

        // Site is output site
        for (IndexType site = min_site; site < max_site; ++site) {

            float *output = spinor_out.GetSiteDataPtr(0, target_cb, site);
            const float *spinor_cb = spinor_in_cb.GetSiteDataPtr(0, target_cb, site);
            const float *clov = gauge_in.GetSiteDiagDataPtr(target_cb, site);
            siteApplyClover(GetNumColorSpin(), output, clov, spinor_cb, dagger, ncol);
            const Gauge_links gauge_links = get_gauge_links(gauge_in, target_cb, site);
            const Neigh_spinors neigh_spinors =
                get_neigh_spinors(_halo, spinor_in_od, target_cb, site);
            genericSiteOffDiagXPayz(GetNumColorSpin(), InitOp::add, output, alpha, gauge_links,
                                    dagger, output, neigh_spinors, ncol);
        }
    }

    void CoarseDiracOp::M_DA_xpayz(CoarseSpinor &spinor_out, const float alpha,
                                   const CoarseGauge &gauge_clov_in, const CoarseSpinor &spinor_cb,
                                   const CoarseSpinor &spinor_in, const IndexType target_cb,
                                   const IndexType dagger, const IndexType tid) const {
        IndexType min_site = _thread_limits[tid].min_site;
        IndexType max_site = _thread_limits[tid].max_site;

        // 	Synchronous for now -- maybe change to comms compute overlap later
        CommunicateHaloSyncInOMPParallel<CoarseSpinor, CoarseAccessor>(_halo, spinor_in, target_cb);

        IndexType ncol = spinor_in.GetNCol();

        // Site is output site
        for (IndexType site = min_site; site < max_site; ++site) {

            float *output = spinor_out.GetSiteDataPtr(0, target_cb, site);
            const Gauge_links gauge_links = get_gauge_ad_links(
                gauge_clov_in, target_cb, site, dagger == LINOP_OP ? LINOP_DAGGER : LINOP_OP);
            const Neigh_spinors neigh_spinors =
                get_neigh_spinors(_halo, spinor_in, target_cb, site);
            const float *in_cb = spinor_cb.GetSiteDataPtr(0, target_cb, site);
            genericSiteOffDiagXPayz(GetNumColorSpin(), InitOp::add, output, alpha, gauge_links,
                                    dagger, in_cb, neigh_spinors, ncol);
        }
    }

    void CoarseDiracOp::M_AD(CoarseSpinor &spinor_out, const CoarseGauge &gauge_clov_in,
                             const CoarseSpinor &spinor_in, const IndexType target_cb,
                             const IndexType dagger, const IndexType tid) const {
        IndexType min_site = _thread_limits[tid].min_site;
        IndexType max_site = _thread_limits[tid].max_site;

        // 	Synchronous for now -- maybe change to comms compute overlap later
        CommunicateHaloSyncInOMPParallel<CoarseSpinor, CoarseAccessor>(_halo, spinor_in, target_cb);

        IndexType ncol = spinor_in.GetNCol();

        // Site is output site
        for (IndexType site = min_site; site < max_site; ++site) {

            const Gauge_links gauge_links =
                get_gauge_ad_links(gauge_clov_in, target_cb, site, dagger);
            const Neigh_spinors neigh_spinors =
                get_neigh_spinors(_halo, spinor_in, target_cb, site);
            float *output = spinor_out.GetSiteDataPtr(0, target_cb, site);
            genericSiteOffDiagXPayz(GetNumColorSpin(), InitOp::zero, output, 1.0, gauge_links,
                                    dagger, output, neigh_spinors, ncol);
        }
    }

    void CoarseDiracOp::M_DA(CoarseSpinor &spinor_out, const CoarseGauge &gauge_clov_in,
                             const CoarseSpinor &spinor_in, const IndexType target_cb,
                             const IndexType dagger, const IndexType tid) const {
        IndexType min_site = _thread_limits[tid].min_site;
        IndexType max_site = _thread_limits[tid].max_site;

        // 	Synchronous for now -- maybe change to comms compute overlap later
        CommunicateHaloSyncInOMPParallel<CoarseSpinor, CoarseAccessor>(_halo, spinor_in, target_cb);

        IndexType ncol = spinor_in.GetNCol();

        // Site is output site
        for (IndexType site = min_site; site < max_site; ++site) {

            const Gauge_links gauge_links = get_gauge_ad_links(
                gauge_clov_in, target_cb, site, dagger == LINOP_OP ? LINOP_DAGGER : LINOP_OP);
            const Neigh_spinors neigh_spinors =
                get_neigh_spinors(_halo, spinor_in, target_cb, site);
            float *output = spinor_out.GetSiteDataPtr(0, target_cb, site);
            genericSiteOffDiagXPayz(GetNumColorSpin(), InitOp::zero, output, 1.0, gauge_links,
                                    dagger, output, neigh_spinors, ncol);
        }
    }

    // Apply a single direction of Dslash -- used for coarsening
    void CoarseDiracOp::DslashDir(CoarseSpinor &spinor_out, const CoarseGauge &gauge_in,
                                  const CoarseSpinor &spinor_in, const IndexType target_cb,
                                  const IndexType dir, const IndexType tid) const {

        // This needs to be figured out.

        IndexType min_site = _thread_limits[tid].min_site;
        IndexType max_site = _thread_limits[tid].max_site;
        const int N_colorspin = GetNumColorSpin();

        // The opposite direction
        int opp_dir = dir / 2 * 2 + 1 - dir % 2;
        if (!_halo.LocalDir(dir / 2)) {
            // Prepost receive
#pragma omp master
            {
                // Start receiving from this direction
                _halo.StartRecvFromDir(dir);
            }
            // No need for barrier here
            // Pack the opposite direction
            packFace<CoarseSpinor, CoarseAccessor>(_halo, spinor_in, 1 - target_cb, opp_dir);

            /// Need barrier to make sure all threads finished packing
#pragma omp barrier

            // Master calls MPI stuff
#pragma omp master
            {
                // Send the opposite direction
                _halo.StartSendToDir(opp_dir);
                _halo.FinishSendToDir(opp_dir);

                // Finish receiving from this direction
                _halo.FinishRecvFromDir(dir);
            }
            // Threads oughtn't start until finish is complete
#pragma omp barrier
        }

        // Site is output site
        for (IndexType site = min_site; site < max_site; ++site) {

            float *output = spinor_out.GetSiteDataPtr(0, target_cb, site);
            const float *gauge_link_dir = gauge_in.GetSiteDirDataPtr(target_cb, site, dir);

            /* The following case statement selects neighbors.
             *  It is culled from the full Dslash
             *  It of course would get complicated if some of the neighbors were in a halo
             */

            const float *neigh_spinor = GetNeighborDir<CoarseSpinor, CoarseAccessor>(
                _halo, spinor_in, dir, target_cb, site);

            // Multiply the link with the neighbor. EasyPeasy?
            CMatMultNaive(output, gauge_link_dir, neigh_spinor, N_colorspin, spinor_in.GetNCol());
        } // Loop over sites
    }

    CoarseDiracOp::CoarseDiracOp(const LatticeInfo &l_info, IndexType n_smt)
        : _lattice_info(l_info),
          _n_color(l_info.GetNumColors()),
          _n_spin(l_info.GetNumSpins()),
          _n_colorspin(_n_color * _n_spin),
          _n_smt(n_smt),
          _n_vrows(2 * _n_colorspin / VECLEN),
          _n_xh(l_info.GetCBLatticeDimensions()[0]),
          _n_x(l_info.GetLatticeDimensions()[0]),
          _n_y(l_info.GetLatticeDimensions()[1]),
          _n_z(l_info.GetLatticeDimensions()[2]),
          _n_t(l_info.GetLatticeDimensions()[3]),
          _halo(l_info) {
#pragma omp parallel
        {
#pragma omp master
            {
                // Set the number of threads
                _n_threads = omp_get_num_threads();

                // ThreadLimits give iteration bounds for a specific thread with a tid.
                // These are things like min_site, max_site, min_row, max_row etc.
                // So here I allocate one for each thread.
                _thread_limits = (ThreadLimits *)MG::MemoryAllocate(
                    _n_threads * sizeof(ThreadLimits), MG::REGULAR);
            } // omp master: barrier implied

#pragma omp barrier

            // Set the number of threads and break down into SIMD ID and Core ID
            // This requires knowledge about the order in which threads are assigned.
            //
            const int tid = omp_get_thread_num();
            const int n_cores = _n_threads / _n_smt;

            // Decompose tid into site_par_id (parallelism over sites)
            // and mv_par_id ( parallelism over rows of the matvec )
            // Order is: mv_par_id + _n_mv_parallel*site_par_id
            // Same as   smt_id + n_smt * core_id

            const int core_id = tid / _n_smt;
            const int smt_id = tid - _n_smt * core_id;
            const int n_floats_per_cacheline = MG_DEFAULT_CACHE_LINE_SIZE / sizeof(float);
            int n_cachelines = _n_vrows * VECLEN / n_floats_per_cacheline;
            int cl_per_smt = n_cachelines / _n_smt;
            if (n_cachelines % _n_smt != 0) cl_per_smt++;
            int min_cl = smt_id * cl_per_smt;
            int max_cl = MinInt((smt_id + 1) * cl_per_smt, n_cachelines);
            int min_vrow = (min_cl * n_floats_per_cacheline) / VECLEN;
            int max_vrow = (max_cl * n_floats_per_cacheline) / VECLEN;

#if 1
            _thread_limits[tid].min_vrow = min_vrow;
            _thread_limits[tid].max_vrow = max_vrow;
#else
            // Hack so that we get 1 thread per core running even in SMT mode
            if (smt_id == 0) {
                _thread_limits[tid].min_vrow = 0;
                _thread_limits[tid].max_vrow = (n_complex * _n_colorspin) / VECLEN;
            } else {
                // Non SMT threads will idle (loop limits too high)
                _thread_limits[tid].min_vrow = 1 + (n_complex * _n_colorspin) / VECLEN;
                _thread_limits[tid].max_vrow = 1 + (n_complex * _n_colorspin) / VECLEN;
            }
#endif

            // Find minimum and maximum site -- assume
            // small lattice so no blocking at this point
            // just linearly divide the sites
            const int n_sites_cb = _lattice_info.GetNumCBSites();
            int sites_per_core = n_sites_cb / n_cores;
            if (n_sites_cb % n_cores != 0) sites_per_core++;
            int min_site = core_id * sites_per_core;
            int max_site = MinInt((core_id + 1) * sites_per_core, n_sites_cb);
            _thread_limits[tid].min_site = min_site;
            _thread_limits[tid].max_site = max_site;

        } // omp parallel
    }

#ifdef MG_WRITE_COARSE
#    include <mpi.h>

    void CoarseDiracOp::write(const CoarseGauge &gauge, std::string &filename) {
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
        MPI_File_delete(filename.c_str(), MPI_INFO_NULL);
        MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
                      MPI_INFO_NULL, &fh);
        if (offset == 0) {
            float header[6] = {4,
                               (float)lattice_dims[0],
                               (float)lattice_dims[1],
                               (float)lattice_dims[2],
                               (float)lattice_dims[3],
                               (float)n_colorspin};
            MPI_Status status;
            MPI_File_write(fh, header, n_dim + 2, MPI_FLOAT, &status);
        }
        MPI_File_set_view(fh,
                          sizeof(float) * (n_dim + 2 +
                                           (n_complex * n_colorspin * n_colorspin + n_dim * 2) *
                                               (2 * n_dim + 1) * offset),
                          MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);

        // Site is output site
        const int n_sites_cb = gauge.GetInfo().GetNumCBSites();
        for (IndexType site_cb = 0; site_cb < n_sites_cb; ++site_cb) {
            for (int target_cb = 0; target_cb < 2; ++target_cb) {

                const float *gauge_base = gauge.GetSiteDirDataPtr(target_cb, site_cb, 0);
                const IndexType gdir_offset = gauge.GetLinkOffset();
                const float *clov = gauge.GetSiteDiagDataPtr(target_cb, site_cb);

                const float *gauge_links[9] = {clov,                          // Diag
                                               gauge_base,                    // X forward
                                               gauge_base + gdir_offset,      // X backward
                                               gauge_base + 2 * gdir_offset,  // Y forward
                                               gauge_base + 3 * gdir_offset,  // Y backward
                                               gauge_base + 4 * gdir_offset,  // Z forward
                                               gauge_base + 5 * gdir_offset,  // Z backward
                                               gauge_base + 6 * gdir_offset,  // T forward
                                               gauge_base + 7 * gdir_offset}; // T backward

                // Turn site into x,y,z,t coords
                IndexArray local_site_coor;
                CBIndexToCoords(site_cb, target_cb, gauge.GetInfo().GetLatticeDimensions(),
                                gauge.GetInfo().GetLatticeOrigin(), local_site_coor);

                // Compute global coordinate
                IndexArray global_site_coor;
                gauge.GetInfo().LocalCoordToGlobalCoord(global_site_coor, local_site_coor);

                // Compute neighbors
                IndexArray coors[9];
                coors[0] = global_site_coor;
                for (int i = 0, j = 1; i < 4; i++) {
                    // Forward
                    global_site_coor[i] = (global_site_coor[i] + 1) % lattice_dims[i];
                    coors[j++] = global_site_coor;

                    // Backward
                    global_site_coor[i] =
                        (global_site_coor[i] + lattice_dims[i] - 2) % lattice_dims[i];
                    coors[j++] = global_site_coor;

                    // Restore
                    global_site_coor[i] = (global_site_coor[i] + 1) % lattice_dims[i];
                }

                for (int i = 0; i < 9; i++) {
                    float coords[8] = {(float)coors[0][0], (float)coors[0][1], (float)coors[0][2],
                                       (float)coors[0][3], (float)coors[i][0], (float)coors[i][1],
                                       (float)coors[i][2], (float)coors[i][3]};
                    MPI_Status status;
                    MPI_File_write(fh, coords, 8, MPI_FLOAT, &status);
                    MPI_File_write(fh, gauge_links[i], n_complex * n_colorspin * n_colorspin,
                                   MPI_FLOAT, &status);
                }
            }
        }

        MPI_File_close(&fh);
    }
#else
    void CoarseDiracOp::write(const CoarseGauge &gauge, std::string &filename) {
        (void)gauge;
        (void)filename;
    }
#endif // MG_WRITE

} // Namespace
