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
#include "lattice/geometry_utils.h"
#include "lattice/lattice_info.h"
#include "utils/timer.h"
#include <omp.h>
#if defined(MG_QMP_COMMS)
#    include "lattice/halo_container_qmp.h"
#else
#    include "lattice/halo_container_single.h"
#endif
namespace MG {

    using SpinorHaloCB = HaloContainer<CoarseSpinor>;
    using CoarseGaugeHaloCB = HaloContainer<CoarseSpinor>;

    template <typename T> struct CoarseAccessor {
        inline static const float *get(const T &in, int cb, int cbsite);
    };

    template <>
    inline const float *CoarseAccessor<CoarseSpinor>::get(const CoarseSpinor &in, int cb,
                                                          int cbsite) {
        return in.GetSiteDataPtr(0, cb, cbsite);
    }

    template <typename T, template <typename> class Accessor>
    inline void packFace(HaloContainer<T> &halo, const T &in, IndexType cb, IndexType dir) {
        const LatticeInfo &info = in.GetInfo();
        const IndexArray &lattice_dims = info.GetLatticeDimensions();
        const IndexArray &orig = halo.GetInfo().GetLatticeOrigin();

        // Get dimensions of the face and its origin
        IndexArray face_dims(lattice_dims);
        face_dims[dir / 2] = 1;
        IndexArray face_orig(orig);
        face_orig[dir / 2] += (lattice_dims[dir / 2] - 1) * (1 - dir % 2);

        // Grab the buffer from the Halo
        float *buffer = halo.GetSendToDirBuf(dir);

        int buffer_site_offset = halo.GetDataTypeSize();
        int buffer_sites = halo.NumSitesInFace(dir / 2);

        // Loop through the sites in the buffer
#pragma omp for
        for (int site = 0; site < buffer_sites; ++site) {
            // Get the coordinates of the site with index 'site' on the face
            IndexArray coor;
            CBIndexToCoords(site, cb, face_dims, face_orig, coor);

            // Get the local coordinates of the site
            coor[dir / 2] += (lattice_dims[dir / 2] - 1) * (1 - dir % 2);

            // Get the index of the site on the local lattice
            IndexType source_cb, body_site;
            CoordsToCBIndex(coor, lattice_dims, orig, source_cb, body_site);
            assert(source_cb == cb);

            float *buffersite = &buffer[site * buffer_site_offset];
            // Grab the body site
            const float *bodysite = Accessor<T>::get(in, cb, body_site);

            // Copy body site into buffer site
            // This is likely to be done in a thread, so
            // use SIMD if you can.
#pragma omp simd
            for (unsigned int cspin_idx = 0; cspin_idx < halo.GetDataTypeSize(); ++cspin_idx) {
                buffersite[cspin_idx] = bodysite[cspin_idx];
            } // Finish copying

        } // finish loop over sites.
    }

    template <typename T, template <typename> class Accessor>
    inline void CommunicateHaloSyncInOMPParallel(HaloContainer<T> &halo, const T &in,
                                                 const int target_cb) {

#pragma omp master
            {
                Timer::TimerAPI::startTimer("CommunicateHaloSync/sp" +
                                            std::to_string(in.GetNumColorSpin()));
                halo.setNCols(in.GetNCol());
            }
#pragma omp barrier

            for (int mu = 0; mu < 8; ++mu) {
                // Pack face; uses omp internally
                if (!halo.LocalDir(mu / 2)) {
                    packFace<T, Accessor>(halo, in, 1 - target_cb, mu);
                    packFace<T, Accessor>(halo, in, 1 - target_cb, mu);
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
                Timer::TimerAPI::stopTimer("CommunicateHaloSync/sp" +
                                           std::to_string(in.GetNumColorSpin()));
            }

            // Barrier after comms to sync master with other threads
#pragma omp barrier
    }

    template <typename T, template <typename> class Accessor>
    inline void CommunicateHaloSync(HaloContainer<T> &halo, const T &in, const int target_cb) {
        halo.setNCols(in.GetNCol());
        if (halo.NumNonLocalDirs() > 0) {
            for (int mu = 0; mu < 8; ++mu) {
                // Pack face usese omp for internally
                if (!halo.LocalDir(mu / 2)) {
                    packFace<T, Accessor>(halo, in, 1 - target_cb, mu);
                    packFace<T, Accessor>(halo, in, 1 - target_cb, mu);
                }
            }

            halo.StartAllRecvs();
            halo.StartAllSends();
            halo.FinishAllSends();
            halo.FinishAllRecvs();
        }
    }

    template <typename T, template <typename> class Accessor>
    inline const float *GetNeighborDir(const HaloContainer<T> &halo, const T &in, int dir,
                                       int target_cb, int cbsite) {
        // Local lattice size and its origin
        const IndexArray &lattice_dims = halo.GetInfo().GetLatticeDimensions();
        const IndexType cborig = halo.GetInfo().GetCBOrigin();

        // Global lattice dimensions
        IndexArray global_lattice_dims;
        halo.GetInfo().LocalDimsToGlobalDims(global_lattice_dims, lattice_dims);

        // Get the local coordinates of the site
        IndexArray coor;
        CBIndexToCoords(cbsite, target_cb, lattice_dims, cborig, coor);

        // Get the local coordinates of the neighbor in direction dir
        coor[dir / 2] += 1 - 2 * (dir % 2);

        // If the neighbor is on the local lattice, get the data from 'in'
        if (halo.LocalDir(dir / 2) ||
            (0 <= coor[dir / 2] && coor[dir / 2] < lattice_dims[dir / 2])) {
            // Avoid negative values on coordinates
            coor[dir / 2] = (coor[dir / 2] + lattice_dims[dir / 2]) % lattice_dims[dir / 2];

            // Get the index of the site
            int source_cb, source_site;
            CoordsToCBIndex(coor, lattice_dims, cborig, source_cb, source_site);
            assert(source_cb == 1 - target_cb);

            // Get the data from 'in'
            return Accessor<T>::get(in, source_cb, source_site);
        }

        // Otherwise, get the data from halo exchanged data
        // (This matches how packFace orders the sites)

        // Dimensions and origin of the face in direction 'dir'
        IndexArray face_dims(lattice_dims);
        face_dims[dir / 2] = 1;
        IndexType face_cborig = cborig + global_lattice_dims[dir / 2] +
                                lattice_dims[dir / 2] * (1 - dir % 2) - (dir % 2);

        // Get coordinates of the site on the face
        coor[dir / 2] = 0;

        // Get index of the site on the face
        int source_cb, source_site;
        CoordsToCBIndex(coor, face_dims, face_cborig, source_cb, source_site);
        assert(source_cb == 1 - target_cb);

        // Grab the data from the halo buffer
        return &(halo.GetRecvFromDirBuf(dir)[halo.GetDataTypeSize() * source_site]);
    }

    template <typename T, template <typename> class Accessor>
    inline std::array<const float *, 8> GetNeighborDirs(const HaloContainer<T> &halo, const T &in,
                                                        int target_cb, int cbsite) {
        // Local lattice size and its origin
        const IndexArray &lattice_dims = halo.GetInfo().GetLatticeDimensions();
        const IndexType cborig = halo.GetInfo().GetCBOrigin();

        // Global lattice dimensions
        IndexArray global_lattice_dims;
        halo.GetInfo().LocalDimsToGlobalDims(global_lattice_dims, lattice_dims);

        // Get the local coordinates of the site
        IndexArray coor;
        CBIndexToCoords(cbsite, target_cb, lattice_dims, cborig, coor);

        std::array<const float *, 8> neigbors;

        for (int dir = 0; dir < 8; ++dir) {
            IndexType coor_at_dir = coor[dir / 2];

            // Get the local coordinates of the neighbor in direction dir
            coor[dir / 2] += 1 - 2 * (dir % 2);

            // If the neighbor is on the local lattice, get the data from 'in'
            if (halo.LocalDir(dir / 2) ||
                (0 <= coor[dir / 2] && coor[dir / 2] < lattice_dims[dir / 2])) {
                // Avoid negative values on coordinates
                coor[dir / 2] = (coor[dir / 2] + lattice_dims[dir / 2]) % lattice_dims[dir / 2];

                // Get the index of the site
                int source_cb, source_site;
                CoordsToCBIndex(coor, lattice_dims, cborig, source_cb, source_site);
                assert(source_cb == 1 - target_cb);

                // Get the data from 'in'
                neigbors[dir] = Accessor<T>::get(in, source_cb, source_site);
            } else {
                // Otherwise, get the data from halo exchanged data
                // (This matches how packFace orders the sites)

                // Dimensions and origin of the face in direction 'dir'
                IndexArray face_dims(lattice_dims);
                face_dims[dir / 2] = 1;
                IndexType face_cborig = cborig + global_lattice_dims[dir / 2] +
                                        lattice_dims[dir / 2] * (1 - dir % 2) - (dir % 2);

                // Get coordinates of the site on the face
                coor[dir / 2] = 0;

                // Get index of the site on the face
                int source_cb, source_site;
                CoordsToCBIndex(coor, face_dims, face_cborig, source_cb, source_site);
                assert(source_cb == 1 - target_cb);

                // Grab the data from the halo buffer
                neigbors[dir] =
                    &(halo.GetRecvFromDirBuf(dir)[halo.GetDataTypeSize() * source_site]);
            }

            // Correct back the original coordinate
            coor[dir / 2] = coor_at_dir;
        }

        return neigbors;
    }

} // namespace

#endif /* INCLUDE_LATTICE_SPINOR_HALO_H_ */
