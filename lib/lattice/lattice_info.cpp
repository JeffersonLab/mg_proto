

#include "lattice/lattice_info.h"
#include "lattice/constants.h"
#include "lattice/indexers.h"
#include "lattice/nodeinfo.h"

#include "utils/print_utils.h"

#include <omp.h>

/* These are for sorting */
#include <algorithm>
#include <functional>
using namespace MGUtils;

namespace MGGeometry { 

LatticeInfo::LatticeInfo(const std::vector<unsigned int>& lat_dims,
		 const unsigned int n_spin,
		 const unsigned int n_color,
		 const NodeInfo& node) : _n_color(n_color),
		 _n_spin(n_spin), _node_info(node)
{
	if( omp_in_parallel() ) {
		LocalLog(ERROR, "Creating LatticeInfo within OMP region");
	}

	/* Copy in the LatticeInfo Dimensions */
	if( lat_dims.size() == n_dim ) {
		for(unsigned int mu=0; mu < n_dim; ++mu) {
			_lat_dims[mu] = lat_dims[mu];
		}
	}
	else {
		MasterLog(ERROR, "lat_dims size is different from n_dim");
		/* This should abort */
	}

	/* Count the number of sites */
	_n_sites = _lat_dims[0];
	for(unsigned int mu=1; mu < n_dim; ++mu) {
		_n_sites *= _lat_dims[mu];
	}

	/* Compute the origin of the lattice */
	const std::vector<unsigned int>& node_coords = _node_info.NodeCoords();
	int _sum_orig_coords = 0;
	for(unsigned int mu=0; mu < n_dim; ++mu) {
		_lat_origin[mu] = node_coords[mu]*_lat_dims[mu];
		_sum_orig_coords += _lat_origin[mu];
	}

	/* Count sites of various checkerboards */
	_n_cb_sites[0] = 0;
	_n_cb_sites[1] = 0;

	// We are in an enclosing parallel region
	// so we need omp for
	// NB: This is a kind of a histogramming operation
	// One way to do it is to collec local histograms and
	// Accumilate in the a critical region at the end.
#pragma omp parallel shared(_sum_orig_coords)
	{
		unsigned int priv_n_cb_sites[2] = {0,0};

#pragma omp for collapse(4) nowait
		for( unsigned int t = 0; t < _lat_dims[T_DIR]; ++t) {
			for( unsigned int z = 0 ; z < _lat_dims[Z_DIR]; ++z) {
				for(unsigned int y = 0; y < _lat_dims[Y_DIR]; ++y) {
					for(unsigned int x = 0; x < _lat_dims[X_DIR]; ++x) {
						int cb = (_sum_orig_coords + x + y + z + t) & 1;
						priv_n_cb_sites[cb]++;
					} // t
				} // z
			} //y
		} // x

#pragma omp critical
		{
		_n_cb_sites[0] += priv_n_cb_sites[0]++;
		_n_cb_sites[1] += priv_n_cb_sites[1]++;
		}

	} //End OMP Parallel section

	/* Now I need info from NodeInfo, primarily my node coordinates
	 * so that I can use the local lattice dims and the node coords
	 * to find out the checkerboard of my origin, and/or to find my
	 * global coordinates
	 */
	for(int cb=0; cb < 2; ++cb) {
		_cb_sites[cb].resize(_n_cb_sites[cb] );
	}

	LatticeCartesianSiteIndexer site_indexer(_lat_dims[0],
											 _lat_dims[1],
											 _lat_dims[2],
											 _lat_dims[3] );

	int cb_fill[2] = {0,0}; // These should count where we are filling the site
							// tables

#pragma omp parallel shared(cb_fill, site_indexer)
	{
		// Each thread can fill its own private version of the site table
		unsigned int priv_cb_fill[2] = {0,0}; // Private progress counters

		std::vector<unsigned int> priv_cb_sites[2]; // Private site tables
		priv_cb_sites[0].resize(_n_cb_sites[0] );
		priv_cb_sites[1].resize(_n_cb_sites[1] );

		// Now a parallel for clause to accumulate the threads local site table
		// nowait is OK, since after the site table is done, it is copied
		// to the global, and that is guaranteed with an OMP critical so
		// I am unable to update the other threads work.
		//
#pragma omp for collapse(4) nowait
		for( unsigned int t = 0; t < _lat_dims[T_DIR]; ++t) {
			for( unsigned int z = 0 ; z < _lat_dims[Z_DIR]; ++z) {
				for(unsigned int y = 0; y < _lat_dims[Y_DIR]; ++y) {
					for(unsigned int x = 0; x < _lat_dims[X_DIR]; ++x) {

						unsigned int cb = (_sum_orig_coords + x + y + z + t) & 1;
						unsigned int index = site_indexer.Index(x,y,z,t);

						priv_cb_sites[cb][priv_cb_fill[cb]] = index;
						priv_cb_fill[cb]++;

					} // t
				} // z
			} //y
		} // x

		// OK this thread has his own private site table now.
		// Let's copy it into the global site table. To avoid races
		// I need a critical section here.
#pragma omp critical
		{
			for(int cb=0; cb < 2; ++cb) {
				// Copy in my part, starting from cb_fill
				for(unsigned int i=0; i < priv_cb_fill[cb]; i++) {
					_cb_sites[cb][ cb_fill[cb]+i ] = priv_cb_sites[cb][i];
				}
				// Increase cb_fill
				cb_fill[cb] += priv_cb_fill[cb];
			}
		}
	} // END OMP PARALLEL REGION

	// Count surfaces. This is tedious. There has to be a better way.

	unsigned int face_volume[n_dim] = { _lat_dims[1] * _lat_dims[2] * _lat_dims[3], // X-face: YZT
									_lat_dims[0] * _lat_dims[2] * _lat_dims[3],	// Y-face: XZT
									_lat_dims[0] * _lat_dims[1] * _lat_dims[3], // Z-face: XYT
									_lat_dims[0] * _lat_dims[1] * _lat_dims[2]  // T-face: XYZ
									};


#pragma omp parallel shared(face_volume, site_indexer)
	{

		std::vector<unsigned int> priv_face_sites[n_dim][2][2];


		for (unsigned int dir = 0; dir < n_dim; dir++) {
#pragma omp for nowait
			for (unsigned int site_3d = 0; site_3d < face_volume[dir]; site_3d++) {

				unsigned int coord_0_0 = 0;           // back face coordinate
				unsigned int coord_0_N = _lat_dims[dir] - 1; // forw face coordinate

				/* The other 3 coordinates */
				unsigned int coord_1 = 0;
				unsigned int coord_2 = 0;
				unsigned int coord_3 = 0;

				/* Compute the 3 coords depending on Dir */
				site_indexer.Coords3D(dir, site_3d, coord_1, coord_2, coord_3);

				/* Compute the corresponding 4D site indices */
				unsigned int back_face_site_4d = 0;
				unsigned int forw_face_site_4d = 0;
				switch (dir) {
				case X_DIR:
					back_face_site_4d = site_indexer.Index(coord_0_0, coord_1,
							coord_2, coord_3);
					forw_face_site_4d = site_indexer.Index(coord_0_N, coord_1,
							coord_2, coord_3);
					break;
				case Y_DIR:
					back_face_site_4d = site_indexer.Index(coord_1, coord_0_0,
							coord_2, coord_3);
					forw_face_site_4d = site_indexer.Index(coord_1, coord_0_N,
							coord_2, coord_3);
					break;
				case Z_DIR:
					back_face_site_4d = site_indexer.Index(coord_1, coord_2,
							coord_0_0, coord_3);
					forw_face_site_4d = site_indexer.Index(coord_1, coord_2,
							coord_0_N, coord_3);
					break;
				case T_DIR:
					back_face_site_4d = site_indexer.Index(coord_1, coord_2,
							coord_3, coord_0_0);
					forw_face_site_4d = site_indexer.Index(coord_1, coord_2,
							coord_3, coord_0_N);
					break;
				default:
					MasterLog(ERROR, "One should not reach here");
					break;
				}

				/* Compute the checkerboards */
				unsigned int back_cb = (_sum_orig_coords + coord_0_0 + coord_1
					+ coord_2 + coord_3) & 1;
				unsigned int forw_cb = (_sum_orig_coords + coord_0_N + coord_1
					+ coord_2 + coord_3) & 1;

				priv_face_sites[dir][BACKWARD][back_cb].push_back(back_face_site_4d);
				priv_face_sites[dir][FORWARD][forw_cb].push_back(forw_face_site_4d);
			}
		}




#pragma omp critical
		{
			for(unsigned int dir=0; dir < n_dim; ++dir) {
				for(unsigned int fb = static_cast<unsigned int>(BACKWARD); fb <= static_cast<unsigned int>(FORWARD); ++fb) {
					for(unsigned int cb =0; cb < 2; ++cb) {
						// Append private list
						_surface_sites[dir][fb][cb].insert(_surface_sites[dir][fb][cb].end(),
													   priv_face_sites[dir][fb][cb].begin(),
													   priv_face_sites[dir][fb][cb].end());
					}
				}
			}
		}
	}

	// OK right now the site tables are unsorted I should sort them?
	// NB: This is annoying in the sense that it is not parallel
	for(int cb = 0; cb < 2;++cb) {
		std::sort(_cb_sites[cb].begin(), _cb_sites[cb].end(), std::greater<unsigned int>());
	}

	for(unsigned int dir=0; dir < n_dim; ++dir) {
		for(unsigned int fb = static_cast<unsigned int>(BACKWARD); fb <= static_cast<unsigned int>(FORWARD); ++fb) {
			for(unsigned int cb =0; cb < 2; ++cb) {
				std::sort(_surface_sites[dir][fb][cb].begin(),
						  _surface_sites[dir][fb][cb].end(), std::greater<unsigned int>());
			}
		}
	}
}

/** Destructor for the lattice class
 */
LatticeInfo::~LatticeInfo() {}


};
