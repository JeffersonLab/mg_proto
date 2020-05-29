/*
 * coarse_op_profile.cpp
 *
 *  Created on: Feb 13, 2018
 *      Author: bjoo
 */
#include "gtest/gtest.h"
#include "lattice/cmat_mult.h"
#include "utils/memory.h"
#include "utils/print_utils.h"
#include <random>
#include "MG_config.h"
#include "../test_env.h"
#include "../mock_nodeinfo.h"

#include <omp.h>
#include <cstdio>

#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_op.h"

using namespace MG;
using namespace MGTesting;

#include "../vol_and_block_args.h"

VolAndBlockArgs args({{16,16,16,16}},{{2,2,2,2}},24,24,1,10);

TEST(CoarseDslash, TestSpeed)
{
	IndexArray latdims=args.ldims;
	IndexArray pe_dims={{1,1,1,1}};  // Pretend 32 nodes
	IndexArray pe_coords={{0,0,0,0}};
	MockNodeInfo mock_node(pe_dims, pe_coords);
	//LatticeInfo linfo(latdims, 2, args.fine_colors, mock_node);
	LatticeInfo linfo(latdims, 2, 12, mock_node);
	std::vector<int> ncols = {1, 2, 4, 5, 16, 64, 256};
	for (int ncoli = 0; ncoli < ncols.size(); ncoli++) {
		int ncol = ncols[ncoli];

		CoarseSpinor x_spinor(linfo, ncol);
		CoarseSpinor y_spinor(linfo, ncol);
		CoarseGauge gauge(linfo);

		const int N_iter =args.iter;
		const int n_smt= 1 ;

		double total_time[288][8]; // Timing info for 72 cores x 4 threads

		// Create Coarse Dirac Op.
		CoarseDiracOp D(linfo,n_smt);

		const int N_dir = 2*n_dim;
		const int N_sites_cb = linfo.GetNumCBSites();
		const int N = D.GetNumColorSpin();

#pragma omp parallel
		{
			const int tid = omp_get_thread_num();
			const int n_threads=omp_get_num_threads();

			// One thread per site -- fill fields with random junk

#pragma omp for schedule(static)
			for(IndexType site=0; site < N_sites_cb; ++site) {
				// Fill spinors with some junk
				for (int col=0; col < ncol; ++col) {
					for(int j=0; j < n_complex*N; ++j) {
						x_spinor.GetSiteDataPtr(col, (IndexType)0,site)[j] = j*ncol + col + site;
						x_spinor.GetSiteDataPtr(col, (IndexType)1,site)[j] = -(j*ncol + col);
						y_spinor.GetSiteDataPtr(col, (IndexType)0,site)[j] = 0;
						y_spinor.GetSiteDataPtr(col, (IndexType)1,site)[j] = 0;
					}
				}

				for(int dir=0; dir < 8; ++dir) {
					for(int row=0; row < n_complex*N; ++row) {
						for(int col=0; col < N; col++) {

							gauge.GetSiteDirDataPtr(0,site,dir)[ row + n_complex*N*col ] = (dir + row*8 + col*8*n_complex*N + site) % 97;
							gauge.GetSiteDirDataPtr(1,site,dir)[ row + n_complex*N*col ] = (dir + row*8 + col*8*n_complex*N + site) % 101;
						}
					}
				}

				for(int dir=0; dir < 8; ++dir) {
				for(int row=0; row < (N/2); ++row) {
					for(int col=0; col < N; col++) {
						for(int z=0; z < n_complex; ++z ) {

							// CB=0 Chiral up
							gauge.GetSiteDirADDataPtr(0,site,dir)[ z + n_complex*(col + (N/2)*row) ] = (z + col*n_complex + row*n_complex*N + site) % 97;

							// CB=0 Chiral down
							gauge.GetSiteDirADDataPtr(0,site,dir)[ z + n_complex*(col + (N/2)*(row + (N/2)) ) ] = (z + col*n_complex + row*n_complex*N + site) % 97;

							// CB=1 Chiral up
							gauge.GetSiteDirADDataPtr(1,site,dir)[ z + n_complex*(col + (N/2)*row) ] = (z + col*n_complex + row*n_complex*N + site) % 97;

							// CB=1 Chiral down
							gauge.GetSiteDirADDataPtr(1,site,dir)[ z + n_complex*(col + (N/2)*(row + (N/2)) ) ] = (z + col*n_complex + row*n_complex*N + site) % 97;


						}
					}
				}
				}
			}
		}

		//Gaussian(x_spinor, SUBSET_ALL);
		double outer_start_time = omp_get_wtime();
#pragma omp parallel shared(x_spinor,y_spinor, gauge)	// Make sure all the data has been filled.
		{
			// Block rows, find minimum and maximum vrow
			int tid=omp_get_thread_num();
			double start_time = omp_get_wtime();

			for(int iter = 0; iter < N_iter; ++iter) {
				D.unprecOp(y_spinor,gauge,x_spinor,0,LINOP_DAGGER,tid);
			} // iter

			double end_time = omp_get_wtime();
			total_time[tid][0]= end_time - start_time;
		} // omp parallel
		double outer_end_time = omp_get_wtime();



		double N_dble = static_cast<double>(N);
		double N_iter_dble = static_cast<double>(N_iter);
		double N_sites_cb_dble = static_cast<double>(N_sites_cb);
		double gflops=ncol*N_sites_cb*N_iter_dble*(N_dir*(N_dble*(8*N_dble-2))+(N_dir-1)*2*N)/1.0e9;

		double min_time=total_time[0][0];
		double max_time =total_time[0][0];
		double avg_time =total_time[0][0];
		for(int thread=1; thread < omp_get_max_threads(); ++thread) {
			if( total_time[thread][0] > max_time ) max_time = total_time[thread][0];
			if( total_time[thread][0] < min_time ) min_time = total_time[thread][0];
			avg_time += total_time[thread][0];
		}
		avg_time /= omp_get_max_threads();
		double outer_time = outer_end_time - outer_start_time;
		MasterLog(INFO, "== Cols %d ==", ncol);
		MasterLog(INFO, "Outer time=%16.8e (sec) => GFLOPS=%16.8e", outer_time,gflops/outer_time);
		MasterLog(INFO, "Average time=%16.8e (sec) => GFLOPs = %16.8e", avg_time, gflops/avg_time);
		MasterLog(INFO, "Min time=%16.8e (sec) => GFLOPs = %16.8e", min_time, gflops/min_time);
		MasterLog(INFO, "Max time=%16.8e (sec) => GFLOPs = %16.8e", max_time, gflops/max_time);
	}
}

int main(int argc, char *argv[])
{
	return MGTesting::TestMain(&argc, argv);
}




