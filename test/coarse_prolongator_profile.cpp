/*
 * coarse_prolongator_profile.cpp
 *
 *  Created on: Mar 19, 2018
 *      Author: bjoo
 */


/*
 * prolongator_profile.cpp
 *
 *  Created on: Mar 1, 2018
 *      Author: bjoo
 */



#include "gtest/gtest.h"
#include "lattice/cmat_mult.h"
#include "utils/memory.h"
#include "utils/print_utils.h"
#include <random>
#include "MG_config.h"
#include "./test_env.h"

#include <omp.h>
#include <cstdio>


#include "lattice/coarse/coarse_types.h"
#include <memory>
#include <vector>
#include <cmath>

#include <lattice/coarse/block.h>

#include <lattice/coarse/coarse_l1_blas.h>
#include <lattice/coarse/aggregate_block_coarse.h>
#include <lattice/coarse/coarse_transfer.h>
#include <utils/auxiliary.h>

using namespace MG;
using namespace MGTesting;
using namespace QDP;
using namespace MG::aux;



TEST(Timing, ProlongatorProfile)
{
	IndexArray latdims={{4,4,4,4}};

	const int n_fine =24;
	const int num_vecs = 32;


	MasterLog(INFO, "Testing Prolongators with n_fine_colors=%d n_coarse_colors=%d",n_fine,num_vecs);

	IndexArray node_orig=NodeInfo().NodeCoords();
	for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];
	LatticeInfo fine_info(node_orig,latdims,2,n_fine,NodeInfo());



	std::vector<std::shared_ptr<CoarseSpinor>> null_vecs(num_vecs);
	for(int k=0; k < num_vecs; ++k) {

	    null_vecs[k]= std::make_shared<CoarseSpinor>(fine_info);
	    Gaussian(*(null_vecs[k]));
	}

	IndexArray blocked_lattice_dims;
	IndexArray blocked_lattice_orig;
	IndexArray block_size={2,2,2,2};
	std::vector<Block> blocklist;

	CreateBlockList(blocklist,
				blocked_lattice_dims,
				blocked_lattice_orig,
				latdims,
				block_size,
				fine_info.GetLatticeOrigin());

	// Orthonormalize the vectors -- I heard once that for GS stability is improved
	// if you do it twice.
	MasterLog(INFO, "MG Level 0: Block Orthogonalizing Aggregates");



	  orthonormalizeBlockAggregates(null_vecs,
	                    blocklist);

	  orthonormalizeBlockAggregates(null_vecs,
			  	  	  	  blocklist);


	  LatticeInfo coarse_info(blocked_lattice_orig,
				  blocked_lattice_dims,
				  2, num_vecs, NodeInfo());

	  CoarseTransfer Transf(blocklist,null_vecs);

	  const int ncol = 3;
	  {
		  MasterLog(INFO, "Testing Prolongator");
		  CoarseSpinor coarse(coarse_info, ncol);
		  Gaussian(coarse);

		  CoarseSpinor fine(fine_info, ncol);

		  CoarseSpinor fine2(fine_info, ncol);
		  CoarseSpinor diff_v(fine_info, ncol);
		  ZeroVec(fine);
		  ZeroVec(fine2);
		  ZeroVec(diff_v);

		  prolongateSpinor(blocklist, null_vecs, coarse, fine);

		  Transf.P(coarse,fine2);

		  std::vector<double> ref = Norm2Vec(fine);
		  std::vector<double> ref2 = Norm2Vec(fine2);

		  XmyzVec(fine2,fine,diff_v);
		  std::vector<double> norm2_diff = Norm2Vec(diff_v);
		  for (int col=0; col < ncol; ++col) {
			MasterLog(INFO," == Col %d ===", col);
			MasterLog(INFO,"Fine Vector has norm=%16.8e", sqrt(ref[col]));
			MasterLog(INFO,"Fine Vector2 has norm=%16.8e",sqrt(ref2[col]));
			MasterLog(INFO, "norm_diff=%16.8e",sqrt(norm2_diff[col]));
		 	double rel_norm_diff = std::sqrt(norm2_diff[col]/ref[col]);
			MasterLog(INFO, "rel_norm_diff = %16.8e", rel_norm_diff);
			  double tol=1.0e-6;
			  if (rel_norm_diff > 1.0e-6) {
			//  ASSERT_LT( rel_norm_diff, tol );
			  int n_col=fine_info.GetNumColors();
			  for(int cb=0; cb < n_checkerboard; ++cb) {
				  for(int site=0; site < fine_info.GetNumCBSites(); ++site) {
					  const float *finesite=fine.GetSiteDataPtr(col,cb,site);
					  const float *fine2site = fine2.GetSiteDataPtr(col,cb,site);
					  const float *diffsite= diff_v.GetSiteDataPtr(col,cb,site);
	
					  for(int chiral =0; chiral < 2; ++chiral ) {
						  for(int color=0; color < n_col;  ++color) {
							  MasterLog(INFO, "cb=%d site=%d chiral=%d color=%d fine=(%16.8e,%16.8e) fine2 =(%16.8e,%16.8e) diff=(%16.8e,%16.8e)",
									  cb, site, chiral, color,
									  finesite[RE+n_complex*(color + n_col*chiral )],
									  finesite[IM+n_complex*(color + n_col*chiral )],
									  fine2site[RE+n_complex*(color + n_col*chiral )],
									  fine2site[IM+n_complex*(color + n_col*chiral )],
									  diffsite[RE+n_complex*(color + n_col*chiral )],
									  diffsite[IM+n_complex*(color + n_col*chiral )]);
	
						  }
					  }
				  }
			  }
			  }
		  }
	  }




#if 0
	  {
	    int N_iters=5000;
	    MasterLog(INFO, "Timing Prolongator with %d iterations", N_iters);
	    double start_time = omp_get_wtime();
	    for(int i=0; i < N_iters; ++i ) {
	      prolongateSpinor(blocklist, null_vecs, coarse, fine);
	    }
	    double end_time = omp_get_wtime();
	    double total_time=end_time - start_time;


	    double Gflops = (double)N_iters
	      *(double)fine_info.GetNumSites()
	      *(double)(2*n_fine*num_vecs*8)/1.0E9;

	    double Gflops_per_sec = Gflops/total_time;
	    MasterLog(INFO, "Prolongator time = %lf", total_time);
	    MasterLog(INFO, "GFLOPS = %lf", Gflops_per_sec);
	  }

#endif

	  int N_iters=5000;
	  MasterLog(INFO, "Timing Opt. Prolongator with %d iterations",N_iters);

	  std::vector<int> ncols = {1, 4, 16, 64, 256};
	  for (int ncoli = 0; ncoli < ncols.size(); ncoli++) {
	    int ncol = ncols[ncoli];
	    MasterLog(INFO, "== Cols %d ==", ncol);
	    CoarseSpinor fine(fine_info, ncol);
	    CoarseSpinor coarse(coarse_info, ncol);

	    Gaussian(coarse);
	    Gaussian(fine);

	    double start_time = omp_get_wtime();
	    for(int i=0; i < N_iters; ++i ) {
                Transf.P(coarse,fine);
	    }
	    double end_time = omp_get_wtime();
	    double total_time=end_time - start_time;

	    //   #blocks * #sites_in_block = GetNumSites()
	    //
	    double Gflops = (double)N_iters*ncol
	      *(double)fine_info.GetNumSites()
	      *(double)(2*n_fine*num_vecs*8)/1.0E9;
	    double Gflops_per_sec = Gflops/total_time;
	    MasterLog(INFO, "Opt. Prolongator time = %lf", total_time);
	    MasterLog(INFO, "GFLOPS = %lf", Gflops_per_sec);
	  }

}

int main(int argc, char *argv[])
{
	return MGTesting::TestMain(&argc, argv);
}


