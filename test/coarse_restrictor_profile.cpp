/*
 * coarse_restrictor_profile.cpp
 *
 *  Created on: Mar 19, 2018
 *      Author: bjoo
 */

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


#include "vol_and_block_args.h"

using namespace MG;
using namespace MGTesting;
using namespace QDP;


VolAndBlockArgs args({{4,4,4,4}},{{2,2,2,2}},24,24,1,10000);

TEST(Timing,RestrictorProfile)
{
	args.Dump();
	IndexArray latdims=args.ldims;

	const int n_fine = args.fine_colors;
	const int num_vecs = args.nvec;

	MasterLog(INFO, "Testing Restrictors with n_fine_colors=%d n_coarse_colors=%d",n_fine,num_vecs);

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
	IndexArray block_size=args.bdims;
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

	  CoarseTransfer Transf(blocklist,null_vecs,args.bthreads);

	  {
		  MasterLog(INFO, "Testing Restrictor");
		  CoarseSpinor coarse(coarse_info);
		  CoarseSpinor coarse2(coarse_info);

		  CoarseSpinor fine(fine_info);
		  Gaussian(fine);

		 restrictSpinor(blocklist, null_vecs, fine, coarse);

		  Transf.R(fine,coarse2);

		  double ref = Norm2Vec(coarse);
		  MasterLog(INFO,"Coarse Vector has norm=%lf", std::sqrt(ref));
		  double ref2 = Norm2Vec(coarse2);
		  MasterLog(INFO,"Coarse Vector has norm=%lf", std::sqrt(ref2));

		  double norm_diff = std::sqrt(XmyNorm2Vec(coarse,coarse2));
		  double rel_norm_diff = norm_diff/std::sqrt(ref);
		  MasterLog(INFO, "norm_diff=%16.8e",norm_diff);
		  MasterLog(INFO, "rel_norm_diff = %16.8e", rel_norm_diff);
		  double tol=1.0e-6;
		  ASSERT_LT( rel_norm_diff, tol );
	  }




	  CoarseSpinor fine(fine_info);
	  CoarseSpinor coarse(coarse_info);

	  Gaussian(fine);


#if 0
	  {
	    int N_iters=args.iter;
	    MasterLog(INFO, "Timing Restrictor with %d iterations", N_iters);
	    double start_time = omp_get_wtime();
	    for(int i=0; i < N_iters; ++i ) {
	    	restrictSpinor(blocklist, null_vecs, fine,coarse);
	    }
	    double end_time = omp_get_wtime();
	    double total_time=end_time - start_time;


	    double Gflops = (double)N_iters
	      *(double)fine_info.GetNumSites()
	      *(double)(2*n_fine*num_vecs*8)/1.0E9;

	    double Gflops_per_sec = Gflops/total_time;
	    MasterLog(INFO, "Restrictorr time = %lf", total_time);
	    MasterLog(INFO, "GFLOPS = %lf", Gflops_per_sec);
	  }

#endif

	  {
	    int N_iters=args.iter;
	    MasterLog(INFO, "Timing Opt. Restrictor with %d iterations",N_iters);

	    double start_time = omp_get_wtime();
	    for(int i=0; i < N_iters; ++i ) {
                Transf.R(fine,coarse);;
	    }
	    double end_time = omp_get_wtime();
	    double total_time=end_time - start_time;

	    //   #blocks * #sites_in_block = GetNumSites()
	    //
	    double Gflops = (double)N_iters
	      *(double)fine_info.GetNumSites()
	      *(double)(2*n_fine*num_vecs*8)/1.0E9;
	    double Gflops_per_sec = Gflops/total_time;
	    MasterLog(INFO, "Opt.  Restrictor time = %lf", total_time);
	    MasterLog(INFO, "GFLOPS = %lf", Gflops_per_sec);
	  }

}

int main(int argc, char *argv[])
{
  args.ProcessArgs(argc,argv);
  return MGTesting::TestMain(&argc, argv);
}






