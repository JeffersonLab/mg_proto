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
#include "../test_env.h"

#include <omp.h>
#include <cstdio>

#include "../qdpxx/qdpxx_latticeinit.h"
#include "lattice/qphix/qphix_types.h"
#include "lattice/coarse/coarse_types.h"
//#include "lattice/coarse/coarse_op.h"
#include <memory>
#include <vector>
#include <cmath>

#include <lattice/qphix/mg_level_qphix.h>
// #include <lattice/qphix/qphix_veclen.h>
#include <lattice/qphix/qphix_blas_wrappers.h>
#include <lattice/qphix/qphix_aggregate.h>
#include <lattice/qphix/qphix_qdp_utils.h>
#include <lattice/coarse/block.h>

#include <lattice/qphix/qphix_transfer.h>

using namespace MG;
using namespace MGTesting;
using namespace QDP;



TEST(Timing, RestrictorProfile)
{
	IndexArray latdims={{16,16,16,16}};

	initQDPXXLattice(latdims);


	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	IndexArray node_orig=NodeInfo().NodeCoords();
	for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];
	LatticeInfo fine_info(node_orig,latdims,4,3,NodeInfo());

	const int num_vecs = 24;
	std::vector<std::shared_ptr<QPhiXSpinorF>> null_vecs(num_vecs);
	for(int k=0; k < num_vecs; ++k) {

	    null_vecs[k]= std::make_shared<QPhiXSpinorF>(fine_info);
	    Gaussian(*(null_vecs[k]));
	}

	IndexArray blocked_lattice_dims;
	IndexArray blocked_lattice_orig;
	IndexArray block_size={4,4,4,4};
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

	  QPhiXTransfer<QPhiXSpinorF> Transf(blocklist,null_vecs);

	  {
		  MasterLog(INFO, "Testing RestrictorArray");
		  QPhiXSpinorF fine(fine_info);
		  Gaussian(fine);

		  CoarseSpinor coarse(coarse_info);
		  CoarseSpinor coarse2(coarse_info);

		  restrictSpinor(blocklist, null_vecs, fine ,coarse);

		  Transf.R(fine,coarse2);

		  double ref = Norm2Vec(coarse);
		  MasterLog(INFO,"Coarse Vector has norm=%lf", sqrt(ref));
		  double ref2 = Norm2Vec(coarse2);
		  MasterLog(INFO,"Coarse Vector has norm=%lf", sqrt(ref2));

		  double norm_diff = sqrt(XmyNorm2Vec(coarse,coarse2));
		  double rel_norm_diff = norm_diff/sqrt(ref);
		  MasterLog(INFO, "norm_diff=%16.8e",norm_diff);
		  MasterLog(INFO, "rel_norm_diff = %16.8e", rel_norm_diff);
		  double tol=5.0e-6;
		  ASSERT_LT( rel_norm_diff, tol );
	  }




	  QPhiXSpinorF fine(fine_info);
	  CoarseSpinor coarse(coarse_info);

	  Gaussian(fine);
	  Gaussian(coarse);


	  {
		  int N_iters=500;
		  MasterLog(INFO, "Timing Restrictor with %d iterations", N_iters);
		  double start_time = omp_get_wtime();
		  for(int i=0; i < N_iters; ++i ) {
			  restrictSpinor(blocklist, null_vecs, fine ,coarse);
		  }
		  double end_time = omp_get_wtime();
		  double total_time=end_time - start_time;

		  //   #blocks * #sites_in_block = GetNumSites()
		  //

		  double Gflops = (double)N_iters
				  *(double)fine_info.GetNumSites()
				  *(double)(2*3*(2*num_vecs)*8)/1.0E9;
		  double Gflops_per_sec = Gflops/total_time;
		  MasterLog(INFO, "Restrictor time = %lf", total_time);
		  MasterLog(INFO, "GFLOPS = %lf", Gflops_per_sec);
	  }


	  {
			  int N_iters=500;
			  MasterLog(INFO, "Timing Restrictor with %d iterations",N_iters);

			  double start_time = omp_get_wtime();
			  for(int i=0; i < N_iters; ++i ) {
				  Transf.R(fine ,coarse);
			  }
			  double end_time = omp_get_wtime();
			  double total_time=end_time - start_time;

			  //   #blocks * #sites_in_block = GetNumSites()
			  //
			  double Gflops = (double)N_iters
					  *(double)fine_info.GetNumSites()
					  *(double)(2*3*(2*num_vecs)*8)/1.0E9;
			  double Gflops_per_sec = Gflops/total_time;
			  MasterLog(INFO, "Restrictor New time = %lf", total_time);
			  MasterLog(INFO, "GFLOPS = %lf", Gflops_per_sec);
		  }

}



int main(int argc, char *argv[])
{
	return MGTesting::TestMain(&argc, argv);
}
