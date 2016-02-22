/*
 * mg_setup.cc

 *
 *  Created on: Feb 12, 2016
 *      Author: bjoo
 */
#include <vector>
#include "lattice/mg_level.h"

namespace MG {


   void mgSetup(const SetupParams& p, std::vector< MGLevel >& mg_levels )
   {
	   MasterLog(INFO, "Setting Up MG Levels with %d levels", p.n_levels);

	   // Resize the mg_levels structure
	   mg_levels.resize(p.n_levels);

	   // Level zero is the fine level
	   mg_levels[0].info = new LatticeInfo(p.local_lattice_size);



	   MasterLog(INFO, "Level 0: Lattice Info Set.");
	   MasterLog(INFO, "Local Vol: (%d, %d, %d, %d),  Ns=%d, Nc=%d",
			   (mg_levels[0].info->GetLatticeDimensions())[0],
			   (mg_levels[0].info->GetLatticeDimensions())[1],
			   (mg_levels[0].info->GetLatticeDimensions())[2],
			   (mg_levels[0].info->GetLatticeDimensions())[3],
			   mg_levels[0].info->GetNumSpins(),
			   mg_levels[0].info->GetNumColors());



	   // Now the lower levels
	   for(int level = 1; level < p.n_levels; ++level) {
		   // First work out new lattice size
		   const IndexArray& fine_lat_size = mg_levels[level-1].info->GetLatticeDimensions();

		   IndexArray blocked_lattice_size;
		   for(int mu=0; mu < n_dim; ++mu) {
			   if ( fine_lat_size[mu] % p.block_sizes[level-1][mu] == 0 ) {
				   blocked_lattice_size[mu] = fine_lat_size[mu]/p.block_sizes[level-1][mu];
			   }
			   else {
				   MasterLog(ERROR, "Block size in dim %d (%d) does not divide previous level lattice size %d",
						   mu, p.block_sizes[level-1][mu], fine_lat_size[mu]);
			   }
		   }


		   // Reuse node info from toplevel
		   mg_levels[level].info = new LatticeInfo(blocked_lattice_size,
				   	   	   	   	   	   	   	   	   	  2,
													  p.n_vecs[level-1],
													  mg_levels[level-1].info->GetNodeInfo());
		   MasterLog(INFO, "Level %d: Lattice Info Set.", level);
		   MasterLog(INFO, "Local Vol: (%d, %d, %d, %d),  Ns=%d, Nc=%d",
				   (mg_levels[level].info->GetLatticeDimensions())[0],
				   (mg_levels[level].info->GetLatticeDimensions())[1],
				   (mg_levels[level].info->GetLatticeDimensions())[2],
				   (mg_levels[level].info->GetLatticeDimensions())[3],
				   mg_levels[level].info->GetNumSpins(),
				   mg_levels[level].info->GetNumColors());
	   }

	   // OK now the info is set up for all the levels.
	   // Now we need to allocate Null vectors themselves
	   for(int level =0; level < p.n_levels; ++level) {

		   // Create the LinOp for this level
		   // Level zero is trivial -- just the fine linOp
		   // Level 1 -- N-1 is less trivial -- need nullvecs from previous level
		  //  Assume this is taken care of here
		   mg_levels[level].M = createLinearOperator(level);

		   // If we are not on the last level, we need to create a nullspace
		   // for the level below us.
		   if ( level < p.n_levels - 1 ) {

			   // Resize the null vector pointer array
			   mg_levels[level].null_vecs.resize(p.n_vecs[level]);

			   // Allocate the vectors
			   for(int vec=0; vec < mg_levels[level].null_vecs.size(); ++vec) {
				   mg_levels[level].null_vecs[vec] = allocateSpinor(*(mg_levels[level].info), level);
			   }


			   // Create the NULL solver & smoother for this level for this level.
			   // Again, bottom level is a direct solve, so no need to make a smoother
			   mg_levels[level].null_solver = createSolver(BICGSTAB, level);
			   mg_levels[level].pre_smoother = createSolver(MR, level);
			   mg_levels[level].post_smoother = mg_levels[level].pre_smoother;

			   // Now generate the nullspace
			   // For this I need a zero vector to use as an RHS
			   MasterLog(INFO, "Level %d: Creating %d NULL vectors", level, p.n_vecs[level]);
			   {
				   Spinor* zero_vec = allocateSpinor(*(mg_levels[level].info), level);
				   zero(*zero_vec);

				   // Partially solve M x = 0 using the null solver
				   for(IndexType vec=0; vec < p.n_vecs[level]; ++vec) {
					   // Fill with random noise
					   gaussian( *(mg_levels[level].null_vecs[vec]));

					   // Solve with a NULL solver
					   (*(mg_levels[level].null_solver))(
							   *(mg_levels[level].null_vecs[vec]),
							   *(zero_vec)
							   );
				   }

				   freeSpinor(zero_vec);
				   // Block Orthogonalize the Null space in Place
				   blockOrthogonalize( mg_levels[level].null_vecs, p.block_sizes[level] );

				   mg_levels[level].R = new Restrictor(level, mg_levels[level].null_vecs);
				   mg_levels[level+1].P = new Prolongator(level, mg_levels[level].null_vecs);
			   }
		   }
		   else {
			   	 // Last Level
			   	   mg_levels[level].R = nullptr;
		   }
	   }

   }

}

