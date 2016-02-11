/*
 * playpen.cpp
 *
 *  Created on: Oct 12, 2015
 *      Author: bjoo
 */

#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/nodeinfo.h"
#include "lattice/aggregation.h"
#include "utils/print_utils.h"

#include <vector>

using namespace MG;
using namespace MG;


class CoarseOperator {
public:
	CoarseOperator(const LatticeInfo& fine_lattice_info,
				   const Aggregation& aggregation,
				   const LatticeInfo& coarse_lattice_info,
				   const LatticeFermionArray& basis_vectors,
				   const LinearOperator<LatticeFermion>& M ) :
					   	  _fine_lattice_info(fine_lattice_info),
						  _aggregation(aggregation),
						  _coarse_lattice_info(coarse_lattice_info),
						  _coarse_op_fields(coarse_lattice_info,aggregation) {

		// Ultimately the operator looks like
		//
		//   M^{i,j,alpha,beta}(x,y) \Phi^{j,beta}(y) =
		//       M^{i,j,alpha,beta}(x)       \Phi^{j,beta}(x)
		//     + M^{i,j,alpha,beta}(x + mu)  \Phi^{j,beta)(x+mu)
		//     + M^{i,j,alpha,beta}(x - mu)  \Phi^{j,beta}(x-mu)
		//
		//  With M^{i,j,alpha,beta}(x,x) =  < v^{i,alpha}(x) M v^{j,beta}(x) >  -- Local
		//       M^{i,j,alpha,beta}(x,x+mu) = < v^{i,alpha}(x) M v^{j,beta}(x+mu) > -- Needs  M v^{j,beta}(x+mu)  -- communicate block
	 	//       M^{i,j,alpha,beta}(x,x-mu) = < v^{i,alpha}(x) M v^{j,beta}(x-mu) > -- Needs  M v^{j,beta}(x-mu)
		//
		//  each M^{i,j,alpha,beta} is an (Naggr x Naggr)x(N_vec*N_vec) matrix
		//
		//  Code should look something like this:
		int N_aggr = aggregate.NumBlocks();
		int N_vec = v.size();

		LatticeFermionArray aggr_v(N_vec * N_aggr, fine_lattice_info);
		LatticeFermionArray Mv(N_vec * N_aggr, fine_lattice_info);  // N_vec fastest

		for(int j=0; j < N_vec; ++j) {
			for(int aggr=0; aggr < _aggregation.GetNumAggregates(); ++aggr) {

				// This call creates a source spinor with the source colors and spins from the aggregation
				// copied into the target and the rest set to zero.
				// NB: I combined the aggregation, and vector indices into a single index, and later
				// we can potentially do matrix multiplies of  (N_vec*N_aggr) x (N_vec*N_aggr) matrices

				aggr_v(j + N_vec*aggr) = SetTransferAggregationSpinColors( v(j), aggr, _aggregation ); // Other spin colors are 0

				// form  M v_j
				M_fine( Mv(j + N_vec*alpha), aggr_v(j,alpha), PLUS );                                   // Hit whole vector with M
		    }
		}

		// Forward and backward leaves
		for(int mu=0; mu < 4; ++mu) {

		       LatticeFermionArray v_x_plus_mu(N_vec  * N_aggr, fine_lattice_info) = block_shift(mu, FORWARD, v);
		       LatticeFermionArray v_x_minus_mu(N_vec * N_aggr, fine_lattice_info) = block_shift(mu, BACKWARD, v);

		       /* Vector Block Inner Product version */
		       for(int block=0; block < N_blocks; ++blocks) {

		    	   	   	   	   // NB: I want to recast the N_aggr x N_aggr x N_vec x N_vec inner products
		    	   	   	   	   //     as Matrix Multiply of (N_aggr N_vec) x V_block and V_block x (N_aggr x N_vec) matrices
		    	   	   	   	   // which can use a CGEMM/ZGEMM or can be rewritten by hand

	 						_coarse_op_fields.Index(block,mu)
									= BlockMatMult( BlockIndex(v_x_plus_mu,block), BlockIndex(Mv,block) );

							_coarse_op_fields(block,2*mu)
									= BlockMatMult( BlockIndex(v_x_minus_mu, block), BlockIndex(Mv,block) );

		       }
		}

		// Central Local piecee
		for(int block=0; block < N_blocks; ++blocks) {
			_coarse_op_fields.Index(block,8) =
					= BlockMatMult( BlockIndex(v, block), BlockIndex(Mv,block) );
		}

	}

private:
	const LatticeInfo& _fine_lattice_info;
	const Aggregation& _aggregation;
	const LatticeInfo& _coarse_lattice_info;
	const LatticeFermionArray& _basis_vectors);
	CoarseOpFields _coarse_op_fields;

}


// CoarseOperator::CoarseOperator(const LatticeInfo& fine_lattice_info,
//								  const Aggregation& aggregation,
//								  const LatticeInfo& coarse_lattice_info,
//								  LatticeFermionArray& basis_vectors)
// {
	// Ultimately the operator looks like
	//
	//   M^{i,j,alpha,beta}(x,y) \Phi^{j,beta}(y) =
	//       M^{i,j,alpha,beta}(x)       \Phi^{j,beta}(x)
	//     + M^{i,j,alpha,beta}(x + mu)  \Phi^{j,beta)(x+mu)
	//     + M^{i,j,alpha,beta}(x - mu)  \Phi^{j,beta}(x-mu)
	//
	//  With M^{i,j,alpha,beta}(x,x) =  < v^{i,alpha}(x) M v^{j,beta}(x) >  -- Local
	//       M^{i,j,alpha,beta}(x,x+mu) = < v^{i,alpha}(x) M v^{j,beta}(x+mu) > -- Needs  M v^{j,beta}(x+mu)  -- communicate block
 	//       M^{i,j,alpha,beta}(x,x-mu) = < v^{i,alpha}(x) M v^{j,beta}(x-mu) > -- Needs  M v^{j,beta}(x-mu)
	//
	//  each M^{i,j,alpha,beta} is an (Naggr x Naggr)x(N_vec*N_vec) matrix
	//
	//  Code should look something like this:

	//  int N_aggr = aggregate.NumBlocks();
	//  int N_vec = v.size();
	//
	//  LatticeFermionArray aggr_v(N_vec * N_aggr, fine_lattice_info);
	//	LatticeFermionArray Mv(N_vec * N_aggr, fine_lattice_info);  // N_vec fastest
	//
	//  for each vector j = 0..N-1 {
	//    for all destination aggregations alpha = 0..N_aggr-1 {
	//      aggr_v(j + Nvec*alpha) = SetTransferAggregationSpinColors( v(j), alpha, aggregation ); // Other spin colors are 0
	//      Mv(j + Nvec*alpha) = M aggr_v(j,alpha);                                   // Hit whole vector with M
    //   }
	//  }
	//
	//  CoarseOpFields D_c(coarse_lattice_info);
	//
	//



									block,dst_ablock_inner_product(v(block), Mv(block))
	// 		   for(src_aggr=0; src_aggr < n_aggr; ++src_aggr) {





}

void userfunc()
{
  LatticeInfo fine_geom({16,16,16,16}, 4, 3, NodeInfo());
  StandardAggregation blocking({4,4,4,4});
  unsigned int num_vec = 24;

  LatticeInfo coarse_geom = CoarsenLattice(fine_geom,blocking,num_vec);

  unsigned int N_array = 24;

  QDP::multi1d<LatticeFermion> basis_vectors(N_array); // Some vectors from QDP++


  LatticeFermionArray v(fine_geom, N_array);
  for(int i=0; i < N_array; ++i) {
	  Import::ImportFermionFromQDPXX( v(i), basis_vectors(N_array) ); // Import into this framework
  }

  // v holds the basis vectors imported from QDP++ somehow.

  // How would I create the coarse operator.
  // Basically I have  (M_coarse)(Lambda',Lambda, aggr_i, aggr_j, i, j) = v^{\Lambda'}_{a,i} M_fine v^{\Lambda'}_{b,j}
  //
  //   But M will only be non-zero if  i) Lambda'=Lambda or ii) Lambda' is a nearest neigbour of Lambda ie

  // So  v^\Lambda',(a,b)(i,j) M v^Lambda = \v^\Lambda (M v_Lambda) + \sum_mu  v^\Lambda-mu (M v_Lambda) + v^\Lambda+mu (M v_Lambda)
                  //             =         C_xx    + \sum_mu  C_{x-mu,x} + C_{x+mu,x}
                  //            =         C_xx    + \sum_mu  C_{xx,-mu} + C_{xx, +mu}

     // So the data type is:  [ coarse_x ][ dir=0..8 ][n_aggr][n_aggr][ N_vec ][ N_vec ][ Re/Im ]  = 7 indices

  	 // To make the coarse operator I need v's from

     //
  // So what I need to do is:
  //   a) Apply z_aggr_j = M_fine to v_b_j   for the whole lattice, ie all Lambda and all vectors
  //      To do this I need to set up v_b_j, so that only the spinor/color components in the Aggregate are non-zero
  //   b) Then for all destination
  for(unsigned int aggr_i = 0; aggr_i < n_aggregate; ++aggr_i) {

	  LatticeFermionArrag v_aggr(N_array); // I need N_array vectors

	  for(unsigned int i=0; i < N_array; ++i) {

              TransfAggregateSpinColor( v_aggr(i), v(i), blocking, aggr_i)
	  }
	  for(unsinged int aggr_j =0; aggr_j < n_aggregate; ++aggr_j) {

	  }
  }


}
