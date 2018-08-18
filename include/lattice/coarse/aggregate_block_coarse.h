/*
 * aggregate_qdpxx.h
 *
 *  Created on: Dec 9, 2016
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_COARSE_AGGREGATE_BLOCK_COARSE_H_
#define INCLUDE_LATTICE_COARSE_AGGREGATE_BLOCK_COARSE_H_

#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_op.h"
#include "lattice/coarse/block.h"
#include <vector>
#include <memory>



namespace MG {


/* --------- COARSE COARSE BLOCK STUFF COming here ------------- */
//! v *= alpha (alpha is real) over and aggregate in a block, v is a CoarseSpinor
void axBlockAggr(const double alpha, CoarseSpinor& v, const Block& block, int aggr);

//! y += alpha * x (alpha is complex) over aggregate in a block, x, y are QDP++ LatticeFermions;
void caxpyBlockAggr(const std::complex<double>& alpha, const CoarseSpinor& x, CoarseSpinor& y,  const Block& block, int aggr);

//! return || v ||^2 over an aggregate in a block, v is a CoarseSpinor
double norm2BlockAggr(const CoarseSpinor& v, const Block& block, int aggr);

//! return < left | right > = sum left^\dagger_i * right_i for an aggregate, over a block
std::complex<double>
innerProductBlockAggr(const CoarseSpinor& left, const CoarseSpinor& right, const Block& block, int aggr);

//! Extract the spins belonging to a given aggregate from QDP++ source vector src, into QDP++ target vector target
void extractAggregateBlock(CoarseSpinor& target, const CoarseSpinor& src, const Block& block, int aggr );

//! ExtractAggregate
//! Extract the spins belonging to a given aggregate from QDP++ source vector src, into QDP++ target vector target
void extractAggregate(CoarseSpinor& target, const CoarseSpinor& src, int aggr );
// ---
//! Orthonormalize vecs over the spin aggregates within the sites
//  FIXME: If I want to have a std::vector of a type which has a constructor I need to hold them by pointer.
//      This is because std::vector<> and multi1d need things which have default constructors. However, objects
// 		such as CoarseSpinor need to take a reference to a LatticeInfo in construction. Is there a best practice/pattern
// 		for creating containers of such objects.
void orthonormalizeBlockAggregates(std::vector<std::shared_ptr<CoarseSpinor> >& vecs, const std::vector<Block>& block_list);


//! 'Restrict' a QDP++ spinor to a CoarseSpinor with the same geometry
void restrictSpinor( const std::vector<Block>& blocklist, const std::vector<std::shared_ptr<CoarseSpinor > >& v, const CoarseSpinor& ferm_in, CoarseSpinor& out);

//! 'Prolongate' a CoarseSpinor to a QDP++ Fine Spinor
void prolongateSpinor(const std::vector<Block>& blocklist, const std::vector<std::shared_ptr<CoarseSpinor> >& v, const CoarseSpinor& coarse_in, CoarseSpinor& fine_out);

//! Coarsen one direction of a dslash link: FIXME: may become a method of CoarseDiracOp later?
void dslashTripleProductDir(const CoarseDiracOp& D_op, const std::vector<Block>& blocklist, int dir, const CoarseGauge& u_fine,
			const std::vector<std::shared_ptr<CoarseSpinor > >& in_vecs, CoarseGauge& u_coarse);

//! Coarsen the clover term (1 block = 1 site ): FIXME: may becme a method of CoarseDiracOp later?
void clovTripleProduct(const CoarseDiracOp& D_op,
			const std::vector<Block>& blocklist,
			const CoarseGauge& fine_clov,
			const std::vector<std::shared_ptr<CoarseSpinor > >& in_fine_vecs,
			CoarseGauge& coarse_clov);

// Invert the diagonal part of u, into eo_clov
void invertCloverDiag(CoarseGauge& u);

// Multiply the inverse part of the clover into eo_clov
void multInvClovOffDiagLeft(CoarseGauge& u);


void multInvClovOffDiagRight(CoarseGauge& u);

};

#endif /* TEST_QDPXX_AGGREGATE_QDPXX_H_ */
