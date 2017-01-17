/*
 * aggregate_qdpxx.h
 *
 *  Created on: Dec 9, 2016
 *      Author: bjoo
 */

#ifndef TEST_QDPXX_AGGREGATE_BLOCK_QDPXX_H_
#define TEST_QDPXX_AGGREGATE_BLOCK_QDPXX_H_

#include "qdp.h"
#include "clover_term_qdp_w.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_op.h"
#include "lattice/coarse/block.h"
#include <vector>


using namespace MG;
using namespace MGTesting;
using namespace QDP;


namespace MGTesting {

//! v *= alpha (alpha is real) over and aggregate in a block, v is a QDP++ Lattice Fermion
void axBlockAggrQDPXX(const double alpha, LatticeFermion& v, const Block& block, int aggr);

//! y += alpha * x (alpha is complex) over aggregate in a block, x, y are QDP++ LatticeFermions;
void caxpyBlockAggrQDPXX(const std::complex<double>& alpha, const LatticeFermion& x, LatticeFermion& y,  const Block& block, int aggr);

//! return || v ||^2 over an aggregate in a block, v is a QDP++ LatticeFermion
double norm2BlockAggrQDPXX(const LatticeFermion& v, const Block& block, int aggr);

//! return < left | right > = sum left^\dagger_i * right_i for an aggregate, over a block
std::complex<double>
innerProductBlockAggrQDPXX(const LatticeFermion& left, const LatticeFermion& right, const Block& block, int aggr);

//! Extract the spins belonging to a given aggregate from QDP++ source vector src, into QDP++ target vector target
void extractAggregateQDPXX(LatticeFermion& target, const LatticeFermion& src, const Block& block, int aggr );

//! Orthonormalize vecs over the spin aggregates within the sites
void orthonormalizeBlockAggregatesQDPXX(multi1d<LatticeFermion>& vecs, const std::vector<Block>& block_list);


//! 'Restrict' a QDP++ spinor to a CoarseSpinor with the same geometry
void restrictSpinorQDPXXFineToCoarse( const std::vector<Block>& blocklist, const multi1d<LatticeFermion>& v, const LatticeFermion& ferm_in, CoarseSpinor& out);

//! 'Prolongate' a CoarseSpinor to a QDP++ Fine Spinor
void prolongateSpinorCoarseToQDPXXFine(const std::vector<Block>& blocklist, const multi1d<LatticeFermion>& v, const CoarseSpinor& coarse_in, LatticeFermion& fine_out);

//! Coarsen one direction of a 'dslash' link
void dslashTripleProductDirQDPXX(const std::vector<Block>& blocklist, int dir, const multi1d<LatticeColorMatrix>& u, const multi1d<LatticeFermion>& in_vecs, CoarseGauge& u_coarse);

//! Coarsen the clover term (1 block = 1 site )
void clovTripleProductQDPXX(const std::vector<Block>& blocklist, const QDPCloverTerm& clov,const multi1d<LatticeFermion>& in_vecs, CoarseClover& cl_coarse);

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
void extractAggregate(CoarseSpinor& target, const CoarseSpinor& src, const Block& block, int aggr );

// ---
//! Orthonormalize vecs over the spin aggregates within the sites
//  FIXME: If I want to have a std::vector of a type which has a constructor I need to hold them by pointer.
//      This is because std::vector<> and multi1d need things which have default constructors. However, objects
// 		such as CoarseSpinor need to take a reference to a LatticeInfo in construction. Is there a best practice/pattern
// 		for creating containers of such objects.
void orthonormalizeBlockAggregates(std::vector<CoarseSpinor*>& vecs, const std::vector<Block>& block_list);


//! 'Restrict' a QDP++ spinor to a CoarseSpinor with the same geometry
void restrictSpinor( const std::vector<Block>& blocklist, const std::vector<CoarseSpinor*>& v, const CoarseSpinor& ferm_in, CoarseSpinor& out);

//! 'Prolongate' a CoarseSpinor to a QDP++ Fine Spinor
void prolongateSpinor(const std::vector<Block>& blocklist, const std::vector<CoarseSpinor*>& v, const CoarseSpinor& coarse_in, CoarseSpinor& fine_out);

//! Coarsen one direction of a dslash link: FIXME: may become a method of CoarseDiracOp later?
void dslashTripleProductDir(const CoarseDiracOp& D_op, const std::vector<Block>& blocklist, int dir, const CoarseGauge& u_fine, const std::vector<CoarseSpinor*>& in_vecs, CoarseGauge& u_coarse);

//! Coarsen the clover term (1 block = 1 site ): FIXME: may becme a method of CoarseDiracOp later?
void clovTripleProduct(const CoarseDiracOp& D_op, const std::vector<Block>& blocklist, const CoarseClover& cl_fine, const std::vector<CoarseSpinor*>& in_vecs, CoarseClover& cl_coarse);
};

#endif /* TEST_QDPXX_AGGREGATE_QDPXX_H_ */
