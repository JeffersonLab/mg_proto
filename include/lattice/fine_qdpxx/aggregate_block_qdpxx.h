/*
 * aggregate_qdpxx.h
 *
 *  Created on: Dec 9, 2016
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_FINE_QDPXX_AGGREGATE_BLOCK_QDPXX_H_
#define INCLUDE_LATTICE_FINE_QDPXX_AGGREGATE_BLOCK_QDPXX_H_

#include "qdp.h"
#include "lattice/fine_qdpxx/clover_term_qdp_w.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_op.h"
#include "lattice/coarse/block.h"
#include <vector>


using namespace QDP;


namespace MG {

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
void clovTripleProductQDPXX(const std::vector<Block>& blocklist, const QDPCloverTerm& clov,const multi1d<LatticeFermion>& in_vecs, CoarseGauge& cl_coarse);

}

#endif /* TEST_QDPXX_AGGREGATE_QDPXX_H_ */
