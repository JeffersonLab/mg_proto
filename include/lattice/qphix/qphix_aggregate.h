/*
 * qphix_aggregate.h
 *
 *  Created on: Oct 19, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_QPHIX_QPHIX_AGGREGATE_H_
#define INCLUDE_LATTICE_QPHIX_QPHIX_AGGREGATE_H_

#include "lattice/qphix/qphix_types.h"
#include "lattice/coarse/block.h"
#include "lattice/qphix/qphix_clover_linear_operator.h"
namespace MG
{
//! Orthonormalize vecs over the spin aggregates within the sites
//  FIXME: If I want to have a std::vector of a type which has a constructor I need to hold them by pointer.
//      This is because std::vector<> and multi1d need things which have default constructors. However, objects
//    such as CoarseSpinor need to take a reference to a LatticeInfo in construction. Is there a best practice/pattern
//    for creating containers of such objects.
//  NB: Keep the vectors in single precision
void orthonormalizeBlockAggregates(std::vector<std::shared_ptr<QPhiXSpinorF> >& vecs, const std::vector<Block>& block_list);


//! 'Restrict' QPPhiX spinor to a CoarseSpinor with the same geometry
//   Restriction and prolongation are part of the preconditioner so work on Single prec fields.
void restrictSpinor( const std::vector<Block>& blocklist, const std::vector<std::shared_ptr<QPhiXSpinorF > >& v, const QPhiXSpinorF& ferm_in, CoarseSpinor& out);

//! 'Prolongate' a CoarseSpinor to a QPhiX Fine Spinor
void prolongateSpinor(const std::vector<Block>& blocklist, const std::vector<std::shared_ptr<QPhiXSpinorF> >& v, const CoarseSpinor& coarse_in, QPhiXSpinorF& fine_out);

//! Coarsen one direction of a dslash link: FIXME: may become a method of CoarseDiracOp later?
void dslashTripleProductDir(const QPhiXWilsonCloverLinearOperatorF& D_op, const std::vector<Block>& blocklist, int dir,
      const std::vector<std::shared_ptr<QPhiXSpinorF > >& in_vecs, CoarseGauge& u_coarse);

//! Coarsen the clover term (1 block = 1 site ): FIXME: may becme a method of CoarseDiracOp later?
void clovTripleProduct(const QPhiXWilsonCloverLinearOperatorF& D_op,
      const std::vector<Block>& blocklist,
      const std::vector<std::shared_ptr<QPhiXSpinorF > >& in_fine_vecs,
      CoarseGauge& coarse_clov);
};


#endif /* INCLUDE_LATTICE_QPHIX_QPHIX_AGGREGATE_H_ */
