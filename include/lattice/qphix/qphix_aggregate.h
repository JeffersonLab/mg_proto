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

void axBlockAggr(const double alpha, QPhiXSpinor& v, const Block& block, int aggr);

void axBlockAggr(const double alpha, QPhiXSpinorF& v, const Block& block, int aggr);

void caxpyBlockAggr(const std::complex<double>& alpha, const QPhiXSpinorF& x,
    QPhiXSpinorF& y,  const Block& block, int aggr);

void caxpyBlockAggr(const std::complex<double>& alpha, const QPhiXSpinor& x,
    QPhiXSpinor& y,  const Block& block, int aggr);
double norm2BlockAggr(const QPhiXSpinor& v, const Block& block, int aggr);
double norm2BlockAggr(const QPhiXSpinorF& v, const Block& block, int aggr);

std::complex<double>
innerProductBlockAggr(const QPhiXSpinor& left, const QPhiXSpinor& right,
    const Block& block, int aggr);

std::complex<double>
innerProductBlockAggr(const QPhiXSpinorF& left, const QPhiXSpinorF& right,
    const Block& block, int aggr);

void extractAggregateBlock(QPhiXSpinor& target, const QPhiXSpinor& src,
    const Block& block, int aggr );

void extractAggregateBlock(QPhiXSpinorF& target, const QPhiXSpinorF& src,
    const Block& block, int aggr );

void extractAggregate(QPhiXSpinor& target, const QPhiXSpinor& src, int aggr );

void extractAggregate(QPhiXSpinorF& target, const QPhiXSpinorF& src, int aggr );

void orthonormalizeBlockAggregates(std::vector<std::shared_ptr<QPhiXSpinorF>>& vecs,
    const std::vector<Block>& block_list);

void orthonormalizeBlockAggregates(std::vector<std::shared_ptr<QPhiXSpinor>>& vecs,
    const std::vector<Block>& block_list);

void restrictSpinor( const std::vector<Block>& blocklist, const std::vector< std::shared_ptr<QPhiXSpinor> >& fine_vecs,
    const QPhiXSpinor& fine_in, CoarseSpinor& coarse_out);

void restrictSpinor( const std::vector<Block>& blocklist, const std::vector< std::shared_ptr<QPhiXSpinorF> >& fine_vecs,
    const QPhiXSpinorF& fine_in, CoarseSpinor& coarse_out);

void restrictSpinor2( const std::vector<Block>& blocklist, const std::vector< std::shared_ptr<QPhiXSpinor> >& fine_vecs,
    const QPhiXSpinor& fine_in, CoarseSpinor& coarse_out);
void restrictSpinor2( const std::vector<Block>& blocklist, const std::vector< std::shared_ptr<QPhiXSpinorF> >& fine_vecs,
    const QPhiXSpinorF& fine_in, CoarseSpinor& coarse_out);

void prolongateSpinor(const std::vector<Block>& blocklist,
    const std::vector<std::shared_ptr<QPhiXSpinor> >& fine_vecs,
    const CoarseSpinor& coarse_in, QPhiXSpinor& fine_out);

void prolongateSpinor(const std::vector<Block>& blocklist,
    const std::vector<std::shared_ptr<QPhiXSpinorF> >& fine_vecs,
    const CoarseSpinor& coarse_in, QPhiXSpinorF& fine_out);

void dslashTripleProductDir(const QPhiXWilsonCloverLinearOperator& D_op,
    const std::vector<Block>& blocklist, int dir,
    const std::vector<std::shared_ptr<QPhiXSpinor> >& in_vecs,
    CoarseGauge& u_coarse);

void dslashTripleProductDir(const QPhiXWilsonCloverLinearOperatorF& D_op,
    const std::vector<Block>& blocklist, int dir,
    const std::vector<std::shared_ptr<QPhiXSpinorF> >& in_vecs,
    CoarseGauge& u_coarse);

void clovTripleProduct(const QPhiXWilsonCloverLinearOperator& D_op,
    const std::vector<Block>& blocklist,
    const std::vector<std::shared_ptr<QPhiXSpinor> >& in_vecs,
    CoarseGauge& gauge_clover);

void clovTripleProduct(const QPhiXWilsonCloverLinearOperatorF& D_op,
    const std::vector<Block>& blocklist,
    const std::vector<std::shared_ptr<QPhiXSpinorF> >& in_vecs,
    CoarseGauge& gauge_clover);

}




#endif /* INCLUDE_LATTICE_QPHIX_QPHIX_AGGREGATE_H_ */
