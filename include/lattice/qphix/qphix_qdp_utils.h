#pragma once
#ifndef QPHIX_QDP_UTILS_H
#define QPHIX_QDP_UTILS_H

#include "qdp.h"
#include "lattice/fine_qdpxx/clover_term_qdp_w.h"
#include "lattice/qphix/qphix_types.h"
#include "lattice/coarse/subset.h"
namespace MG {


  void
  QDPSpinorToQPhiXSpinor( const QDP::LatticeFermion& qdp_in, QPhiXSpinor& qphix_out, IndexType col, const CBSubset& subset = SUBSET_ALL);

  void
   QDPSpinorToQPhiXSpinor( const QDP::LatticeFermion& qdp_in, QPhiXSpinorF& qphix_out, IndexType col, const CBSubset& subset = SUBSET_ALL);

  void
  QPhiXSpinorToQDPSpinor( const QPhiXSpinor& qphix_in, IndexType col, QDP::LatticeFermion& qdp_out, const CBSubset& subset = SUBSET_ALL);

  void
    QPhiXSpinorToQDPSpinor( const QPhiXSpinorF& qphix_in, IndexType col, QDP::LatticeFermion& qdp_out, const CBSubset& subset = SUBSET_ALL);

  void
  QDPGaugeFieldToQPhiXGauge( const QDP::multi1d<QDP::LatticeColorMatrix>& qdp_u, QPhiXGauge& qphix_out);


  void
  QDPCloverTermToQPhiXClover( const MG::QDPCloverTermT<QDP::LatticeFermion,
                                    QDP::LatticeColorMatrix>& clov,
                              const MG::QDPCloverTermT<QDP::LatticeFermion,
                                    QDP::LatticeColorMatrix>& invclov,
                              QPhiXClover& qphix_out);

#if 0
  void
  QDPSpinorToQPhiXSpinor( const QDP::LatticeFermion& qdp_in, QPhiXSpinorF& qphix_out);

  void
  QPhiXSpinorToQDPSpinor( const QPhiXSpinorF& qphix_in, QDP::LatticeFermion& qdp_out);

#endif

  void
  QDPGaugeFieldToQPhiXGauge( const QDP::multi1d<QDP::LatticeColorMatrix>& qdp_u, QPhiXGaugeF& qphix_out);


  void
  QDPCloverTermToQPhiXClover( const MG::QDPCloverTermT<QDP::LatticeFermion,
                                    QDP::LatticeColorMatrix>& clov,
                              const MG::QDPCloverTermT<QDP::LatticeFermion,
                                    QDP::LatticeColorMatrix>& invclov,
                              QPhiXCloverF& qphix_out);
}

#endif
