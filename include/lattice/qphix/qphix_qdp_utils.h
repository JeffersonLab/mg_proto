#pragma once
#ifndef QPHIX_QDP_UTILS_H
#define QPHIX_QDP_UTILS_H

#include "qdp.h"
#include "lattice/fine_qdpxx/clover_term_qdp_w.h"
#include "lattice/qphix/qphix_types.h"

namespace MG {


  void
  QDPSpinorToQPhiXSpinor( const QDP::LatticeFermion& qdp_in, QPhiXSpinor& qphix_out);

  void
  QPhiXSpinorToQDPSpinor( const QPhiXSpinor& qphix_in, QDP::LatticeFermion& qdp_out);

  void
  QDPGaugeFieldToQPhiXGauge( const QDP::multi1d<QDP::LatticeColorMatrix>& qdp_u, QPhiXGauge& qphix_out);


  void
  QDPCloverTermToQPhiXClover( const MG::QDPCloverTermT<QDP::LatticeFermion,
                                    QDP::LatticeColorMatrix>& clov,
                              const MG::QDPCloverTermT<QDP::LatticeFermion,
                                    QDP::LatticeColorMatrix>& invclov,
                              QPhiXClover& qphix_out);
}

#endif
