/*
 * qphix_qdp_utils.cpp
 *
 *  Created on: Oct 13, 2017
 *      Author: bjoo
 */


#include "lattice/qphix/qphix_types.h"
#include "lattice/qphix/qphix_qdp_utils.h"

#include <qphix/qdp_packer.h>

namespace MG {
void
 QDPSpinorToQPhiXSpinor( const QDP::LatticeFermion& qdp_in, QPhiXSpinor& qphix_out)
{
  for(int cb=0; cb < 2; ++cb) {
    QPhiX::qdp_pack_cb_spinor<>(qdp_in, qphix_out.getCB(cb).get(),MGQPhiX::GetGeom(),cb);
  }
}

 void
 QPhiXSpinorToQDPSpinor( const QPhiXSpinor& qphix_in, QDP::LatticeFermion& qdp_out)
 {
   for(int cb=0; cb < 2; ++cb) {
     QPhiX::qdp_unpack_cb_spinor<>(qphix_in.getCB(cb).get(),qdp_out, MGQPhiX::GetGeom(),cb);
   }
 }

 void
 QDPGaugeFieldToQPhiXGauge( const QDP::multi1d<QDP::LatticeColorMatrix>& qdp_u, QPhiXGauge& qphix_out)
 {
   QPhiX::qdp_pack_gauge<>(qdp_u, qphix_out.getCB(0).get(),qphix_out.getCB(1).get(),
         MGQPhiX::GetGeom());

 }

 void
 QDPCloverTermToQPhiXClover( const MG::QDPCloverTermT<QDP::LatticeFermion,
                                  QDP::LatticeColorMatrix>& clov,
                             const MG::QDPCloverTermT<QDP::LatticeFermion,
                                   QDP::LatticeColorMatrix>& invclov,
                             QPhiXClover& qphix_out)
 {
   // Use clover from D_full
   for(int cb=0; cb < 2; ++cb) {
      QPhiX::qdp_pack_clover<>(clov,qphix_out.getCB(cb).get(),MGQPhiX::GetGeom(),cb);
   }
   QPhiX::qdp_pack_clover<>(invclov,qphix_out.getInv().get(),MGQPhiX::GetGeom(),EVEN);

 }
}


