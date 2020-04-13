/*
 * qphix_qdp_utils.cpp
 *
 *  Created on: Oct 13, 2017
 *      Author: bjoo
 */


#include "lattice/qphix/qphix_types.h"
#include "lattice/qphix/qphix_qdp_utils.h"

#include <qphix/qdp_packer.h>
#include <lattice/coarse/subset.h>

namespace MG {
template<typename QDPT,typename QPhiXT>
void
 QDPSpinorToQPhiXSpinorT( const QDPT& qdp_in, QPhiXT& qphix_out, IndexType col, const CBSubset& subset)
{
  for(int cb=subset.start; cb < subset.end; ++cb) {
    // Loose const-ness on Geom due to QPhiX interface...
    QPhiX::qdp_pack_cb_spinor<>(qdp_in, qphix_out.getCB(col, cb).get(),qphix_out.getGeom(),cb);
  }
}

void
QDPSpinorToQPhiXSpinor(const QDP::LatticeFermion& qdp_in, QPhiXSpinor& qphix_out, IndexType col, const CBSubset& subset)
{
  QDPSpinorToQPhiXSpinorT(qdp_in,qphix_out,col,subset);
}

void
QDPSpinorToQPhiXSpinor(const LatticeFermion& qdp_in, QPhiXSpinorF& qphix_out, IndexType col, const CBSubset& subset)
{
  QDPSpinorToQPhiXSpinorT(qdp_in,qphix_out, col, subset);
}

template<typename QDPT, typename QPhiXT>
 void
 QPhiXSpinorToQDPSpinorT( const QPhiXT& qphix_in, IndexType col, QDPT& qdp_out, const CBSubset& subset)
 {
   for(int cb=subset.start; cb < subset.end; ++cb) {
     // Loose const-ness on Geom due to QPhiX interface...
     QPhiX::qdp_unpack_cb_spinor<>(qphix_in.getCB(col, cb).get(),qdp_out, qphix_in.getGeom(),cb);
   }
 }


void QPhiXSpinorToQDPSpinor( const QPhiXSpinor& qphix_in, IndexType col, QDP::LatticeFermion& qdp_out,const CBSubset& subset ) {
  QPhiXSpinorToQDPSpinorT( qphix_in, col, qdp_out, subset);
}

void QPhiXSpinorToQDPSpinor( const QPhiXSpinorF& qphix_in, IndexType col,  QDP::LatticeFermion& qdp_out, const CBSubset& subset) {
  QPhiXSpinorToQDPSpinorT( qphix_in,col, qdp_out,subset);
}



template<typename QDPT, typename QPhiXT>
 void
 QDPGaugeFieldToQPhiXGaugeT( const QDPT& qdp_u, QPhiXT& qphix_out)
 {
   // Loose const-ness on Geom due to QPhiX interface...
   QPhiX::qdp_pack_gauge<>(qdp_u, qphix_out.getCB(0).get(),qphix_out.getCB(1).get(),
       qphix_out.getGeom());

 }
 void
  QDPGaugeFieldToQPhiXGauge( const QDP::multi1d<QDP::LatticeColorMatrix>& qdp_u, QPhiXGauge& qphix_out)
 {
   QDPGaugeFieldToQPhiXGaugeT( qdp_u, qphix_out);

 }
 void
  QDPGaugeFieldToQPhiXGauge( const QDP::multi1d<QDP::LatticeColorMatrix>& qdp_u, QPhiXGaugeF& qphix_out)
 {
   QDPGaugeFieldToQPhiXGaugeT( qdp_u, qphix_out);
 }




  template<typename QDPT, typename QPhiXT>
  void
  QDPCloverTermToQPhiXCloverT( const QDPT& clov,
                              const QDPT& invclov,
                              QPhiXT& qphix_out)
  {
    // Use clover from D_full
    for(int cb=0; cb < 2; ++cb) {
      // Loose const-ness on Geom due to QPhiX interface...
       QPhiX::qdp_pack_clover<>(clov,qphix_out.getCB(cb).get(),qphix_out.getGeom(),cb);
    }
    // Loose const-ness on Geom due to QPhiX interface...
    QPhiX::qdp_pack_clover<>(invclov,qphix_out.getInv().get(),qphix_out.getGeom(),EVEN);

  }


void
  QDPCloverTermToQPhiXClover( const MG::QDPCloverTermT<QDP::LatticeFermion,
                                   QDP::LatticeColorMatrix>& clov,
                              const MG::QDPCloverTermT<QDP::LatticeFermion,
                                    QDP::LatticeColorMatrix>& invclov,
                              QPhiXClover& qphix_out)
{
  QDPCloverTermToQPhiXCloverT( clov, invclov, qphix_out);
}


void
  QDPCloverTermToQPhiXClover( const MG::QDPCloverTermT<QDP::LatticeFermion,
                                   QDP::LatticeColorMatrix>& clov,
                              const MG::QDPCloverTermT<QDP::LatticeFermion,
                                    QDP::LatticeColorMatrix>& invclov,
                              QPhiXCloverF& qphix_out)
{
  QDPCloverTermToQPhiXCloverT( clov, invclov, qphix_out);
}
} // namespace
