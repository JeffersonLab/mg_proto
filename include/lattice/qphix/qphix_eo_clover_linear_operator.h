/*
 * qphix_clover_linear_operator.h
 *
 *  Created on: Oct 13, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_QPHIX_QPHIX_EO_CLOVER_LINEAR_OPERATOR_H_
#define INCLUDE_LATTICE_QPHIX_QPHIX_EO_CLOVER_LINEAR_OPERATOR_H_


/*!
 * Wrap a QDP++ linear operator, so as to provide the interface
 * with new types
 */
#include "lattice/linear_operator.h"
#include "lattice/fine_qdpxx/clover_fermact_params_w.h"
#include "lattice/fine_qdpxx/clover_term_qdp_w.h"
#include "lattice/fine_qdpxx/qdpxx_helpers.h"

#include "lattice/qphix/qphix_types.h"
#include "lattice/qphix/qphix_qdp_utils.h"
#include "lattice/qphix/qphix_blas_wrappers.h"
#include "lattice/coarse/block.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_l1_blas.h"
#include "lattice/coarse/aggregate_block_coarse.h"
#include <qphix/clover.h>
#include "lattice/coarse/subset.h"
#include "utils/auxiliary.h"

namespace MG {

template<typename FT>
class QPhiXWilsonCloverEOLinearOperatorT : public EOLinearOperator<QPhiXSpinorT<FT>,QPhiXGaugeT<FT> >,
                                           public AuxiliarySpinors<QPhiXSpinorT<FT>> {
public:
  using QDPGauge = QDP::multi1d<QDP::LatticeColorMatrix>;
  using Spinor = QPhiXSpinorT<FT>;
  using CBSpinor = QPhiXCBSpinorT<FT>;

  QPhiXWilsonCloverEOLinearOperatorT( const LatticeInfo& info, double m_q, double c_sw, int t_bc, const QDPGauge& gauge_in ) :
    _info(info), _t_bc(t_bc),_u(info), _clov(info)
  {
    // The QPhiX Operator takes unmodified fields.
     QDPGaugeFieldToQPhiXGauge(gauge_in, _u);

    {
      QDPGauge working_u(n_dim);
      for(int mu=0; mu < n_dim; ++mu) {
        working_u[mu] =gauge_in[mu];
      }
      // Apply boundary
      working_u[Nd-1] *= where(Layout::latticeCoordinate(Nd-1) == (Layout::lattSize()[Nd-1]-1),
                                    Integer(_t_bc), Integer(1));
      MG::CloverFermActParams _params;
      MG::QDPCloverTerm tmp_clov;
      MG::QDPCloverTerm tmp_invclov;

      _params.Mass =  Real(m_q);
      _params.clovCoeffR = Real(c_sw);
      _params.clovCoeffT = Real(c_sw);
      tmp_clov.create(working_u,_params);  // Make the clover term
      tmp_invclov.create(working_u,_params,tmp_clov);
      tmp_invclov.choles(EVEN);
      QDPCloverTermToQPhiXClover(tmp_clov,tmp_invclov,_clov);
    }

    IndexArray latdims = {{ QDP::Layout::subgridLattSize()[0],
                QDP::Layout::subgridLattSize()[1],
                QDP::Layout::subgridLattSize()[2],
                QDP::Layout::subgridLattSize()[3] }};


    double t_bcf = static_cast<double>(t_bc);
    qphix_gauge[0]=_u.getCB(0).get();
    qphix_gauge[1]=_u.getCB(1).get();
    qphix_clover[0] = _clov.getCB(0).get();
    qphix_clover[1] = _clov.getCB(1).get();

    QPhiXEOClov.reset(new QPhiXClovOpT<FT>(qphix_gauge,
                            qphix_clover,
                            _clov.getInv().get(),
                            &(MGQPhiX::GetGeom<FT>()),
                           t_bcf,1.0,1.0));
  }

  QPhiXWilsonCloverEOLinearOperatorT(const LatticeInfo& info, double m_q, double u0, double xi0, double nu, double c_sw_r, double c_sw_t,
      int t_bc, const QDPGauge& gauge_in ) : _info(info),_t_bc(t_bc),_u(info), _clov(info)
  {

    // The QPhiX Operator takes unmodified fields.
    QDPGaugeFieldToQPhiXGauge(gauge_in, _u);

    {
      QDPGauge working_u(n_dim);
      // Copy in the gauge field
      for(int mu=0; mu < Nd; ++mu) {
        working_u[mu] = gauge_in[mu];
      }

      // Apply boundary
      working_u[Nd-1] *= where(Layout::latticeCoordinate(Nd-1) == (Layout::lattSize()[Nd-1]-1),
                              Integer(_t_bc), Integer(1));

      // Set up and create the clover term
      // Use the unscaled links, because the scale factors will play
      // Into the clover calc.
      MG::CloverFermActParams _params;
      MG::QDPCloverTerm tmp_clov;
      MG::QDPCloverTerm tmp_invclov;

      _params.Mass =  Real(m_q);
      _params.clovCoeffR = Real(c_sw_r);
      _params.clovCoeffT = Real(c_sw_t);
      _params.u0 = Real(u0);
      _params.anisoParam.anisoP = true;
      _params.anisoParam.t_dir = 3;
      _params.anisoParam.xi_0 = Real(xi0);
      _params.anisoParam.nu = Real(nu);

      tmp_clov.create(working_u,_params);  // Make the clover term
      tmp_invclov.create(working_u,_params,tmp_clov);
      tmp_invclov.choles(EVEN);
      QDPCloverTermToQPhiXClover(tmp_clov,tmp_invclov,_clov);
    }

    // Now scale the links for use in the dslash
    // By the anisotropy.
    Real aniso_scale_fac = Real( nu/xi0 );

    IndexArray latdims = {{ QDP::Layout::subgridLattSize()[0],
                QDP::Layout::subgridLattSize()[1],
                QDP::Layout::subgridLattSize()[2],
                QDP::Layout::subgridLattSize()[3] }};

    double t_bcf = static_cast<double>(t_bc);
    double anisoFacS=nu/xi0;
    double anisoFacT=1.0;
    qphix_gauge[0]=_u.getCB(0).get();
    qphix_gauge[1]=_u.getCB(1).get();
    qphix_clover[0] = _clov.getCB(0).get();
    qphix_clover[1] = _clov.getCB(1).get();
    QPhiXEOClov.reset(new QPhiXClovOpT<FT>(qphix_gauge,
                            qphix_clover,
                           _clov.getInv().get(),
                            &(MGQPhiX::GetGeom<FT>()),
                            t_bcf,anisoFacS,anisoFacT));
  }

  ~QPhiXWilsonCloverEOLinearOperatorT() {

    // The QPhiX op will delete when the smart pointer calls its destructor
  }

  // The predoncidioned operator
  void operator()(Spinor& out, const Spinor& in, IndexType type = LINOP_OP) const override{
    int isign = (type == LINOP_OP) ? 1 : -1;
    assert(out.GetNCol() == in.GetNCol());
    IndexType ncol = out.GetNCol();
    for (int col=0; col < ncol; ++col)
      QPhiXEOClov->operator()(out.getCB(col,ODD).get(),in.getCB(col,ODD).get(),isign,ODD);
  }

  // The unpreconditioned operator
  void unprecOp(Spinor& out, const Spinor& in, IndexType type = LINOP_OP) const override {
    int isign = (type == LINOP_OP) ? 1 : -1;
    assert(out.GetNCol() == in.GetNCol());
    IndexType ncol = out.GetNCol();
    for (int col=0; col < ncol; ++col)
      QPhiXEOClov->M_unprec(out.get(col),in.get(col),isign);
  }


  void leftOp(Spinor& out, const Spinor& in) const override{

    // tmp1 = M^{-1}_ee in[even] -- use out[even] as temporary
    const int isign = 1;
    assert(out.GetNCol() == in.GetNCol());
    std::shared_ptr<Spinor> tmp = this->tmp(in); 
    IndexType ncol = out.GetNCol();
    for (int col=0; col < ncol; ++col) {
      QPhiXEOClov->M_diag_inv(tmp->getCB(col,EVEN).get(), in.getCB(col,EVEN).get(), isign);

      // tmp2 = M_oe M tmp1 = M_oe M_ee^{-1} in
      QPhiXEOClov->M_offdiag(tmp->getCB(col,ODD).get() ,tmp->getCB(col,EVEN).get(), isign, ODD);
    }
    CopyVec(out, in,  SUBSET_ALL);
    YpeqXVec(*tmp, out,SUBSET_ODD);

  }

  void leftInvOp(Spinor& out, const Spinor& in) const override {
    // L^{-1}[ in_e ] = [ 1       0 ][ in_e ] = [ in_e                       ]
    //       [ in_o ]   [-DA^{-1} 1 ][ in_o ]   [ in_o - D_oe A^{-1}_ee in_e ]

    // tmp1 = M^{-1}_ee in[even] -- use out[even] as temporary
    const int isign = 1;
    assert(out.GetNCol() == in.GetNCol());
    IndexType ncol = out.GetNCol();
    std::shared_ptr<Spinor> tmp = this->tmp(in); 
    for (int col=0; col < ncol; ++col){
      QPhiXEOClov->M_diag_inv(tmp->getCB(col,EVEN).get(), in.getCB(col,EVEN).get(), isign);

      // tmp2 = M_oe M tmp1 = M_oe M_ee^{-1} in
      QPhiXEOClov->M_offdiag(tmp->getCB(col,ODD).get(), tmp->getCB(col,EVEN).get(), isign, ODD);
    }

    CopyVec(out, in, SUBSET_ALL);
    YmeqXVec(*tmp, out,SUBSET_ODD);

  }

#if 0
  // Special case when in even is known to be zero.
  void leftInvOpZero(Spinor& out,const Spinor& in) const override {
    // L^{-1}[ in_e ] = [ 1       0 ][ in_e ] = [ in_e                       ]
    //       [ in_o ]   [-DA^{-1} 1 ][ in_o ]   [ in_o - D_oe A^{-1}_ee in_e ]
      // when in_e = 0 we have
    //
    // L^{-1} [ in_e ] = [ in_e     ] = [in_e]
    //        [ in_o ]   [ in_o - 0 ]   [in_o]

      CopyVec(out,in,SUBSET_ALL);   // This automatically zeros out's even subset

    }
#endif

  void rightOp(Spinor& out, const Spinor& in) const override {

    const int isign = 1;
    assert(out.GetNCol() == in.GetNCol());
    IndexType ncol = out.GetNCol();
    std::shared_ptr<Spinor> tmp = this->tmp(in); 

    // use odd cb of tmp1 as a temporary storage
    // it is not really the odd checkerboardl
    for (int col=0; col < ncol; ++col) {
      QPhiXEOClov->M_offdiag(tmp->getCB(col,ODD).get(), in.getCB(col,ODD).get(), isign, EVEN);

      // we will go from the odd into the final target.
      QPhiXEOClov->M_diag_inv(tmp->getCB(col,EVEN).get(), tmp->getCB(col,ODD).get(), isign);
    }

    CopyVec(out, in, SUBSET_ALL);
    YpeqXVec(*tmp, out,SUBSET_EVEN);

  }

  void rightInvOp(Spinor& out, const Spinor& in) const override  {

    // tmp1 = M^{-1}_ee in[even] -- use out[even] as temporary
    const int isign = 1;
    assert(out.GetNCol() == in.GetNCol());
    IndexType ncol = out.GetNCol();
    std::shared_ptr<Spinor> tmp = this->tmp(in); 


    for (int col=0; col < ncol; ++col){
      QPhiXEOClov->M_offdiag(tmp->getCB(col,ODD).get(), in.getCB(col,ODD).get(), isign, EVEN);
      QPhiXEOClov->M_diag_inv(tmp->getCB(col,EVEN).get(), tmp->getCB(col,ODD).get(), isign);
    }
   CopyVec(out, in, SUBSET_ALL);
   YmeqXVec(*tmp, out,SUBSET_EVEN);


   }

  void M_ee_inv(Spinor& out, const Spinor& in, IndexType type=LINOP_OP) const override{
      const int isign = (type == LINOP_OP) ? 1: -1;
    assert(out.GetNCol() == in.GetNCol());
    IndexType ncol = out.GetNCol();
    for (int col=0; col < ncol; ++col)
      QPhiXEOClov->M_diag_inv(out.getCB(col,EVEN).get(),in.getCB(col,EVEN).get(),isign);
    }

  void M_diag(Spinor& out, const Spinor& in, int cb) const override {
    assert(out.GetNCol() == in.GetNCol());
    IndexType ncol = out.GetNCol();
    for (int col=0; col < ncol; ++col)
      QPhiXEOClov->M_diag(out.get(col),in.get(col),1,cb);
  }

  void M_oo(Spinor& out, const Spinor& in, IndexType type=LINOP_OP) const  {
     int isign = (type == LINOP_OP) ? 1 : -1;
    assert(out.GetNCol() == in.GetNCol());
    IndexType ncol = out.GetNCol();
    for (int col=0; col < ncol; ++col)
      QPhiXEOClov->M_diag(out.get(col),in.get(col),isign,1);
   }

  void M_ee(Spinor& out, const Spinor& in, IndexType type=LINOP_OP) const{
      int isign = (type == LINOP_OP) ? 1 : -1;
    assert(out.GetNCol() == in.GetNCol());
    IndexType ncol = out.GetNCol();
    for (int col=0; col < ncol; ++col)
      QPhiXEOClov->M_diag(out.get(col),in.get(col),isign,0);
    }
  int GetLevel(void) const override {
    return 0;
  }

  const CBSubset& GetSubset() const override {
    return SUBSET_ODD;
  }

  const LatticeInfo& GetInfo(void) const override {
    return _info;
  }

  // Apply a single direction of Dslash -- used for coarsening
  // Coarsening is done as per the unprec op.
  void DslashDir(Spinor& spinor_out,
        const Spinor& spinor_in,
        const IndexType dir) const {
    assert(spinor_out.GetNCol() == spinor_in.GetNCol());
    IndexType ncol = spinor_out.GetNCol();
    for (int col=0; col < ncol; ++col)
      QPhiXEOClov->DslashDir(spinor_out.get(col),spinor_in.get(col),dir);
  }

  QPhiXClovOpT<FT>& getQPhiXOp()
  {
    return *QPhiXEOClov;
  }

  // Generate Coarse is done as per the coarse op
  void generateCoarse(const std::vector<Block>& blocklist,
      const std::vector<std::shared_ptr<Spinor>>& in_vecs,
      CoarseGauge& u_coarse) const
  {

    const LatticeInfo& info = u_coarse.GetInfo();
    int num_colorspin = info.GetNumColorSpins();

    // Generate the triple products directly into the u_coarse
    ZeroGauge(u_coarse);
    for(int mu=0; mu < 8; ++mu) {
        MasterLog(INFO,"QPhiXWilsonCloverEOLinearOperator: Dslash Triple Product in direction: %d ",mu);
      dslashTripleProductDir(*this, blocklist, mu, in_vecs, u_coarse);
    }


    for(int cb=0; cb < n_checkerboard; ++cb) {
        for(int cbsite=0; cbsite < u_coarse.GetInfo().GetNumCBSites(); ++cbsite) {
          for(int mu=0; mu < 8; ++mu) {
            float* link=u_coarse.GetSiteDirDataPtr(cb,cbsite,mu);
            for(int j=0; j < n_complex*num_colorspin*num_colorspin; ++j) {
              link[j] *= -0.5;
            }
          }
          float* diag_link=u_coarse.GetSiteDiagDataPtr(cb,cbsite);
          for(int j=0; j < n_complex*num_colorspin*num_colorspin; ++j) {
            diag_link[j] *= -0.5;
          }
        }
      }


    MasterLog(INFO,"QPhiXWilsonCloverEOLinearOperator: Clover Triple Product");
    clovTripleProduct(*this,blocklist,  in_vecs, u_coarse);

    MasterLog(INFO,"QPhiXWilsonCloverEOLinearOperator: Inverting Diagonal (A) Links");
  invertCloverDiag(u_coarse);

  MasterLog(INFO,"QPhiXWilsonCloverEOLinearOperator: Computing A^{-1} D Links");
  multInvClovOffDiagLeft(u_coarse);

  MasterLog(INFO, "QPhiXWilsonCloverEOLinearOperator: Computing D A^{-1} Links");
  multInvClovOffDiagRight(u_coarse);

  }



private:

  const int _t_bc;
  QPhiXGaugeT<FT> _u;
  QPhiXCloverT<FT> _clov;
  // This will be hidden
  typename QPhiXGeomT<FT>::SU3MatrixBlock *qphix_gauge[2] ;
  typename QPhiXGeomT<FT>::CloverBlock *qphix_clover[2];
  const LatticeInfo& _info;

  std::unique_ptr<QPhiXClovOpT<FT>> QPhiXEOClov;

};

using QPhiXWilsonCloverEOLinearOperator = QPhiXWilsonCloverEOLinearOperatorT<double>;
using QPhiXWilsonCloverEOLinearOperatorF = QPhiXWilsonCloverEOLinearOperatorT<float>;

}



#endif /* INCLUDE_LATTICE_QPHIX_QPHIX_CLOVER_LINEAR_OPERATOR_H_ */
