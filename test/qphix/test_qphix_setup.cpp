/*
 * test_qphix_setup.cpp
 *
 *  Created on: Oct 20, 2017
 *      Author: bjoo
 */
#include <gtest/gtest.h>
#include <lattice/fine_qdpxx/invfgmres_qdpxx.h>

#include "../test_env.h"
#include "lattice/qphix/qphix_types.h"
#include "lattice/qphix/qphix_qdp_utils.h"
#include "./qphix_testutils.h"

#include "lattice/constants.h"
#include "../qdpxx/qdpxx_latticeinit.h"
#include "../qdpxx/qdpxx_utils.h"
#include "lattice/fine_qdpxx/qdpxx_helpers.h"

#include "lattice/fine_qdpxx/clover_term_qdp_w.h"
#include "lattice/fine_qdpxx/dslashm_w.h"
#include "lattice/fine_qdpxx/wilson_clover_linear_operator.h"

#include "lattice/qphix/qphix_clover_linear_operator.h"
#include "lattice/qphix/invfgmres_qphix.h"
#include "lattice/qphix/invbicgstab_qphix.h"
#include "lattice/qphix/invmr_qphix.h"
#include "lattice/mr_params.h"
#include "lattice/fine_qdpxx/invmr_qdpxx.h"

#include "lattice/fine_qdpxx/mg_params_qdpxx.h"
#include "lattice/qphix/mg_level_qphix.h"
#include "lattice/fine_qdpxx/mg_level_qdpxx.h"
#include "lattice/qphix/qphix_aggregate.h"
#include "lattice/fine_qdpxx/aggregate_block_qdpxx.h"
#include "lattice/fine_qdpxx/aggregate_qdpxx.h"

#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_l1_blas.h"

#include <cmath>
using namespace QDP;
using namespace MG;
using namespace MGTesting;

TEST(QPhiXIntegration, TestSetupPieces)
{
  // Init the lattice
  IndexArray latdims={{8,8,8,8}};
  initQDPXXLattice(latdims);
  multi1d<LatticeColorMatrix> u(Nd);
  for(int mu=0; mu < Nd; ++mu) {

    gaussian(u[mu]);
    reunit(u[mu]);
  }

  LatticeInfo info(latdims);

  float m_q = 0.01;
  float c_sw = 1.25;

  int t_bc=-1; // Antiperiodic t BCs

  QDPIO::cout << "Creating M" << std::endl;

  std::shared_ptr<QPhiXWilsonCloverLinearOperatorF> M=
      std::make_shared<QPhiXWilsonCloverLinearOperatorF>(info,m_q, c_sw, t_bc,u);

  SetupParams level_setup_params = {
      3,       // Number of levels
      {6,8},   // Null vecs on L0, L1
      {
          {2,2,2,2},  // Block Size from L0->L1
          {2,2,2,2}   // Block Size from L1->L2
      },
      {500,500},          // Max Nullspace Iters
      {5e-6,5e-6},        // Nullspace Target Resid
      {false,false}
  };

  MGLevelQPhiXF qphix_level;
  MGLevelCoarse coarse_level;
  MGLevelQDPXX qdpxx_level;
  MGLevelCoarse coarse_level2;
  qdpxx_level.null_vecs.resize(6);
  qphix_level.null_vecs.resize(6);

  // I want QPhiX and QDP++ to have the same gaussian vectors.
  for(int j=0; j < 6; ++j ) {
    gaussian(qdpxx_level.null_vecs[j]);
    qphix_level.null_vecs[j] = std::make_shared<QPhiXSpinorF>(info);
    QDPSpinorToQPhiXSpinor(qdpxx_level.null_vecs[j], *(qphix_level.null_vecs[j]));
  }

  std::shared_ptr<QDPWilsonCloverLinearOperator> M_qdp=
      std::make_shared<QDPWilsonCloverLinearOperator>(m_q, c_sw, t_bc,u);


  // This should now just make blocks and block orthogonalize twice
  SetupQDPXXToCoarse(level_setup_params, M_qdp, qdpxx_level, coarse_level2);

  // OK. We now have a block list.
  // We can test the block blas:
  LatticeFermion x, y, y2;
  QPhiXSpinorF qx(info), qy(info), qy2(info);

  std::complex<double> calpha(1.2,-2.3);

  double dfalfa = 1.2;

  // Test BLAS axBlock
  {
    MasterLog(INFO, "Testing axBlockAggr");
    gaussian(x);

    for(int aggr=0; aggr < 2; ++aggr) {
      for(int b = 0; b < qdpxx_level.blocklist.size(); ++b) {

        y = x;
        QDPSpinorToQPhiXSpinor(x,qy);

        axBlockAggrQDPXX(dfalfa, y, qdpxx_level.blocklist[b], aggr);
        axBlockAggr(dfalfa, qy, qdpxx_level.blocklist[b], aggr);
        DiffSpinorPerSite(y,qy,5.0e-7);
      }
    }
  }

  // Test CaxpyBlockAggr
  {
    MasterLog(INFO, "Testing caxpyBlockAggr");
    gaussian(x); gaussian(y);

    for(int aggr=0; aggr < 2; ++aggr) {
      for(int b = 0; b < qdpxx_level.blocklist.size(); ++b) {
        LatticeFermion x2=x; LatticeFermion y2=y;

        QDPSpinorToQPhiXSpinor(x2,qx);
        QDPSpinorToQPhiXSpinor(y2,qy);
        caxpyBlockAggrQDPXX(calpha, x2, y2, qdpxx_level.blocklist[b], aggr);
        caxpyBlockAggr(calpha,qx,qy,qdpxx_level.blocklist[b],aggr);

        DiffSpinorPerSite(y2,qy,8.0e-7);
        DiffSpinorPerSite(x2,qx,8.0e-7);
      }
    }

  }

  // Test Norm2BlockAggr
  {
    MasterLog(INFO, "Testing norm2BlockAggr");
    gaussian(x);
    for(int aggr=0; aggr < 2; ++aggr) {
      for(int b = 0; b < qdpxx_level.blocklist.size(); ++b) {
        LatticeFermion x2=x;
        QDPSpinorToQPhiXSpinor(x2,qx);
        Block& block = qdpxx_level.blocklist[b];
        double qdp_norm = norm2BlockAggrQDPXX(x2,block,aggr);
        double qphix_norm = norm2BlockAggr(qx,block,aggr);
        double normdiff = std::abs(qdp_norm - qphix_norm );
        ASSERT_LT( normdiff, 6e-6);
        double normdiff_per_site = normdiff / static_cast<double>(block.getNumSites());
        ASSERT_LT( normdiff_per_site, 5.0e-7);
      }
    }
  }

  // Test Norm2BlockAggr
  {
    MasterLog(INFO, "Testing innerProductBlockAggr");
    gaussian(x);
    for(int aggr=0; aggr < 2; ++aggr) {
      for(int b = 0; b < qdpxx_level.blocklist.size(); ++b) {
        LatticeFermion x2=x;
        LatticeFermion y2=y;
        QDPSpinorToQPhiXSpinor(x2,qx);
        QDPSpinorToQPhiXSpinor(y2,qy);
        Block& block = qdpxx_level.blocklist[b];
        std::complex<double> qdp_iprod = innerProductBlockAggrQDPXX(x2,y2,block,aggr);
        std::complex<double> qphix_iprod = innerProductBlockAggr(qx,qy,block,aggr);
        double diff_re = std::abs(real(qdp_iprod) - real(qphix_iprod));
        double diff_im = std::abs(imag(qdp_iprod) - imag(qphix_iprod));
        ASSERT_LT( diff_re, 5.0e-6);
        ASSERT_LT( diff_im, 5.0e-6);
        double diff_re_per_site = diff_re / static_cast<double>(block.getNumSites());
        double diff_im_per_site = diff_im / static_cast<double>(block.getNumSites());
        ASSERT_LT( diff_re_per_site, 5.0e-7);
        ASSERT_LT( diff_im_per_site, 5.0e-7);
      }
    }
  }

  // Test extract aggregate
  {
    MasterLog(INFO, "Testing extractAggregateBlock");
    gaussian(x);
    gaussian(y);
    for(int aggr=0; aggr < 2; ++aggr) {
      for(int b = 0; b < qdpxx_level.blocklist.size(); ++b) {
        LatticeFermion x2=x;
        LatticeFermion y2=y;
        QDPSpinorToQPhiXSpinor(x2,qx);
        QDPSpinorToQPhiXSpinor(y2,qy);
        Block& block = qdpxx_level.blocklist[b];
        extractAggregateQDPXX(x2,y2,block,aggr);
        extractAggregateBlock(qx,qy,block,aggr);
        DiffSpinorPerSite(x2,qx,5.0e-6);
        DiffSpinorPerSite(y2,qy,5.0e-6);
      }
    }
  }

  // Test Extract aggregate
  {
    MasterLog(INFO, "Testing extractAggregate");
    gaussian(x);
    gaussian(y);
    for(int aggr=0; aggr < 2; ++aggr ) {
      LatticeFermion x2 = x;
      LatticeFermion y2 = y;
      QDPSpinorToQPhiXSpinor(x2,qx);
      QDPSpinorToQPhiXSpinor(y2,qy);
      extractAggregateQDPXX(x2,y2,aggr);
      extractAggregate(qx,qy,aggr);
      DiffSpinorPerSite(x2,qx,5.0e-6);
      DiffSpinorPerSite(y2,qy,5.0e-6);
    }
  }

  // Test Orthonormalize Block Aggregates
  {
    MasterLog(INFO, "Testing block orthonormalization\n");

    for(int j=0; j < 6; ++j ) {
      gaussian(qdpxx_level.null_vecs[j]);
      qphix_level.null_vecs[j] = std::make_shared<QPhiXSpinorF>(info);
      QDPSpinorToQPhiXSpinor(qdpxx_level.null_vecs[j], *(qphix_level.null_vecs[j]));
    }

    orthonormalizeBlockAggregatesQDPXX(qdpxx_level.null_vecs,qdpxx_level.blocklist);
    orthonormalizeBlockAggregates(qphix_level.null_vecs, qdpxx_level.blocklist);
    for(int j=0; j < 6; ++j) {
      DiffSpinorPerSite( qdpxx_level.null_vecs[j], *(qphix_level.null_vecs[j]), 5.0e-6);
    }
  }
}

TEST(QPhIXIntegration, RNGSeeds)
{
  IndexArray latdims={{8,8,8,8}};
  initQDPXXLattice(latdims);
  LatticeInfo info(latdims);
  LatticeFermion q1,q2;


  // I want QPhiX and QDP++ to have the same gaussian vectors.
  QDP::Seed saved_seed;
  QDP::RNG::savern(saved_seed);
  gaussian(q1);
  gaussian(q2);

  QDP::RNG::setrn(saved_seed);
  QPhiXSpinorF qq(info), qq2(info);
  Gaussian(qq);
  Gaussian(qq2);
  DiffSpinorPerSite(q1,qq,5.0e-6);
  DiffSpinorPerSite(q2,qq2,5.0e-6);;

}

TEST(QPhiXIntegration, TestSetupQDPXXVecs)
{
  // Init the lattice
  IndexArray latdims={{8,8,8,8}};
  initQDPXXLattice(latdims);
  LatticeInfo info(latdims);

  multi1d<LatticeColorMatrix> u(Nd);
  for(int mu=0; mu < Nd; ++mu) {
    // u[mu]=1;
    gaussian(u[mu]);
    reunit(u[mu]);
  }


  float m_q = 0.01;
  float c_sw = 1.25;

  int t_bc=-1; // Antiperiodic t BCs

  QDPIO::cout << "Creating M" << std::endl;

  std::shared_ptr<QPhiXWilsonCloverLinearOperatorF> M=
      std::make_shared<QPhiXWilsonCloverLinearOperatorF>(info,m_q, c_sw, t_bc,u);

  SetupParams level_setup_params = {
      3,       // Number of levels
      {6,8},   // Null vecs on L0, L1
      {
          {2,2,2,2},  // Block Size from L0->L1
          {2,2,2,2}   // Block Size from L1->L2
      },
      {500,500},          // Max Nullspace Iters
      {5e-6,5e-6},        // Nullspace Target Resid
      {false,false}
  };

  MGLevelQPhiXF qphix_level;
  MGLevelCoarse coarse_level;
  MGLevelQDPXX qdpxx_level;
  MGLevelCoarse coarse_level2;


  // I want QPhiX and QDP++ to have the same gaussian vectors.



  std::shared_ptr<QDPWilsonCloverLinearOperator> M_qdp=
      std::make_shared<QDPWilsonCloverLinearOperator>(m_q, c_sw, t_bc,u);

  MasterLog(INFO, "GeneratingVecs with QPhiX Setup");
  // This makes the vecs
  SetupQPhiXToCoarseGenerateVecs(level_setup_params, M, qphix_level, coarse_level2);

  MasterLog(INFO, "Injecting QPhiX solutions into QDPXX Level");
  // Now let us inject the ves into the QPhiX One
  qdpxx_level.null_vecs.resize(qphix_level.null_vecs.size());
  for(int k = 0; k < qdpxx_level.null_vecs.size(); ++k ) {
    QPhiXSpinorToQDPSpinor(*(qphix_level.null_vecs[k]), qdpxx_level.null_vecs[k]);
  }

  MasterLog(INFO, "Finishing QDPXX Setup");
  SetupQDPXXToCoarseVecsIn(level_setup_params, M_qdp,qdpxx_level,coarse_level);

  MasterLog(INFO, "Finishing QPhiX Setup");
  SetupQPhiXToCoarseVecsIn(level_setup_params, M, qphix_level, coarse_level2);

  // NB: If I do the solves, I cannot possibly compare the vectors, since
  // the solves themselves solve Dx=0 and x is going to be approximate to 0.
  // Further QPhiX is done with EO solves, so different iterates.
  // Upshot: solves won't be identical in components necessarily per-site just
  // both will fulfill some absolute residuum condition. Since they are essentially
  // Zero it could be that this can be swamped easily.
  // So will copy in QDP++ vecs

  MasterLog(INFO, "Comparing Block Lists");
  // Compare block lists...
  ASSERT_EQ( qphix_level.blocklist.size(), qdpxx_level.blocklist.size());
  for(int b=0; b < qdpxx_level.blocklist.size(); ++b) {
    Block& qphix_block = qphix_level.blocklist[b];
    Block& qdpxx_block = qdpxx_level.blocklist[b];

    const vector<CBSite>& qphix_sitelist = qphix_block.getCBSiteList();
    const vector<CBSite>& qdpxx_sitelist = qphix_block.getCBSiteList();
    ASSERT_EQ( qphix_sitelist.size(), qdpxx_sitelist.size());

    for(int bsite = 0; bsite < qphix_sitelist.size(); bsite++) {
      const CBSite& qphix_site = qphix_sitelist[bsite];
      const CBSite& qdpxx_site = qdpxx_sitelist[bsite];
      ASSERT_EQ( qphix_site.cb, qdpxx_site.cb);
      ASSERT_EQ( qphix_site.site, qdpxx_site.site);
    }
  }

  MasterLog(INFO, "Comparing Block Orthogonalized NULL Vecs");
  for(int k=0; k < 6; ++k) {
    DiffSpinorPerSite(qdpxx_level.null_vecs[k], *(qphix_level.null_vecs[k]), 5.0e-6);
  }

  // Check Coarse Level info
  MasterLog(INFO, "Checking pointers for QDP++ Coarse level");
  ASSERT_NE( coarse_level.info, nullptr );
  ASSERT_NE( coarse_level.gauge, nullptr );
  ASSERT_NE( coarse_level.M, nullptr );

  MasterLog(INFO, "Checking poiners for QPhiX Coarse level");
  ASSERT_NE( coarse_level2.info, nullptr );
  ASSERT_NE( coarse_level2.gauge, nullptr );
  ASSERT_NE( coarse_level2.M, nullptr );


  auto& qdpxx_coarse_info = *(coarse_level.info);
  auto& qdpxx_u_coarse = *(coarse_level.gauge);
  auto& qdpxx_coarse_M = *(coarse_level.M);

  ASSERT_EQ( qdpxx_coarse_info.GetNumSpins(), 2);
  ASSERT_EQ( qdpxx_coarse_info.GetNumColors(), 6);

  const IndexArray& qdpxx_coarse_dims = qdpxx_coarse_info.GetLatticeDimensions();
  for(int j=0; j < 4; ++j ) ASSERT_EQ( qdpxx_coarse_dims[j], 4);

  auto& qphix_coarse_info = *(coarse_level2.info);
  auto& qphix_u_coarse = *(coarse_level2.gauge);
  auto& qphix_coarse_M = *(coarse_level2.M);

  ASSERT_EQ( qphix_coarse_info.GetNumSpins(), 2);
  ASSERT_EQ( qphix_coarse_info.GetNumColors(), 6);

  const IndexArray& qphix_coarse_dims = qphix_coarse_info.GetLatticeDimensions();
  for(int j=0; j < 4; ++j ) ASSERT_EQ( qphix_coarse_dims[j], 4);

  MasterLog(INFO, "Cross Checking Gauge Fields ");
  {
    int n_coarse_cbsites = qphix_coarse_info.GetNumCBSites();

    for(IndexType cb=0; cb < 2; ++cb) {
      for(IndexType cbsite=0; cbsite < n_coarse_cbsites; cbsite++) {
        for(IndexType mu=0; mu < 9; ++mu) {
          float *qdp_site = qdpxx_u_coarse.GetSiteDirDataPtr(cb, cbsite,mu);
          float *qphix_site = qphix_u_coarse.GetSiteDirDataPtr(cb,cbsite,mu);
          for(int ij=0; ij < n_complex*12*12; ++ij) {
            float diff = std::abs(qphix_site[ij]-qdp_site[ij]);
            ASSERT_LT(diff, 1.0e-6);
          }
        }

      }
    }
  }

  // Check Prolongation
  CoarseSpinor in( qphix_coarse_info );
  CoarseSpinor out( qphix_coarse_info );
  Gaussian(in);
  ZeroVec(out);
  double norm_in = Norm2Vec(in);

  const LatticeInfo& fine_info = *(qphix_level.info);

  QPhiXSpinorF qphix_out(fine_info);
  ZeroVec(qphix_out);

   prolongateSpinor(qphix_level.blocklist, qphix_level.null_vecs, in, qphix_out);
   restrictSpinor(qphix_level.blocklist, qphix_level.null_vecs, qphix_out, out);

   double diff = XmyNorm2Vec(in,out);

   MasterLog(INFO, "QPHIX: || (1 - RP) psi || = %16.8e || (1 - RP) psi || / || psi || = %16.8e",
         sqrt(diff), sqrt(diff/norm_in));

   ASSERT_LT( sqrt(diff/norm_in), 5.0e-7);

   Gaussian(in);
   norm_in = Norm2Vec(in);

   ZeroVec(out);
   LatticeFermion tmp;
   prolongateSpinorCoarseToQDPXXFine(qdpxx_level.blocklist,  qdpxx_level.null_vecs, in , tmp);
   restrictSpinorQDPXXFineToCoarse(qdpxx_level.blocklist, qdpxx_level.null_vecs, tmp, out);
   diff = XmyNorm2Vec(in,out);

   MasterLog(INFO, "QDP++: || (1 - RP) psi || = %16.8e || (1 - RP) psi || / || psi || = %16.8e",
       sqrt(diff), sqrt(diff/norm_in));
   ASSERT_LT( sqrt(diff/norm_in), 5.0e-7);


}

TEST(QPhiXIntegration, TestSetup)
{
  // Init the lattice
  IndexArray latdims={{8,8,8,8}};
  initQDPXXLattice(latdims);
  LatticeInfo info(latdims);

  multi1d<LatticeColorMatrix> u(Nd);
  for(int mu=0; mu < Nd; ++mu) {
    // u[mu]=1;
    gaussian(u[mu]);
    reunit(u[mu]);
  }


  float m_q = 0.01;
  float c_sw = 1.25;

  int t_bc=-1; // Antiperiodic t BCs

  QDPIO::cout << "Creating M" << std::endl;

  std::shared_ptr<QPhiXWilsonCloverLinearOperatorF> M=
      std::make_shared<QPhiXWilsonCloverLinearOperatorF>(info,m_q, c_sw, t_bc,u);

  SetupParams level_setup_params = {
      3,       // Number of levels
      {6,8},   // Null vecs on L0, L1
      {
          {2,2,2,2},  // Block Size from L0->L1
          {2,2,2,2}   // Block Size from L1->L2
      },
      {500,500},          // Max Nullspace Iters
      {5e-6,5e-6},        // Nullspace Target Resid
      {false,false}
  };

  QPhiXMultigridLevels mg_levels;

  MultigridLevels mg_levels_qdpxx;

  SetupQPhiXMGLevels(level_setup_params, mg_levels, M);

  // I want to test that restriction and prolongation work.
  const LatticeInfo& coarse_info = *(mg_levels.coarse_levels[0].info);
  CoarseSpinor in(coarse_info);
  Gaussian(in);
  double norm_in = Norm2Vec(in);
  CoarseSpinor out(coarse_info);
  ZeroVec(out);

  const LatticeInfo& fine_info = *(mg_levels.fine_level.info);
  QPhiXSpinorF qphix_out(fine_info);
  ZeroVec(qphix_out);

  // Test Prolongator and Restrictor
  const std::vector<Block>& blocklist = mg_levels.fine_level.blocklist;
  prolongateSpinor(blocklist,mg_levels.fine_level.null_vecs, in, qphix_out);
  restrictSpinor(blocklist,mg_levels.fine_level.null_vecs, qphix_out, out);

  double diff = XmyNorm2Vec(in,out);
  MasterLog(INFO, "QPHIX: || (1 - RP) psi || = %16.8e || (1 - RP) psi || / || psi || = %16.8e",
         sqrt(diff), sqrt(diff/norm_in));

   ASSERT_LT( sqrt(diff/norm_in), 5.0e-7);

   // Test Fake Coarse Op:  R D P = D_c
   Gaussian(in);
   auto& D_c = *(mg_levels.coarse_levels[0].M);
   D_c(out,in,LINOP_OP);

   QPhiXSpinorF DP_in(fine_info);
   prolongateSpinor(blocklist, mg_levels.fine_level.null_vecs, in, qphix_out);
   (*M)(DP_in,qphix_out,LINOP_OP);
   CoarseSpinor RDP_in(coarse_info);
   restrictSpinor(blocklist, mg_levels.fine_level.null_vecs, DP_in,RDP_in );

   diff = XmyNorm2Vec(RDP_in,out);
   MasterLog(INFO, "QPHIX: || (RDP - D_c) psi || = %16.8e || (RDP - D_c) psi || / || psi || = %16.8e",
          sqrt(diff), sqrt(diff/norm_in));

    ASSERT_LT( sqrt(diff/norm_in), 5.0e-6);

}

int main(int argc, char *argv[])
{
  return ::MGTesting::TestMain(&argc, argv);
}




