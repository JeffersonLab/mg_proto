/*
 * test_qphix_interface.cpp
 *
 *  Created on: Oct 13, 2017
 *      Author: bjoo
 */

#include <gtest/gtest.h>
#include "../test_env.h"
#include "lattice/qphix/qphix_types.h"
#include "lattice/qphix/qphix_qdp_utils.h"

#include "lattice/constants.h"
#include "../qdpxx/qdpxx_latticeinit.h"
#include "../qdpxx/qdpxx_utils.h"
#include "lattice/fine_qdpxx/qdpxx_helpers.h"

#include "lattice/fine_qdpxx/clover_term_qdp_w.h"
#include "lattice/fine_qdpxx/dslashm_w.h"
#include "lattice/fine_qdpxx/wilson_clover_linear_operator.h"
#include "lattice/qphix/qphix_clover_linear_operator.h"

using namespace QDP;
using namespace MG;
using namespace MGTesting;

TEST(QPhiXIntegration, DeathUninitializedGeom)
{
  // Init the lattice
   IndexArray latdims={{8,8,8,8}};
   initQDPXXLattice(latdims);

   ASSERT_EQ( MGQPhiX::IsGeomInitialized(), false);
   ASSERT_DEATH( MGQPhiX::GetGeom(), "Aborted" );
}

using namespace MGQPhiX;
TEST(QPhiXIntegration, InitGeom)
{
  // Init the lattice
     IndexArray latdims={{8,8,8,8}};
     initQDPXXLattice(latdims);
     LatticeInfo info(latdims);

     ASSERT_EQ( IsGeomInitialized(), false);
     InitializeGeom(info);
     ASSERT_EQ( IsGeomInitialized(), true);

     Geom& geom = GetGeom();
     IndexArray dims_back={{geom.Nx(),geom.Ny(),geom.Nz(),geom.Nt() }};
     for(int mu=0; mu < 4; ++mu) {
       ASSERT_EQ( latdims[mu], dims_back[mu] );
     }

}

TEST(QPhiXIntegration, CreateQPhiXContainers)
{
  // Init the lattice
     IndexArray latdims={{8,8,8,8}};
     initQDPXXLattice(latdims);
     LatticeInfo info(latdims);
     QPhiXSpinor spinor(info);
     QPhiXGauge gauge(info);
     QPhiXClover clover(info);

}

TEST(QPhiXIntegration, TestQPhiXBiCGStabAbsolute)
{
  // Init the lattice
  IndexArray latdims={{8,8,8,8}};
  initQDPXXLattice(latdims);
  LatticeInfo info(latdims);



  int t_bc = +1;
  double t_bcf = static_cast<double>(t_bc);
  double m_q = 0.01;
  double c_sw = 1.2;


  multi1d<LatticeColorMatrixF> u_f(Nd);
  multi1d<LatticeColorMatrix> u(Nd);
  for(int mu=0; mu < Nd; ++mu) {
    gaussian(u[mu]);
    reunit(u[mu]);
    u_f[mu] =  u[mu]; // Downcast to single prec
  }

  // Make the QPhiX Clover op
  MG::QDPWilsonCloverLinearOperator D_full(m_q, c_sw, t_bc, u);
  MG::QPhiXWilsonCloverLinearOperator D_qphix(info, m_q,c_sw,t_bc,u);

  // Let us make the source
  LatticeFermion source;
  source = zero;

  LatticeFermion solution;
  gaussian(solution); // Initial guess


  LatticeFermion transf_source = source;

  QPhiXSpinor source_full(info);
  QPhiXSpinor solution_full(info);
  QPhiXGauge  qphix_u(info);
  QPhiXClover qphix_clov(info);

  QDPGaugeFieldToQPhiXGauge(u,qphix_u);

  QDPSpinorToQPhiXSpinor(transf_source,source_full);
  QDPSpinorToQPhiXSpinor(solution,solution_full);


  // make a BiCGStab Solver
  BiCGStab solver(D_qphix.getQPhiXOp(), 5000);
  QPhiXUnprecSolver unprec_wrapper(solver,D_qphix.getQPhiXOp());


  int n_iters;
  double rsd_sq_final;
  unsigned long site_flops;
  unsigned long mv_apps;

  unprec_wrapper(&(solution_full.get()),&(source_full.get()), 1.0e-7, n_iters, rsd_sq_final,site_flops,mv_apps,1, true, ODD, QPhiX::ABSOLUTE);

  // Solution[odd] is the same between the transformed and untransformed system
  QPhiXSpinorToQDPSpinor(solution_full,solution);

  // Check solution
  LatticeFermion tmp=QDP::zero;
  D_full(tmp, solution, LINOP_OP);
  LatticeFermion r = source - tmp;
  double r_norm_cb0 = toDouble(sqrt(norm2(r,rb[0])));
  double r_norm_cb1 = toDouble(sqrt(norm2(r,rb[1])));


  MasterLog(INFO, "CB 0 : || r || = %16.8e", r_norm_cb0);
  MasterLog(INFO, "CB 1 : || r || = %16.8e", r_norm_cb1);

  double r_norm = toDouble(sqrt(norm2(r)));
  MasterLog(INFO, "Full: || r || = %16.8e", r_norm);
  ASSERT_LT( r_norm, 1.0e-6);
}


int main(int argc, char *argv[])
{
  return ::MGTesting::TestMain(&argc, argv);
}
