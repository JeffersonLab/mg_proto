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

void DiffCBSpinor(const LatticeFermion& s1, const QPhiXSpinor& qphix_spinor, int cb, double tol)
{

  LatticeFermion s2;

  QPhiXSpinorToQDPSpinor(qphix_spinor,s2);

  LatticeFermion diff=zero;
  diff[rb[cb]]=s1-s2;
  double vol_cb=static_cast<double>(Layout::vol())/2;


  double r_norm_cb = toDouble(sqrt(norm2(diff,rb[cb])));
  double r_norm_cb_per_site = r_norm_cb / vol_cb;
  MasterLog(INFO, "CB %d : || r || = %16.8e  || r ||/site = %16.8e",cb, r_norm_cb, r_norm_cb_per_site);

  ASSERT_LT( r_norm_cb, tol);

}

void DiffSpinor(const LatticeFermion& s1, const QPhiXSpinor& qphix_spinor, double tol)
{
  LatticeFermion s2;
  QPhiXSpinorToQDPSpinor(qphix_spinor,s2);

  LatticeFermion diff=s1-s2;
  double vol=static_cast<double>(Layout::vol());
  double vol_cb=vol/2;

  double r_norm_cb0 = toDouble(sqrt(norm2(diff,rb[0])));
  double r_norm_cb1 = toDouble(sqrt(norm2(diff,rb[1])));
  double r_norm_cb0_per_site = r_norm_cb0 / vol_cb;
  double r_norm_cb1_per_site = r_norm_cb1 / vol_cb;
  MasterLog(INFO, "CB 0 : || r || = %16.8e  || r ||/site = %16.8e", r_norm_cb0, r_norm_cb0_per_site);
  MasterLog(INFO, "CB 1 : || r || = %16.8e  || r ||/site = %16.8e", r_norm_cb1, r_norm_cb1_per_site);

  double r_norm = toDouble(sqrt(norm2(diff)));
  double r_norm_per_site = r_norm /vol;
  MasterLog(INFO, "Full: || r || = %16.8e || r ||/site = %16.8e", r_norm, r_norm_per_site);
  ASSERT_LT( r_norm, tol);

}

TEST(QPhiXIntegration, TestQPhiXUnprecOpIsoConstructor)
{
  // Init the lattice
  IndexArray latdims={{8,8,8,8}};
  initQDPXXLattice(latdims);
  LatticeInfo info(latdims);



  int t_bc = +1;
  double t_bcf = static_cast<double>(t_bc);
  double m_q = 0.01;
  double c_sw = 1.2;


  multi1d<LatticeColorMatrix> u(Nd);
  for(int mu=0; mu < Nd; ++mu) {
    gaussian(u[mu]);
    reunit(u[mu]);
  }

  // Make the QPhiX Clover op
  MG::QDPWilsonCloverLinearOperator D_full(m_q, c_sw, t_bc, u);
  MG::QPhiXWilsonCloverLinearOperator D_qphix(info, m_q,c_sw,t_bc,u);

  // Let us make the source
  LatticeFermion source;
  gaussian(source);

  LatticeFermion result=zero;
  LatticeFermion qphix_result=zero;

  QPhiXSpinor q_src(info);
  QPhiXSpinor q_res(info);

  QDPSpinorToQPhiXSpinor(source,q_src);

  // Test Full Op
  MasterLog(INFO,"Checking Unprec Op");
  for(IndexType l_type=LINOP_OP; l_type <= LINOP_DAGGER; ++l_type ) {
    l_type == LINOP_OP ? MasterLog(INFO, "Checking Op") : MasterLog(INFO, "Checking Dagger");
    D_full(result,source,l_type);
    D_qphix(q_res,q_src,l_type);

    DiffSpinor(result,q_res,5.0e-13);
  }

  // Test Full Op
  MasterLog(INFO, "Checking Diag Op");
  for(IndexType l_type=LINOP_OP; l_type <= LINOP_DAGGER; ++l_type ) {
    l_type == LINOP_OP ? MasterLog(INFO, "Checking Op") : MasterLog(INFO, "Checking Dagger");
    MasterLog(INFO, "M_ee:");
    gaussian(result);
    D_full.M_ee(result,source,l_type);
    D_qphix.M_ee(q_res,q_src,l_type);
    DiffCBSpinor(result,q_res,EVEN,5.0e-13);

    gaussian(result);
    MasterLog(INFO, "M_oo:");
    D_full.M_oo(result,source,l_type);
    D_qphix.M_oo(q_res,q_src,l_type);
    DiffCBSpinor(result,q_res,ODD,5.0e-13);

    gaussian(result);
    MasterLog(INFO,"M_ee_inv:");
    D_full.M_ee_inv(result,source,l_type);
    D_qphix.M_ee_inv(q_res,q_src,l_type);
    DiffCBSpinor(result,q_res,EVEN,5.0e-13);

    gaussian(result);
    MasterLog(INFO, "M_eo:");
    D_full.M_eo(result,source,l_type);
    D_qphix.M_eo(q_res,q_src,l_type);
    DiffCBSpinor(result,q_res,EVEN,5.0e-13);

    gaussian(result);
    MasterLog(INFO,"M_oe:");
    D_full.M_oe(result,source,l_type);
    D_qphix.M_oe(q_res,q_src,l_type);
    DiffCBSpinor(result,q_res,ODD,5.0e-13);

  }


}

TEST(QPhiXIntegration, TestQPhiXUnprecOpAnisoConstructor)
{
  // Init the lattice
  IndexArray latdims={{8,8,8,8}};
  initQDPXXLattice(latdims);
  LatticeInfo info(latdims);



  int t_bc = +1;
  double t_bcf = static_cast<double>(t_bc);
  double m_q = 0.01;
  double u0=0.897;
  double xi0=3.54;
  double nu = 0.95;
  double csw_r = 1.245;
  double csw_t = 0.956;

  multi1d<LatticeColorMatrix> u(Nd);
  for(int mu=0; mu < Nd; ++mu) {
    gaussian(u[mu]);
    reunit(u[mu]);
  }

  // Make the QPhiX Clover op
  MG::QDPWilsonCloverLinearOperator D_full(m_q, u0,xi0,nu, csw_r,csw_t, t_bc, u);
  MG::QPhiXWilsonCloverLinearOperator D_qphix(info, m_q, u0, xi0, nu, csw_r, csw_t, t_bc, u);

  // Let us make the source
  LatticeFermion source;
  gaussian(source);

  LatticeFermion result=zero;
  LatticeFermion qphix_result=zero;

  QPhiXSpinor q_src(info);
  QPhiXSpinor q_res(info);

  QDPSpinorToQPhiXSpinor(source,q_src);

  // Test Full Op
  MasterLog(INFO,"Checking Unprec Op");
  for(IndexType l_type=LINOP_OP; l_type <= LINOP_DAGGER; ++l_type ) {
    l_type == LINOP_OP ? MasterLog(INFO, "Checking Op") : MasterLog(INFO, "Checking Dagger");
    D_full(result,source,l_type);
    D_qphix(q_res,q_src,l_type);

    DiffSpinor(result,q_res,5.0e-13);
  }

  // Test Full Op
  MasterLog(INFO, "Checking Diag Op");
  for(IndexType l_type=LINOP_OP; l_type <= LINOP_DAGGER; ++l_type ) {
    l_type == LINOP_OP ? MasterLog(INFO, "Checking Op") : MasterLog(INFO, "Checking Dagger");

    MasterLog(INFO, "M_ee:");
    gaussian(result);
    D_full.M_ee(result,source,l_type);
    D_qphix.M_ee(q_res,q_src,l_type);
    DiffCBSpinor(result,q_res,EVEN,5.0e-13);

    gaussian(result);
    MasterLog(INFO, "M_oo:");
    D_full.M_oo(result,source,l_type);
    D_qphix.M_oo(q_res,q_src,l_type);
    DiffCBSpinor(result,q_res,ODD,5.0e-13);

    gaussian(result);
    MasterLog(INFO,"M_ee_inv:");
    D_full.M_ee_inv(result,source,l_type);
    D_qphix.M_ee_inv(q_res,q_src,l_type);
    DiffCBSpinor(result,q_res,EVEN,5.0e-13);

    gaussian(result);
    MasterLog(INFO, "M_eo:");
    D_full.M_eo(result,source,l_type);
    D_qphix.M_eo(q_res,q_src,l_type);
    DiffCBSpinor(result,q_res,EVEN,5.0e-13);

    gaussian(result);
    MasterLog(INFO,"M_oe:");
    D_full.M_oe(result,source,l_type);
    D_qphix.M_oe(q_res,q_src,l_type);
    DiffCBSpinor(result,q_res,ODD,5.0e-13);

  }


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
