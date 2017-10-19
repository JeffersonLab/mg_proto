/*
 * test_qphix_interface.cpp
 *
 *  Created on: Oct 13, 2017
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

using namespace QDP;
using namespace MG;
using namespace MGTesting;

TEST(QPhiXIntegration, DeathUninitializedGeom)
{
  // Init the lattice
   IndexArray latdims={{8,8,8,8}};
   initQDPXXLattice(latdims);

   ASSERT_EQ( MGQPhiX::IsGeomInitialized(), false);
   ASSERT_DEATH( MGQPhiX::GetGeom<float>(), "Aborted" );
   ASSERT_DEATH( MGQPhiX::GetGeom<double>(), "Aborted" );
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

     Geom& geom = GetGeom<double>();
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

TEST(QPhiXIntegration, TestQPhiXUnprecOpAnisoConstructorF)
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
  MG::QPhiXWilsonCloverLinearOperatorF D_qphix(info, m_q, u0, xi0, nu, csw_r, csw_t, t_bc, u);

  // Let us make the source
  LatticeFermion source;
  gaussian(source);

  LatticeFermion result=zero;
  LatticeFermion qphix_result=zero;

  QPhiXSpinorF q_src(info);
  QPhiXSpinorF q_res(info);

  QDPSpinorToQPhiXSpinor(source,q_src);

  // Test Full Op
  MasterLog(INFO,"Checking Unprec Op");
  for(IndexType l_type=LINOP_OP; l_type <= LINOP_DAGGER; ++l_type ) {
    l_type == LINOP_OP ? MasterLog(INFO, "Checking Op") : MasterLog(INFO, "Checking Dagger");
    D_full(result,source,l_type);
    D_qphix(q_res,q_src,l_type);

    DiffSpinor(result,q_res,5.0e-7,true);
  }

  // Test Full Op
  MasterLog(INFO, "Checking Diag Op");
  for(IndexType l_type=LINOP_OP; l_type <= LINOP_DAGGER; ++l_type ) {
    l_type == LINOP_OP ? MasterLog(INFO, "Checking Op") : MasterLog(INFO, "Checking Dagger");

    MasterLog(INFO, "M_ee:");
    gaussian(result);
    D_full.M_ee(result,source,l_type);
    D_qphix.M_ee(q_res,q_src,l_type);
    DiffCBSpinor(result,q_res,EVEN,5.0e-7,true);

    gaussian(result);
    MasterLog(INFO, "M_oo:");
    D_full.M_oo(result,source,l_type);
    D_qphix.M_oo(q_res,q_src,l_type);
    DiffCBSpinor(result,q_res,ODD,5.0e-7,true);

    gaussian(result);
    MasterLog(INFO,"M_ee_inv:");
    D_full.M_ee_inv(result,source,l_type);
    D_qphix.M_ee_inv(q_res,q_src,l_type);
    DiffCBSpinor(result,q_res,EVEN,5.0e-7,true);

    gaussian(result);
    MasterLog(INFO, "M_eo:");
    D_full.M_eo(result,source,l_type);
    D_qphix.M_eo(q_res,q_src,l_type);
    DiffCBSpinor(result,q_res,EVEN,5.0e-7,true);

    gaussian(result);
    MasterLog(INFO,"M_oe:");
    D_full.M_oe(result,source,l_type);
    D_qphix.M_oe(q_res,q_src,l_type);
    DiffCBSpinor(result,q_res,ODD,5.0e-7, true);

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


  LinearSolverParamsBase params;
  params.MaxIter = 5000;
  params.RsdTarget= 1.0e-7;
  params.VerboseP = true;
  BiCGStabSolverQPhiX solver(D_qphix,params);
  LinearSolverResults res = solver(solution_full, source_full,ABSOLUTE);

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


TEST(QPhiXIntegration, TestQPhiXBiCGStabRelativeF)
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
  MG::QPhiXWilsonCloverLinearOperatorF D_qphix(info, m_q,c_sw,t_bc,u);

  // Let us make the source
  LatticeFermion source;
  gaussian(source);

  LatticeFermion solution;
  gaussian(solution); // Initial guess


  LatticeFermion transf_source = source;

  QPhiXSpinorF source_full(info);
  QPhiXSpinorF solution_full(info);
  QPhiXGaugeF  qphix_u(info);
  QPhiXCloverF qphix_clov(info);

  QDPGaugeFieldToQPhiXGauge(u,qphix_u);

  QDPSpinorToQPhiXSpinor(transf_source,source_full);
  QDPSpinorToQPhiXSpinor(solution,solution_full);

  LinearSolverParamsBase params;
  params.MaxIter = 5000;
  params.RsdTarget= 1.0e-6;
  params.VerboseP = true;
  BiCGStabSolverQPhiXF solver(D_qphix,params);
  LinearSolverResults res = solver(solution_full, source_full);

  QPhiXSpinorF Ax(info);
  D_qphix(Ax,solution_full,LINOP_OP);
  LatticeFermion Ax_qdp;
  QPhiXSpinorToQDPSpinor(Ax,Ax_qdp);
  DiffSpinorRelative(source,Ax_qdp,1.0e-6);
}

TEST(QPhiXIntegration, QPhiXUnprecFGMRES)
{
  IndexArray latdims={{8,8,8,8}};
  initQDPXXLattice(latdims);

  float m_q = 0.1;
  float c_sw = 1.25;

  int t_bc=-1; // Antiperiodic t BCs

  LatticeFermion in,out;
  gaussian(in);
  out=zero;

  multi1d<LatticeColorMatrix> u(Nd);
  for(int mu=0; mu < Nd; ++mu) {
    gaussian(u[mu]);
    reunit(u[mu]);
  }

  LatticeInfo info(latdims);

  // Create linear operator
  QPhiXWilsonCloverLinearOperator M(info,m_q, c_sw, t_bc, u);

  FGMRESParams params;
  params.MaxIter = 500;
  params.RsdTarget = 1.0e-5;
  params.VerboseP = true;
  params.NKrylov = 10;

  FGMRESSolverQPhiX FGMRES(M, params,nullptr);
  QPhiXSpinor q_b(info);
  QPhiXSpinor q_x(info);

  QDPSpinorToQPhiXSpinor(in,q_b);
  Gaussian(q_x);

  QDPIO::cout << "|| b ||=  " << sqrt(Norm2Vec(q_b)) << std::endl;
  QDPIO::cout << "|| x || = " << sqrt(Norm2Vec(q_x)) << std::endl;

  LinearSolverResults res = FGMRES(q_x,q_b);

  QDPIO::cout << "FGMRES Solver Took: " << res.n_count << " iterations"
      << std::endl;

  ASSERT_EQ(res.resid_type, RELATIVE);
  ASSERT_LT(res.resid, 9e-6);
  QDPWilsonCloverLinearOperator M_qdp(m_q, c_sw, t_bc, u);
  QPhiXSpinorToQDPSpinor(q_x,out);
  LatticeFermion Ax;
  M_qdp(Ax,out, LINOP_OP);

  DiffSpinorRelative(in,Ax,1.0e-5);
}

TEST(QPhiXIntegration, QPhiXUnprecFGMRES2)
{
  IndexArray latdims={{8,8,8,8}};
  initQDPXXLattice(latdims);

  float m_q = 0.1;
  float c_sw = 1.25;

  int t_bc=-1; // Antiperiodic t BCs

  LatticeFermion in,out;
  gaussian(in);
  out=zero;

  multi1d<LatticeColorMatrix> u(Nd);
  for(int mu=0; mu < Nd; ++mu) {
    gaussian(u[mu]);
    reunit(u[mu]);
  }

  LatticeInfo info(latdims);

  // Create linear operator
  QPhiXWilsonCloverLinearOperator M(info,m_q, c_sw, t_bc, u);
  QDPWilsonCloverLinearOperator M_qdp(m_q, c_sw, t_bc, u);

  FGMRESParams params;
  params.MaxIter = 500;
  params.RsdTarget = 1.0e-5;
  params.VerboseP = true;
  params.NKrylov = 10;

  FGMRESSolverQPhiX FGMRES(M, params,nullptr);
  FGMRESSolverQDPXX FGMRESQDP(M_qdp,params,nullptr);

  QPhiXSpinor q_b(info);
  QPhiXSpinor q_x(info);

  QDPSpinorToQPhiXSpinor(in,q_b);
  ZeroVec(q_x);

  QDPIO::cout << "|| b ||=  " << sqrt(Norm2Vec(q_b)) << std::endl;
  QDPIO::cout << "|| x || = " << sqrt(Norm2Vec(q_x)) << std::endl;

  StopWatch swatch;
  swatch.reset(); swatch.start();
  LinearSolverResults res = FGMRES(q_x,q_b);
  swatch.stop();

  // Check Answer
  QDPIO::cout << "FGMRES Solver Took: " << res.n_count << " iterations"
      << std::endl;
  ASSERT_EQ(res.resid_type, RELATIVE);
  ASSERT_LT(res.resid, 9e-6);
  QPhiXSpinorToQDPSpinor(q_x,out);
  LatticeFermion Ax;
  M_qdp(Ax,out, LINOP_OP);
  DiffSpinorRelative(in,Ax,1.0e-5);

  out = zero;
  StopWatch swatch_qdp;
  swatch_qdp.reset(); swatch_qdp.start();
  res = FGMRESQDP(out,in);
  swatch_qdp.stop();
  QDPIO::cout << "QDP FGMRES Solver Took: " << res.n_count << " iterations"
       << std::endl;

  // Check Answers
  ASSERT_EQ(res.resid_type, RELATIVE);
  ASSERT_LT(res.resid, 9e-6);
  M_qdp(Ax,out, LINOP_OP);
  DiffSpinorRelative(in,Ax,1.0e-5);

  // Check Out against q_x
  DiffSpinor(out,q_x, 1.0e-5);

  // Compare times
  double qdp_time = swatch_qdp.getTimeInSeconds();
  double qphix_time = swatch.getTimeInSeconds();
  QDPIO::cout << "QPhiX Based FGMRES took " << qphix_time << " sec" << std::endl;
  QDPIO::cout << "QDP++ Based FGMRES took " << qdp_time << " sec" << std::endl;
  QDPIO::cout << "Speedup = " << qdp_time / qphix_time << " x " << std::endl;
}

TEST(QPhiXIntegration, QPhiXUnprecFGMRES2F)
{
  IndexArray latdims={{8,8,8,8}};
  initQDPXXLattice(latdims);

  float m_q = 0.1;
  float c_sw = 1.25;

  int t_bc=-1; // Antiperiodic t BCs

  LatticeFermion in,out;
  gaussian(in);
  out=zero;

  multi1d<LatticeColorMatrix> u(Nd);
  for(int mu=0; mu < Nd; ++mu) {
    gaussian(u[mu]);
    reunit(u[mu]);
  }

  LatticeInfo info(latdims);

  // Create linear operator
  QPhiXWilsonCloverLinearOperatorF M(info,m_q, c_sw, t_bc, u);
  QDPWilsonCloverLinearOperator M_qdp(m_q, c_sw, t_bc, u);

  FGMRESParams params;
  params.MaxIter = 500;
  params.RsdTarget = 1.0e-5;
  params.VerboseP = true;
  params.NKrylov = 10;

  FGMRESSolverQPhiXF FGMRES(M, params,nullptr);
  FGMRESSolverQDPXX FGMRESQDP(M_qdp,params,nullptr);

  QPhiXSpinorF q_b(info);
  QPhiXSpinorF q_x(info);

  QDPSpinorToQPhiXSpinor(in,q_b);
  ZeroVec(q_x);

  QDPIO::cout << "|| b ||=  " << sqrt(Norm2Vec(q_b)) << std::endl;
  QDPIO::cout << "|| x || = " << sqrt(Norm2Vec(q_x)) << std::endl;

  StopWatch swatch;
  swatch.reset(); swatch.start();
  LinearSolverResults res = FGMRES(q_x,q_b);
  swatch.stop();

  // Check Answer
  QDPIO::cout << "FGMRES Solver Took: " << res.n_count << " iterations"
      << std::endl;
  ASSERT_EQ(res.resid_type, RELATIVE);
  ASSERT_LT(res.resid, 9e-6);
  QPhiXSpinorToQDPSpinor(q_x,out);
  LatticeFermion Ax;
  M_qdp(Ax,out, LINOP_OP);
  DiffSpinorRelative(in,Ax,1.0e-5);

  out = zero;
  StopWatch swatch_qdp;
  swatch_qdp.reset(); swatch_qdp.start();
  res = FGMRESQDP(out,in);
  swatch_qdp.stop();
  QDPIO::cout << "QDP FGMRES Solver Took: " << res.n_count << " iterations"
       << std::endl;

  // Check Answers
  ASSERT_EQ(res.resid_type, RELATIVE);
  ASSERT_LT(res.resid, 9e-6);
  M_qdp(Ax,out, LINOP_OP);
  DiffSpinorRelative(in,Ax,1.0e-5);

  // Check Out against q_x
  DiffSpinor(out,q_x, 1.0e-7,true);

  // Compare times
  double qdp_time = swatch_qdp.getTimeInSeconds();
  double qphix_time = swatch.getTimeInSeconds();
  QDPIO::cout << "QPhiX Based FGMRES (SP) took " << qphix_time << " sec" << std::endl;
  QDPIO::cout << "QDP++ Based FGMRES (SP) took " << qdp_time << " sec" << std::endl;
  QDPIO::cout << "Speedup (SP) = " << qdp_time / qphix_time << " x " << std::endl;
}


TEST(QPhiXIntegration, QPhiXMRSmootherTime)
{
  IndexArray latdims={{8,8,8,8}};
  initQDPXXLattice(latdims);

  float m_q = 0.1;
  float c_sw = 1.25;

  int t_bc=-1; // Antiperiodic t BCs

  LatticeFermion in,out;
  gaussian(in);

  multi1d<LatticeColorMatrix> u(Nd);
  for(int mu=0; mu < Nd; ++mu) {
    gaussian(u[mu]);
    reunit(u[mu]);
  }

  LatticeInfo info(latdims);

  // Create linear operator
  QPhiXWilsonCloverLinearOperatorF M(info,m_q, c_sw, t_bc, u);
  QDPWilsonCloverLinearOperator M_qdp(m_q, c_sw, t_bc, u);
  MRSolverParams params;

  params.MaxIter = 5;
  params.RsdTarget = 1.0e-5;
  params.VerboseP = true;
  params.Omega = 1.1;

  MRSmootherQDPXX QDPXXMR(M_qdp, params);
  MRSmootherQPhiXF QPhiXMRSmoother(M,params);

  QPhiXSpinorF q_b(info);
  QPhiXSpinorF q_x(info);

  QDPSpinorToQPhiXSpinor(in,q_b);

  QDPIO::cout << "|| b ||=  " << sqrt(Norm2Vec(q_b)) << std::endl;
  QDPIO::cout << "|| x || = " << sqrt(Norm2Vec(q_x)) << std::endl;

  QDPIO::cout << "Timing QPhiX MR Smoother" << std::endl;
  StopWatch swatch;
  {
    int n_iters;

    double rsd_sq_final;
    unsigned long site_flops;
    unsigned long mv_apps;
    int isign=1;

    ZeroVec(q_x);

    swatch.reset(); swatch.start();
    QPhiXMRSmoother(q_x,q_b);
    swatch.stop();
  }

  StopWatch swatch_qdp;
  QDPIO::cout << "Timing QDP++ MR Smoother" << std::endl;
  {
    out = zero;


    swatch_qdp.reset(); swatch_qdp.start();
    QDPXXMR(out,in);
    swatch_qdp.stop();
  }

  // Check Answers
  DiffSpinorPerSite(out,q_x,7.0e-2);

  // Compare times
  double qdp_time = swatch_qdp.getTimeInSeconds();
  double qphix_time = swatch.getTimeInSeconds();
  QDPIO::cout << "QPhiX Based Smoother took " << qphix_time << " sec" << std::endl;
  QDPIO::cout << "QDP++ Based Smoother took " << qdp_time << " sec" << std::endl;
  QDPIO::cout << "Speedup = " << qdp_time / qphix_time << " x " << std::endl;
}

TEST(QPhiXIntegration, TestQPhiXMRRelativeF)
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
  MG::QPhiXWilsonCloverLinearOperatorF D_qphix(info, m_q,c_sw,t_bc,u);

  // Let us make the source
  LatticeFermion source;
  gaussian(source);

  LatticeFermion solution;
  gaussian(solution); // Initial guess


  LatticeFermion transf_source = source;

  QPhiXSpinorF source_full(info);
  QPhiXSpinorF solution_full(info);
  QPhiXGaugeF  qphix_u(info);
  QPhiXCloverF qphix_clov(info);

  QDPGaugeFieldToQPhiXGauge(u,qphix_u);

  QDPSpinorToQPhiXSpinor(transf_source,source_full);
  QDPSpinorToQPhiXSpinor(solution,solution_full);

  MRSolverParams params;
  params.MaxIter = 5000;
  params.RsdTarget= 1.0e-6;
  params.VerboseP = true;
  params.Omega = 1.1;
  MRSolverQPhiXF solver(D_qphix,params);
  LinearSolverResults res = solver(solution_full, source_full);

  QPhiXSpinorF Ax(info);
  D_qphix(Ax,solution_full,LINOP_OP);
  LatticeFermion Ax_qdp;
  QPhiXSpinorToQDPSpinor(Ax,Ax_qdp);
  DiffSpinorRelative(source,Ax_qdp,1.0e-6);
}

int main(int argc, char *argv[])
{
  return ::MGTesting::TestMain(&argc, argv);
}
