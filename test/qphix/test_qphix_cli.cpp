/*
 * test_qphix_cli.cpp
 *
 *  Created on: Jun 23, 2017
 *      Author: bjoo
 */

/* Test Environment */
#include <gtest/gtest.h>
#include "../test_env.h"


/* SOA And Veclens */
#include <qphix/qphix_config.h>
#include "./veclen.h"

/* Constants */
#include "lattice/constants.h"
#include "../qdpxx/qdpxx_latticeinit.h"
#include "../qdpxx/qdpxx_utils.h"
#include "lattice/fine_qdpxx/qdpxx_helpers.h"
#include <utils/initialize.h>
#include <qphix/qphix_cli_args.h>
#include <qphix/geometry.h>

#include "lattice/fine_qdpxx/clover_term_qdp_w.h"
#include "lattice/fine_qdpxx/dslashm_w.h"
#include "lattice/fine_qdpxx/wilson_clover_linear_operator.h"

#include <qphix/invbicgstab.h>
#include <qphix/clover.h>
#include <qphix/invbicgstab.h>
#include <qphix/qdp_packer.h>

using namespace QDP;
using namespace MG;
using namespace MGTesting;

using Geom = QPhiX::Geometry<float, VECLEN_SP, QPHIX_SOALEN, false>;
using ClovOp = QPhiX::EvenOddCloverOperator<float, VECLEN_SP, QPHIX_SOALEN, false>;
using BiCGStab = QPhiX::InvBiCGStab<float,VECLEN_SP,QPHIX_SOALEN,false>;
using QPhiXCBSpinor = QPhiX::FourSpinorHandle<float,VECLEN_SP,QPHIX_SOALEN,false>;
using QPhiXCBGauge = QPhiX::GaugeHandle<float,VECLEN_SP,QPHIX_SOALEN, false>;
using QPhiXCBClover = QPhiX::CloverHandle<float, VECLEN_SP,QPHIX_SOALEN, false>;

TEST(QPhiXIntegration, TestCreateGeometry)
{
	// Init the lattice
	IndexArray latdims={{8,8,8,8}};
	initQDPXXLattice(latdims);

	// Get QPhiX Command Line args
	QPhiX::QPhiXCLIArgs& CLI = MG::getQPhiXCLIArgs();

	int lattSize[4] = { latdims[0],
						latdims[1],
						latdims[2],
						latdims[3] };

	int t_bc = -1;

	Geom fine_geometry(lattSize,
						CLI.getBy(),
						CLI.getBz(),
						CLI.getNCores(),
						CLI.getSy(),
						CLI.getSz(),
						CLI.getPxy(),
						CLI.getPxyz(),
						CLI.getMinCt(),
						true);

}

TEST(QPhiXIntegration, TestQPhiXBiCGStab)
{
	// Init the lattice
	IndexArray latdims={{8,8,8,8}};
	initQDPXXLattice(latdims);

	// Get QPhiX Command Line args
	QPhiX::QPhiXCLIArgs& CLI = MG::getQPhiXCLIArgs();

	int lattSize[4] = { latdims[0],
						latdims[1],
						latdims[2],
						latdims[3] };

	int t_bc = +1;
	double t_bcf = static_cast<double>(t_bc);
	double m_q = 0.01;
	double c_sw = 1.2;

	Geom fine_geometry(lattSize,
					    CLI.getBy(),
						CLI.getBz(),
						CLI.getNCores(),
						CLI.getSy(),
						CLI.getSz(),
						CLI.getPxy(),
						CLI.getPxyz(),
						CLI.getMinCt(),
						true);

	multi1d<LatticeColorMatrixF> u_f(Nd);
	multi1d<LatticeColorMatrix> u(Nd);
	for(int mu=0; mu < Nd; ++mu) {
		gaussian(u[mu]);
		reunit(u[mu]);
		u_f[mu] =  u[mu]; // Downcast to single prec
	}

	// Make the QPhiX Clover op
        MG::QDPWilsonCloverLinearOperator D_full(m_q, c_sw, t_bc, u);

	// Let us make the source 
	LatticeFermion source;
	gaussian(source);

	LatticeFermion solution;

	// Now let us prepare the source for QPhiX
	{
	  LatticeFermion transf_source = source;
	       
	  { 
	    LatticeFermion t1,t2;
	    D_full.M_ee_inv(t1, source, LINOP_OP);
	    D_full.M_oe( t2, t1, LINOP_OP);
   	    transf_source[ rb[ODD] ] -= t2;
           }
	
	   QPhiXCBSpinor qphix_source(fine_geometry);
   	   QPhiXCBSpinor qphix_solution(fine_geometry);
	   QPhiXCBGauge u_cb0(fine_geometry);
	   QPhiXCBGauge u_cb1(fine_geometry);
	   QPhiXCBClover A_oo(fine_geometry);
	   QPhiXCBClover A_inv_ee(fine_geometry);

	   // Now pack fields.
	   QPhiX::qdp_pack_gauge<>(u, u_cb0.get(),u_cb1.get(),fine_geometry);
	   QPhiX::qdp_pack_cb_spinor<>(transf_source, qphix_source.get(),fine_geometry, ODD);

	   QPhiX::qdp_pack_cb_spinor<>(solution, qphix_solution.get(),fine_geometry,ODD);

	   // Use clover from D_full
	   QPhiX::qdp_pack_clover<>(D_full.getClov(),A_oo.get(),fine_geometry,ODD);
	   QPhiX::qdp_pack_clover<>(D_full.getInvClov(),A_inv_ee.get(),fine_geometry, EVEN);

	   // Gropu the u-s into an array
	   Geom::SU3MatrixBlock *qphix_gauge[2] = { u_cb0.get(),u_cb1.get() };

	   //  ClovDslash calls an abort and there may
	   //  be several in scope. Needs fix in QPhiX
	   ClovOp QPhiXEOClov(qphix_gauge,
					   A_oo.get(),
					   A_inv_ee.get(),
					   &fine_geometry,
					   t_bcf,1.0,1.0);

	   // make a BiCGStab Solver
	   BiCGStab solver(QPhiXEOClov, 5000);


	   QPhiXCBSpinor::ValueType* soln[1] = { qphix_solution.get() };
   	   QPhiXCBSpinor::ValueType* rhs[1] = { qphix_source.get() };

	   int n_iters;
	   double rsd_sq_final;
	   unsigned long site_flops;
	   unsigned long mv_apps;

	   solver(soln,rhs, 1.0e-7, n_iters, rsd_sq_final,site_flops,mv_apps,1, ODD);

	   // Solution[odd] is the same between the transformed and untransformed system
	   QPhiX::qdp_unpack_cb_spinor<>(qphix_solution.get(),solution,fine_geometry,ODD);
	
	   // Solution[even] = M^{-1}_ee source_e - M^{-1}_ee M_eo solution_odd	
	   //                = M^{-1}_ee [ source_e - M_eo solution_odd ]
	   //
	   {
		LatticeFermion t1,t2;

	        // Fill t1 even	
		D_full.M_eo(t1,solution, LINOP_OP);

		t2[rb[EVEN]] = source-t1;
		D_full.M_ee_inv(solution,t2,LINOP_OP);
           }
	}

	// Check solution
	LatticeFermion tmp=QDP::zero;
	D_full(tmp, solution, LINOP_OP);
	LatticeFermion r = source - tmp;
	double r_norm_cb0 = toDouble(sqrt(norm2(r,rb[0])));
	double r_norm_cb1 = toDouble(sqrt(norm2(r,rb[1])));

	double r_rel_norm_cb0 = r_norm_cb0 / toDouble(sqrt(norm2(source, rb[0])));
	double r_rel_norm_cb1 = r_norm_cb1 / toDouble(sqrt(norm2(source, rb[1])));

	MasterLog(INFO, "CB 0 : || r || = %16.8e  || r ||/|| b ||=%16.8e", r_norm_cb0, r_rel_norm_cb0);
	MasterLog(INFO, "CB 1 : || r || = %16.8e  || r ||/|| b ||=%16.8e", r_norm_cb1, r_rel_norm_cb1);

        double r_rel_norm = toDouble(sqrt(norm2(r))/sqrt(norm2(source)));
	MasterLog(INFO, "Full: || r || / || b || = %16.8e", r_rel_norm);
	ASSERT_LT( r_rel_norm, 5.0e-7);
}

int main(int argc, char *argv[])
{
	return ::MGTesting::TestMain(&argc, argv);
}

