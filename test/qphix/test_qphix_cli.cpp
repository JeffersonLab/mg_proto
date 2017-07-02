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

	int t_bc = -1;
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

	multi1d<LatticeColorMatrixF> u(Nd);
	for(int mu=0; mu < Nd; ++mu) {
		gaussian(u[mu]);
		reunit(u[mu]);
	}

	CloverFermActParams p;
	p.Mass=Real(m_q);
	p.clovCoeffR=Real(c_sw);
	p.clovCoeffT=Real(c_sw);
	p.u0 = Real(1);
	p.anisoParam.anisoP = false;
	p.anisoParam.xi_0 = Real(1);
	p.anisoParam.nu = Real(1);
	p.anisoParam.t_dir =3;

	QDPCloverTermF clov;
	clov.create(u,p);
	QDPCloverTermF invclov;
	invclov.create(u,p,clov);
	for(int cb=0; cb < 2; ++cb) {
		invclov.choles(cb);
	}

	LatticeFermionF source;
	gaussian(source);

	LatticeFermionF transf_source = source;
	LatticeFermionF solution;


	{
		// transformed source is:
		// [ -D_oe A^{-1} source_even + source_odd ]
		LatticeFermionF tmp1, tmp2;
		invclov.apply(tmp1, source, LINOP_OP, EVEN); // tmp1[even] = A^{-1} src_even
		dslash(tmp2, u, tmp1, LINOP_OP, ODD);
		transf_source[rb[ODD]] -= tmp2;
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

	LatticeFermionF transf_solution = QDP::zero;
	QPhiX::qdp_pack_cb_spinor<>(transf_solution, qphix_solution.get(),fine_geometry,ODD);

	QPhiX::qdp_pack_clover<>(clov,A_oo.get(),fine_geometry,ODD);
	QPhiX::qdp_pack_clover<>(invclov,A_inv_ee.get(),fine_geometry, EVEN);

	Geom::SU3MatrixBlock *qphix_gauge[2] = { u_cb0.get(),u_cb1.get() };

	//  ClovDslash calls an abort and there may
	//  be several in scope. Needs fix in QPhiX
	ClovOp QPhiXEOClov(qphix_gauge,
					   A_oo.get(),
					   A_inv_ee.get(),
					   &fine_geometry,
					   t_bcf,1.0,1.0);
	BiCGStab solver(QPhiXEOClov, 5000);

	QPhiXCBSpinor::ValueType* soln[1] = { qphix_solution.get() };
	QPhiXCBSpinor::ValueType* rhs[1] = { qphix_source.get() };

	int n_iters;
	double rsd_sq_final;
	unsigned long site_flops;
	unsigned long mv_apps;

	solver(soln,rhs, 1.0e-7, n_iters, rsd_sq_final,site_flops,mv_apps,1, ODD);

	solution = zero;
	QPhiX::qdp_unpack_cb_spinor<>(qphix_solution.get(),solution,fine_geometry,ODD);

	// Need to reconstruct...
	{
		// For: tmp[even] = source_even - D_eo solution
		LatticeFermionF t1 = QDP::zero;
		dslash( t1, u, solution, LINOP_OP, EVEN);
		LatticeFermionF t2;
		t2[ rb[0] ] = source - t1;
		// Solution ODD is already set
		invclov.apply(solution, t2, 1, EVEN);
	}

	// Check solution
	multi1d<LatticeColorMatrix> u_full(Nd);
	for(int mu=0; mu < Nd; ++mu ) u_full[mu] = u[mu];

	MG::QDPWilsonCloverLinearOperator D_full(m_q, c_sw, t_bc, u_full );
	LatticeFermion sol_full = solution;
	LatticeFermion b = source;
	LatticeFermion tmp=QDP::zero;
	D_full(tmp, sol_full, LINOP_OP);
	LatticeFermion r = b - tmp;
	double r_norm_cb0 = toDouble(sqrt(norm2(r,rb[0])));
	double r_norm_cb1 = toDouble(sqrt(norm2(r,rb[1])));

	double r_rel_norm_cb0 = r_norm_cb0 / toDouble(sqrt(norm2(source, rb[0])));
	double r_rel_norm_cb1 = r_norm_cb1 / toDouble(sqrt(norm2(source, rb[1])));

	MasterLog(INFO, "CB 0 : || r || = %16.8e  || r ||/|| b ||=%16.8e", r_norm_cb0, r_rel_norm_cb0);
	MasterLog(INFO, "CB 1 : || r || = %16.8e  || r ||/|| b ||=%16.8e", r_norm_cb1, r_rel_norm_cb1);


}

int main(int argc, char *argv[])
{
	return ::MGTesting::TestMain(&argc, argv);
}

