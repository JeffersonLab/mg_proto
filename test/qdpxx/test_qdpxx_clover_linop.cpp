#include "gtest/gtest.h"
#include "../test_env.h"
#include "../mock_nodeinfo.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "qdpxx_helpers.h"
#include "reunit.h"
#include "wilson_clover_linear_operator.h"
#include "invmr.h"
#include "invbicgstab.h"
#include "invfgmres.h"

using namespace MG;
using namespace MGTesting;
using namespace QDP;

TEST(TestQDPXX, TestQDPXXCloverOpInvMR)
{
	IndexArray latdims={{4,4,4,4}};
	initQDPXXLattice(latdims);
	using Spinor = LatticeFermion;
	using Gauge = multi1d<LatticeColorMatrix>;

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

	// Create linear operator
	QDPWilsonCloverLinearOperator M(m_q, c_sw, t_bc,u);
	MRSolverParams params;
	params.MaxIter = 100;
	params.RsdTarget = 0.02;
	params.Omega = 1.1;
	params.VerboseP = true;
	MRSolver<Spinor,Gauge>  MR(M, params);
	Spinor b;
	gaussian(b);
	Spinor x=zero;

	QDPIO::cout << "|| b ||=  " << sqrt(norm2(b)) << std::endl;
	QDPIO::cout << "|| x || = " << sqrt(norm2(x)) << std::endl;

	LinearSolverResults res = MR(x,b);
	QDPIO::cout << "MR Solver Took: " << res.n_count << " iterations"
			<< std::endl;
	ASSERT_LT( res.resid, 0.02);
	ASSERT_EQ( res.resid_type, RELATIVE);

	params.MaxIter = 6;
	MRSmoother<Spinor,Gauge> MRS(M,params);
	x=zero;
	MRS(x,b);
	// Compute residuum:
	Spinor r=zero;
	M(r,x,LINOP_OP);
	r -= b;
	double resid = toDouble(sqrt(norm2(r)/norm2(b)));
	QDPIO::cout << "After 6 iterations we have: ||r||/||b||=" << resid << std::endl;
	ASSERT_LT( resid, 0.02);


}

TEST(TestQDPXX, TestQDPXXCloverOpInvBiCGStab)
{
	IndexArray latdims={{4,4,4,4}};
	initQDPXXLattice(latdims);
	using Spinor = LatticeFermion;
	using Gauge = multi1d<LatticeColorMatrix>;

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

	// Create linear operator
	QDPWilsonCloverLinearOperator M(m_q, c_sw, t_bc,u);
	LinearSolverParamsBase params;
	params.MaxIter = 500;
	params.RsdTarget = 1.0e-5;
	params.VerboseP = true;
	BiCGStabSolver<Spinor,Gauge>  BiCGStab(M, params);
	Spinor b;
	gaussian(b);
	Spinor x=zero;

	QDPIO::cout << "|| b ||=  " << sqrt(norm2(b)) << std::endl;
	QDPIO::cout << "|| x || = " << sqrt(norm2(x)) << std::endl;

	LinearSolverResults res = BiCGStab(x,b);
	QDPIO::cout << "BiCGStab Solver Took: " << res.n_count << " iterations"
			<< std::endl;
	ASSERT_EQ(res.resid_type, RELATIVE);
	ASSERT_LT(res.resid, 5.75e-6);
}


TEST(TestQDPXX, TestQDPXXCloverOpInvBiCGStabZeroRHS)
{
	IndexArray latdims={{4,4,4,4}};
	initQDPXXLattice(latdims);
	using Spinor = LatticeFermion;
	using Gauge = multi1d<LatticeColorMatrix>;

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

	// Create linear operator
	QDPWilsonCloverLinearOperator M(m_q, c_sw, t_bc,u);
	LinearSolverParamsBase params;
	params.MaxIter = 500;
	params.RsdTarget = 1.0e-5;
	params.VerboseP = true;
	BiCGStabSolver<Spinor,Gauge>  BiCGStab(M, params);

	// Special Test to solver for Zero RHS for nullspace
	Spinor b=zero;
	const int Nvec = 6;
	multi1d<Spinor> v(Nvec);

	for(int mu=0; mu < Nd; ++mu) {
		gaussian(v[mu]);

		LinearSolverResults res = BiCGStab(v[mu],b, ABSOLUTE);
		QDPIO::cout << "BiCGStab Solver Took: " << res.n_count << " iterations"
				<< std::endl;

		ASSERT_LT(res.resid, 1.0e-5);
		ASSERT_EQ(res.resid_type, ABSOLUTE);
	}




}

TEST(TestQDPXX, TestQDPXXCloverOpInvUnprecFGMRES)
{
	IndexArray latdims={{4,4,4,4}};
	initQDPXXLattice(latdims);
	using Spinor = LatticeFermion;
	using Gauge = multi1d<LatticeColorMatrix>;

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

	// Create linear operator
	QDPWilsonCloverLinearOperator M(m_q, c_sw, t_bc,u);
	FGMRESParams params;
	params.MaxIter = 500;
	params.RsdTarget = 1.0e-5;
	params.VerboseP = true;
	params.NKrylov = 10;

	FGMRESSolver FGMRES(M, params,nullptr);
	Spinor b;
	gaussian(b);
	Spinor x=zero;

	QDPIO::cout << "|| b ||=  " << sqrt(norm2(b)) << std::endl;
	QDPIO::cout << "|| x || = " << sqrt(norm2(x)) << std::endl;

	LinearSolverResults res = FGMRES(x,b);
	QDPIO::cout << "FGMRES Solver Took: " << res.n_count << " iterations"
			<< std::endl;
	ASSERT_EQ(res.resid_type, RELATIVE);
	ASSERT_LT(res.resid, 9e-6);
}

TEST(TestQDPXX, TestQDPXXCloverOpInvRightPrecFGMRES)
{
	IndexArray latdims={{4,4,4,4}};
	initQDPXXLattice(latdims);
	using Spinor = LatticeFermion;
	using Gauge = multi1d<LatticeColorMatrix>;

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

	// Create linear operator
	QDPWilsonCloverLinearOperator M(m_q, c_sw, t_bc,u);

	// Create a simple MR Preconditioner
	MRSolverParams mr_params;
	mr_params.Omega = 1.1;
	mr_params.RsdTarget = 1.0e-1;
	mr_params.MaxIter = 20;
	mr_params.VerboseP = true;

	MRSolver<Spinor,Gauge> MR(M, mr_params);

	FGMRESParams params;
	params.MaxIter = 500;
	params.RsdTarget = 1.0e-5;
	params.VerboseP = true;
	params.NKrylov = 10;

	FGMRESSolver  FGMRES(M, params,&MR);
	Spinor b;
	gaussian(b);
	Spinor x=zero;

	QDPIO::cout << "|| b ||=  " << sqrt(norm2(b)) << std::endl;
	QDPIO::cout << "|| x || = " << sqrt(norm2(x)) << std::endl;

	LinearSolverResults res = FGMRES(x,b);
	QDPIO::cout << "FGMRES Solver Took: " << res.n_count << " iterations"
			<< std::endl;
	ASSERT_EQ(res.resid_type, RELATIVE);
	ASSERT_LT(res.resid, 9e-6);
}

int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

