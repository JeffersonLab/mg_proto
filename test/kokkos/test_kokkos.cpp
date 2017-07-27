#include "gtest/gtest.h"
#include "../test_env.h"
#include "../mock_nodeinfo.h"
#include "../qdpxx/qdpxx_utils.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/fine_qdpxx/dslashm_w.h"
#include "./kokkos_types.h"
#include "./kokkos_qdp_utils.h"
#include "./kokkos_spinproj.h"
#include "./kokkos_matvec.h"
#include "./kokkos_dslash.h"

using namespace MG;
using namespace MGTesting;
using namespace QDP;

TEST(TestKokkos, TestLatticeInitialization)
{
	IndexArray latdims={{8,8,8,8}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;


}


TEST(TestKokkos, TestSpinorInitialization)
{
	IndexArray latdims={{8,8,8,8}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	LatticeInfo info(latdims,4,3,NodeInfo());

	KokkosCBFineSpinor<Kokkos::complex<float>,4> cb_spinor_e(info, EVEN);
	KokkosCBFineSpinor<Kokkos::complex<float>,4> cb_spinor_o(info, ODD);

	KokkosCBFineGaugeField<Kokkos::complex<float>> gauge_field_even(info, EVEN);
	KokkosCBFineGaugeField<Kokkos::complex<float>> gauge_field_odd(info,ODD);
 }

TEST(TestKokkos, TestQDPCBSpinorImportExport)
{
	IndexArray latdims={{4,6,8,10}};
	initQDPXXLattice(latdims);

	LatticeFermion qdp_out;
	LatticeFermion qdp_in;

	gaussian(qdp_in);

	LatticeInfo info(latdims,4,3,NodeInfo());
	KokkosCBFineSpinor<Kokkos::complex<REAL>,4>  kokkos_spinor_e(info, EVEN);
	KokkosCBFineSpinor<Kokkos::complex<REAL>,4>  kokkos_spinor_o(info, ODD);
	{
		qdp_out = zero;
		// Import Checkerboard, by checkerboard
		QDPLatticeFermionToKokkosCBSpinor<REAL,LatticeFermion>(qdp_in, kokkos_spinor_e);
		// Export back out
		KokkosCBSpinorToQDPLatticeFermion<REAL,LatticeFermion>(kokkos_spinor_e,qdp_out);
		qdp_out[rb[0]] -= qdp_in;

		// Elements of QDP_out should now be zero.
		double norm_diff = toDouble(sqrt(norm2(qdp_out)));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);
	}

	{
		qdp_out = zero;
		QDPLatticeFermionToKokkosCBSpinor(qdp_in, kokkos_spinor_o);
		// Export back out
		KokkosCBSpinorToQDPLatticeFermion(kokkos_spinor_o,qdp_out);

		qdp_out[rb[1]] -= qdp_in;
		double norm_diff = toDouble(sqrt(norm2(qdp_out)));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);
	}

}

TEST(TestKokkos, TestQDPCBHalfSpinorImportExport)
{
	IndexArray latdims={{4,6,8,10}};
	initQDPXXLattice(latdims);

	LatticeHalfFermion qdp_out;
	LatticeHalfFermion qdp_in;

	gaussian(qdp_in);

	LatticeInfo info(latdims,2,3,NodeInfo());
	KokkosCBFineSpinor<Kokkos::complex<REAL>,2>  kokkos_hspinor_e(info, EVEN);
	KokkosCBFineSpinor<Kokkos::complex<REAL>,2>  kokkos_hspinor_o(info, ODD);
	{
		qdp_out = zero;
		// Import Checkerboard, by checkerboard
		QDPLatticeHalfFermionToKokkosCBSpinor2(qdp_in, kokkos_hspinor_e);
		// Export back out
		KokkosCBSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_e,qdp_out);
		qdp_out[rb[0]] -= qdp_in;

		// Elements of QDP_out should now be zero.
		double norm_diff = toDouble(sqrt(norm2(qdp_out)));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);
	}

	{
		qdp_out = zero;
		QDPLatticeHalfFermionToKokkosCBSpinor2(qdp_in, kokkos_hspinor_o);
		// Export back out
		KokkosCBSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_o,qdp_out);

		qdp_out[rb[1]] -= qdp_in;
		double norm_diff = toDouble(sqrt(norm2(qdp_out)));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);
	}

}

TEST(TestKokkos, TestSpinProject)
{
	IndexArray latdims={{4,2,2,4}};
	initQDPXXLattice(latdims);

	LatticeInfo info(latdims,4,3,NodeInfo());
	LatticeInfo hinfo(latdims,2,3,NodeInfo());


	LatticeFermion qdp_in;
	LatticeHalfFermion qdp_out;
	LatticeHalfFermion kokkos_out;


	gaussian(qdp_in);
	KokkosCBFineSpinor<Kokkos::complex<REAL>,4> kokkos_in(info,EVEN);
	KokkosCBFineSpinor<Kokkos::complex<REAL>,2> kokkos_hspinor_out(hinfo,EVEN);

	QDPLatticeFermionToKokkosCBSpinor(qdp_in, kokkos_in);

	for(int sign = -1; sign <= +1; sign+=2 ) {
		for(int dir=0; dir < 4; ++dir ) {

			MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",dir,sign);

			if( sign == -1 ) {
				switch(dir) {
				case 0:
					qdp_out[rb[0]] = spinProjectDir0Minus(qdp_in);
					break;
				case 1:
					qdp_out[rb[0]] = spinProjectDir1Minus(qdp_in);
					break;
				case 2:
					qdp_out[rb[0]] = spinProjectDir2Minus(qdp_in);
					break;
				case 3:
					qdp_out[rb[0]] = spinProjectDir3Minus(qdp_in);
					break;
				default:
					MasterLog(ERROR, "Wrong direction in SpinProject test: %d",dir);
					break;
				};
			}
			else {
				switch(dir) {
				case 0:
					qdp_out[rb[0]] = spinProjectDir0Plus(qdp_in);
					break;
				case 1:
					qdp_out[rb[0]] = spinProjectDir1Plus(qdp_in);
					break;
				case 2:
					qdp_out[rb[0]] = spinProjectDir2Plus(qdp_in);
					break;
				case 3:
					qdp_out[rb[0]] = spinProjectDir3Plus(qdp_in);
					break;
				default:
					MasterLog(ERROR, "Wrong direction in SpinProject test: %d",dir);
				};
			}
			qdp_out[rb[1]] = zero;

			KokkosProjectLattice(kokkos_in,dir,sign,kokkos_hspinor_out);

			// Export back out
			KokkosCBSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_out,kokkos_out);

			qdp_out[rb[0]] -= kokkos_out;

			double norm_diff = toDouble(sqrt(norm2(qdp_out)));
			MasterLog(INFO, "norm_diff = %lf", norm_diff);
			ASSERT_LT( norm_diff, 1.0e-5);
		}
	}
}

TEST(TestKokkos, TestSpinRecons)
{
	IndexArray latdims={{4,2,2,4}};
	initQDPXXLattice(latdims);

	LatticeInfo info(latdims,4,3,NodeInfo());
	LatticeInfo hinfo(latdims,2,3,NodeInfo());

	LatticeHalfFermion qdp_in;
	LatticeFermion     qdp_out;
	LatticeFermion     kokkos_out;


	gaussian(qdp_in);

	KokkosCBFineSpinor<Kokkos::complex<REAL>,2> kokkos_hspinor_in(hinfo,EVEN);
	KokkosCBFineSpinor<Kokkos::complex<REAL>,4> kokkos_spinor_out(info,EVEN);

	QDPLatticeHalfFermionToKokkosCBSpinor2(qdp_in, kokkos_hspinor_in);

	for(int s=-1; s <= +1; s+=2 ) {
		for(int dir=0; dir < 4; ++dir ) {
			MasterLog(INFO, "Spin Recons Test: dir = %d sign = %d", dir, s);

			if ( s == -1 ) {
				switch(dir) {
				case 0:
					qdp_out[rb[0]] = spinReconstructDir0Minus(qdp_in);
					break;
				case 1:
					qdp_out[rb[0]] = spinReconstructDir1Minus(qdp_in);
					break;
				case 2:
					qdp_out[rb[0]] = spinReconstructDir2Minus(qdp_in);
					break;
				case 3:
					qdp_out[rb[0]] = spinReconstructDir3Minus(qdp_in);
					break;
				default:
					MasterLog(ERROR, "Bad direction in SpinReconstruction Test");
					break;
				}
			}
			else {
				switch(dir) {
				case 0:
					qdp_out[rb[0]] = spinReconstructDir0Plus(qdp_in);
					break;
				case 1:
					qdp_out[rb[0]] = spinReconstructDir1Plus(qdp_in);
					break;
				case 2:
					qdp_out[rb[0]] = spinReconstructDir2Plus(qdp_in);
					break;
				case 3:
					qdp_out[rb[0]] = spinReconstructDir3Plus(qdp_in);
					break;
				default:
					MasterLog(ERROR, "Bad direction in SpinReconstruction Test");
					break;
				}

			}


			qdp_out[rb[1]] = zero;

			KokkosReconsLattice(kokkos_hspinor_in,dir,s,kokkos_spinor_out);
			KokkosCBSpinorToQDPLatticeFermion(kokkos_spinor_out,kokkos_out);

			qdp_out[rb[0]] -= kokkos_out;

			double norm_diff = toDouble(sqrt(norm2(qdp_out)));
			MasterLog(INFO, "norm_diff = %lf", norm_diff);
			ASSERT_LT( norm_diff, 1.0e-5);
		} // dir
 	} // s
}




TEST(TestKokkos, TestQDPCBGaugeFIeldImportExport)
{
	IndexArray latdims={{4,4,4,4}};
	initQDPXXLattice(latdims);

	multi1d<LatticeColorMatrix> gauge_in(n_dim);
	for(int mu=0; mu < n_dim; ++mu) {
		gaussian(gauge_in[mu]);
	}

	multi1d<LatticeColorMatrix> gauge_out;  // Basic uninitialized

	LatticeInfo info(latdims,4,3,NodeInfo());
	KokkosCBFineGaugeField<Kokkos::complex<REAL>>  kokkos_gauge_e(info, EVEN);
	KokkosCBFineGaugeField<Kokkos::complex<REAL>>  kokkos_gauge_o(info, ODD);
	{
		// Import Checkerboard, by checkerboard
		QDPGaugeFieldToKokkosCBGaugeField(gauge_in, kokkos_gauge_e);
		// Export back out
		KokkosCBGaugeFieldToQDPGaugeField(kokkos_gauge_e, gauge_out);

		for(int mu=0; mu < n_dim; ++mu) {
			(gauge_out[mu])[rb[0]] -= gauge_in[mu];

			// In this test, the copy back initialized gauge_out
			// so its non-checkerboard piece is junk
			// Take norm over only the checkerboard of interest
			double norm_diff = toDouble(sqrt(norm2(gauge_out[mu], rb[0])));
			MasterLog(INFO, "norm_diff[%d] = %lf", mu, norm_diff);
			ASSERT_LT( norm_diff, 1.0e-5);
		}
	}

		{
		// gauge out is now allocated, so zero it
		for(int mu=0; mu < n_dim; ++mu) {
			gauge_out[mu] = zero;
		}
		QDPGaugeFieldToKokkosCBGaugeField(gauge_in, kokkos_gauge_o);
		// Export back out
		KokkosCBGaugeFieldToQDPGaugeField(kokkos_gauge_o, gauge_out);

		for(int mu=0; mu < n_dim; ++mu) {
			(gauge_out[mu])[rb[1]] -= gauge_in[mu];
			// Other checkerboard was zeroed initially.

			double norm_diff = toDouble(sqrt(norm2(gauge_out[mu])));
			MasterLog(INFO, "norm_diff[%d] = %lf", mu, norm_diff);
			ASSERT_LT( norm_diff, 1.0e-5);
		}
	}

}

TEST(TestKokkos, TestQDPGaugeFIeldImportExport)
{
	IndexArray latdims={{4,4,4,4}};
	initQDPXXLattice(latdims);

	multi1d<LatticeColorMatrix> gauge_in(n_dim);
	for(int mu=0; mu < n_dim; ++mu) {
		gaussian(gauge_in[mu]);
	}

	multi1d<LatticeColorMatrix> gauge_out;  // Basic uninitialized

	LatticeInfo info(latdims,4,3,NodeInfo());
	KokkosFineGaugeField<Kokkos::complex<REAL>>  kokkos_gauge(info);
	{
		// Import Checkerboard, by checkerboard
		QDPGaugeFieldToKokkosGaugeField(gauge_in, kokkos_gauge);
		// Export back out
		KokkosGaugeFieldToQDPGaugeField(kokkos_gauge, gauge_out);

		for(int mu=0; mu < n_dim; ++mu) {
			(gauge_out[mu]) -= gauge_in[mu];

			// In this test, the copy back initialized gauge_out
			// so its non-checkerboard piece is junk
			// Take norm over only the checkerboard of interest
			double norm_diff = toDouble(sqrt(norm2(gauge_out[mu])));
			MasterLog(INFO, "norm_diff[%d] = %lf", mu, norm_diff);
			ASSERT_LT( norm_diff, 1.0e-5);
		}
	}

}


TEST(TestKokkos, TestMultHalfSpinor)
{
	IndexArray latdims={{4,4,4,4}};
	initQDPXXLattice(latdims);
	multi1d<LatticeColorMatrix> gauge_in(n_dim);
	for(int mu=0; mu < n_dim; ++mu) {
		gaussian(gauge_in[mu]);
	}

	LatticeHalfFermion psi_in;
	gaussian(psi_in);

	LatticeInfo info(latdims,4,3,NodeInfo());
	LatticeInfo hinfo(latdims,2,3,NodeInfo());

	KokkosCBFineSpinor<Kokkos::complex<REAL>,2> kokkos_hspinor_in(hinfo,EVEN);
	KokkosCBFineSpinor<Kokkos::complex<REAL>,2> kokkos_hspinor_out(hinfo,EVEN);
	KokkosCBFineGaugeField<Kokkos::complex<REAL>>  kokkos_gauge_e(info, EVEN);


	// Import Gauge Field
	QDPGaugeFieldToKokkosCBGaugeField(gauge_in, kokkos_gauge_e);

	LatticeHalfFermion psi_out, kokkos_out;
	MasterLog(INFO, "Testing MV");
	{
		psi_out[rb[0]] = gauge_in[0]*psi_in;
		psi_out[rb[1]] = zero;


		// Import Gauge Field
		QDPGaugeFieldToKokkosCBGaugeField(gauge_in, kokkos_gauge_e);
		// Import input halfspinor
		QDPLatticeHalfFermionToKokkosCBSpinor2(psi_in, kokkos_hspinor_in);

		KokkosMVLattice(kokkos_gauge_e, kokkos_hspinor_in, 0, kokkos_hspinor_out);

		// Export result HalfFermion
		KokkosCBSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_out,kokkos_out);
		psi_out[rb[0]] -= kokkos_out;
		double norm_diff = toDouble(sqrt(norm2(psi_out)));

		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);

	}

	// ********* TEST ADJOINT ****************
	{

		psi_out[rb[0]] = adj(gauge_in[0])*psi_in;
		psi_out[rb[1]] = zero;

		// Import input halfspinor -- Gauge still imported

		QDPLatticeHalfFermionToKokkosCBSpinor2(psi_in, kokkos_hspinor_in);
		KokkosHVLattice(kokkos_gauge_e, kokkos_hspinor_in, 0, kokkos_hspinor_out);

		// Export result HalfFermion
		KokkosCBSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_out,kokkos_out);
		psi_out[rb[0]] -= kokkos_out;

		double norm_diff = toDouble(sqrt(norm2(psi_out)));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);

	}
}


TEST(TestKokkos, TestDslash)
{
	IndexArray latdims={{4,4,4,4}};
	initQDPXXLattice(latdims);
	multi1d<LatticeColorMatrix> gauge_in(n_dim);
	for(int mu=0; mu < n_dim; ++mu) {
		gaussian(gauge_in[mu]);
		reunit(gauge_in[mu]);
	}

	LatticeFermion psi_in=zero;
	gaussian(psi_in);

	LatticeInfo info(latdims,4,3,NodeInfo());
	LatticeInfo hinfo(latdims,2,3,NodeInfo());

	KokkosCBFineSpinor<Kokkos::complex<REAL>,4> kokkos_spinor_even(info,EVEN);
	KokkosCBFineSpinor<Kokkos::complex<REAL>,4> kokkos_spinor_odd(info,ODD);
	KokkosFineGaugeField<Kokkos::complex<REAL>>  kokkos_gauge(info);


	// Import Gauge Field
	QDPGaugeFieldToKokkosGaugeField(gauge_in, kokkos_gauge);
	KokkosDslash<REAL> D(info);

	LatticeFermion psi_out, kokkos_out;
	for(int cb=0; cb < 2; ++cb) {
		KokkosCBFineSpinor<Kokkos::complex<REAL>,4>& out_spinor = (cb == EVEN) ? kokkos_spinor_even : kokkos_spinor_odd;
		KokkosCBFineSpinor<Kokkos::complex<REAL>,4>& in_spinor = (cb == EVEN) ? kokkos_spinor_odd: kokkos_spinor_even;

		for(int isign=-1; isign < 2; isign+=2) {

			// In the Host
			psi_out = zero;

			// Target cb=1 for now.
			dslash(psi_out,gauge_in,psi_in,isign,cb);

			QDPLatticeFermionToKokkosCBSpinor(psi_in, in_spinor);


			D(in_spinor,kokkos_gauge,out_spinor,isign);

			kokkos_out = zero;
			KokkosCBSpinorToQDPLatticeFermion(out_spinor, kokkos_out);

			MasterLog(DEBUG,"After export kokkos_out has norm = %16.8e on rb[0]", toDouble(sqrt(norm2(kokkos_out,rb[0]))));
			MasterLog(DEBUG,"After export psi_out has norm = %16.8e on rb[0]", toDouble(sqrt(norm2(psi_out,rb[0]))));

			MasterLog(DEBUG,"After export kokkos_out has norm = %16.8e on rb[1]", toDouble(sqrt(norm2(kokkos_out,rb[1]))));
			MasterLog(DEBUG,"After export psi_out has norm = %16.8e on rb[1]", toDouble(sqrt(norm2(psi_out,rb[1]))));

			// Check Diff on Odd
			psi_out[rb[cb]] -= kokkos_out;
			double norm_diff = toDouble(sqrt(norm2(psi_out,rb[cb])));

			MasterLog(INFO, "norm_diff = %lf", norm_diff);
			ASSERT_LT( norm_diff, 1.0e-5);
		}
	}
}



int main(int argc, char *argv[]) 
{
	return ::MGTesting::TestMain(&argc, argv);
}

