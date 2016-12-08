#include "gtest/gtest.h"
#include "../test_env.h"
#include "../mock_nodeinfo.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "qdpxx_helpers.h"
#include "reunit.h"
#include "transf.h"
#include "clover_fermact_params_w.h"
#include "clover_term_qdp_w.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_op.h"
#include "dslashm_w.h"

using namespace MG;
using namespace MGTesting;
using namespace QDP;

// Apply a single direction of Dslash
void DslashDir(LatticeFermion& out, const multi1d<LatticeColorMatrix>& u, const LatticeFermion& in, int dir)
{
	switch(dir) {
	case 0: // Dir 0, Forward
		out = spinReconstructDir0Minus(u[0] * shift(spinProjectDir0Minus(in), FORWARD, 0));
		break;
	case 1: // Dir 0, Backward
		out = spinReconstructDir0Plus(shift(adj(u[0]) * spinProjectDir0Plus(in), BACKWARD, 0));
		break;
	case 2: // Dir 1, Forward
		out = spinReconstructDir1Minus(u[1] * shift(spinProjectDir1Minus(in), FORWARD, 1));
		break;
	case 3: // Dir 1, Backward
		out = spinReconstructDir1Plus(shift(adj(u[1]) * spinProjectDir1Plus(in), BACKWARD, 1));
		break;
	case 4: // Dir 2, Forward
		out = spinReconstructDir2Minus(u[2] * shift(spinProjectDir2Minus(in), FORWARD, 2));
		break;
	case 5: // Dir 2, Backward
		out = spinReconstructDir2Plus(shift(adj(u[2]) * spinProjectDir2Plus(in), BACKWARD, 2));
		break;
	case 6: // Dir 3, Forward,
		out = spinReconstructDir3Minus(u[3] * shift(spinProjectDir3Minus(in), FORWARD, 3));
		break;
	case 7: // Dir 3, Backward
		out = spinReconstructDir3Plus(shift(adj(u[3]) * spinProjectDir3Plus(in), BACKWARD, 3));
		break;
	default:
		QDPIO::cerr<< "Unknown direction. You oughtnt call this" << std::endl;
		QDP_abort(1);
	}
}

// In the full blown version, instead of the 'site' we will sum over sites in
// a block
void axAggr(const double alpha, LatticeFermion& v, int site, int aggr)
{
	for(int spin=aggr*(Ns/2); spin < (aggr+1)*(Ns/2); ++spin) {
		for(int color=0; color < Nc; ++color) {
			v.elem(site).elem(spin).elem(color).real() *= alpha;
			v.elem(site).elem(spin).elem(color).imag() *= alpha;
		}
	}
}

// In the full blown version, instead of the 'site' we will sum over sites in
// a block
void caxpyAggr(const std::complex<double>& alpha, const LatticeFermion& x, LatticeFermion& y, int site, int aggr)
{
	for(int spin=aggr*(Ns/2); spin < (aggr+1)*(Ns/2); ++spin) {
		for(int color=0; color < Nc; ++color) {
			double ar = real(alpha);
			double ai = imag(alpha);
			double xr = x.elem(site).elem(spin).elem(color).real();
			double xi = x.elem(site).elem(spin).elem(color).imag();

			y.elem(site).elem(spin).elem(color).real() += ar*xr - ai*xi;
			y.elem(site).elem(spin).elem(color).imag() += ar*xi + ai*xr;

		}
	}
}

// In the full blown version, instead of the 'site' we will sum over sites in
// a block
double norm2Aggregate(const LatticeFermion& v, int site, int aggr)
{
	double norm2 = (double)0;
	for(int spin=aggr*(Ns/2); spin < (aggr+1)*(Ns/2); ++spin) {
		for(int color=0; color < Nc; ++color) {
			norm2 += v.elem(site).elem(spin).elem(color).real() * v.elem(site).elem(spin).elem(color).real();
			norm2 += v.elem(site).elem(spin).elem(color).imag() * v.elem(site).elem(spin).elem(color).imag();
		}
	}

	return norm2;
}

// In the full blown version, instead of the 'site' we will sum over sites in
// a block
std::complex<double>
innerProductAggr(const LatticeFermion& left, const LatticeFermion& right, int site, int aggr)
{
	std::complex<double> iprod;
	double re=0;
	double im=0;

	for(int spin=aggr*(Ns/2); spin < (aggr+1)*(Ns/2); ++spin) {
		for(int color=0; color < Nc; ++color) {
			REAL left_r = left.elem(site).elem(spin).elem(color).real();
			REAL left_i = left.elem(site).elem(spin).elem(color).imag();

			REAL right_r = right.elem(site).elem(spin).elem(color).real();
			REAL right_i = right.elem(site).elem(spin).elem(color).imag();
			re += left_r*right_r + left_i*right_i;
			im += left_r*right_i - left_i*right_r;
		}
	}

	iprod=std::complex<double>(re,im);
	return iprod;
}

void normalizeAggregates(multi1d<LatticeFermion>& vecs)
{
	for(int aggr=0; aggr < 2; ++aggr) {
#pragma omp parallel for
			// This will be over blocks...
			for(int site=all.start(); site <= all.end(); ++site) {


				// do vecs[0] ... vecs[N]
				for(int curr_vec=0; curr_vec < vecs.size(); curr_vec++) {

					// orthogonalize against previous vectors
					// if curr_vec == 0 this will be skipped
					for(int prev_vec=0; prev_vec < curr_vec; prev_vec++) {
						QDPIO::cout << "Orthogonalizing vector " << curr_vec << " against vector " << prev_vec << std::endl;
						std::complex<double> iprod = innerProductAggr( vecs[curr_vec], vecs[prev_vec], site, aggr);
						std::complex<double> minus_iprod = -iprod;

						// curr_vec = curr_vec - iprod*prev_vec = -iprod*prev_vec + curr_vec
						caxpyAggr( minus_iprod, vecs[prev_vec], vecs[curr_vec], site, aggr);
					}

					// Normalize current vector
					double inv_norm = ((double)1)/sqrt(norm2Aggregate(vecs[curr_vec], site, aggr));

					// vecs[curr_vec] = inv_norm * vecs[curr_vec]
					axAggr(inv_norm, vecs[curr_vec], site, aggr);
				}


			}	// site
	}// aggregates
}

//  prop_out(x) = prop_in^\dagger(x) ( 1 +/- gamma_mu ) U mu(x) prop_in(x + mu)
//
//  This routine is used to test Coarse Dslash
//  Initially, prop_in should be just I_{12x12}
//  However, any unitary matrix on the sites (orthonormal basis) would do
void dslashTripleProduct12x12SiteDir(int dir, const multi1d<LatticeColorMatrix>& u, const LatticePropagator& in_prop, LatticePropagator& out_prop)
{
	LatticeFermion in, out;
	LatticePropagator prop_tmp=zero;
	// Loop through spins and colors

	for(int spin=0; spin < 4; ++spin ) {
		for(int color=0; color < 3; ++color ) {

			// Extract component into 'in'
			PropToFerm(in_prop,in,color,spin);

			// Apply Dlsash in that Direction
			DslashDir(out, u, in, dir);

			// Place back into prop
			FermToProp(out, prop_tmp, color, spin);
		} // color
	} // spin

	// Technically I don't need this last part, since for this test
	// in_prop is the identity.
	//
	// However, if in_prop was some 12x12 unitary matrix (basis rotation)
	// This part would be necessary.
	// And actually that would constitute a useful test.
	out_prop=adj(in_prop)*prop_tmp;

}


void extractAggregate(LatticeFermion& target, const LatticeFermion& src, int aggr )
{
	for(int spin=aggr*Ns/2; spin < (aggr+1)*Ns/2; ++spin) {
		pokeSpin(target, peekSpin(src,spin) , spin);
	}
}



//  prop_out(x) = prop_in^\dagger(x) ( 1 +/- gamma_mu ) U mu(x) prop_in(x + mu)
//
//  This routine is used to test Coarse Dslash
//  Initially, prop_in should be just I_{12x12}
//  However, any unitary matrix on the sites (orthonormal basis) would do
void dslashTripleProductSiteDir(int dir, const multi1d<LatticeColorMatrix>& u, const multi1d<LatticeFermion>& in_vecs, CoarseGauge& u_coarse)
{


	int Ncolor_c = u_coarse.GetNumColor();
	int Ncolorspin_c = u_coarse.GetNumColorSpin();
	int Naggr_c = Ncolorspin_c/Ncolor_c;

	QDPIO::cout << "Ncolor_c=" << Ncolor_c << " Ncolorspin_c=" << Ncolorspin_c << " Nc*Ns=" << Nc*Ns << std::endl;

	// in vecs has size Ncolor_c = Ncolorspin_c/2
	// But this mixes both upper and lower spins
	// Once we deal with those separately we will need Ncolorspin_c results
	// And we will need to apply the 'DslashDir' separately to each aggregate

	assert( in_vecs.size() == Ncolor_c);

	multi1d<LatticeFermion> out_vecs( Ncolorspin_c );

	for(int j=0; j < Ncolorspin_c; ++j) {
		out_vecs[j]=zero;
	}


	// Apply DslashDir to each aggregate separately.
	// DslashDir may mix spins with (1 +/- gamma_mu)
	for(int j=0; j < Ncolor_c; ++j) {
		for(int aggr=0; aggr < Naggr_c; ++aggr) {
			LatticeFermion tmp=zero;
			extractAggregate(tmp, in_vecs[j], aggr);
			DslashDir(out_vecs[aggr*Ncolor_c+j], u, tmp, dir);
		}
	}


	// So now we should have 12 out vecs Which contain the upper and lower aggregates
	// The first 6 should have upper aggregation components, the second 6 should have the lower ones.
	IndexType coarse_cbsites = u_coarse.GetInfo().GetNumCBSites();


	// Technically these outer loops should be over all the blocks.
	for(int cb=0; cb < 2; ++cb) {
		for(int cbsite=0; cbsite < coarse_cbsites; ++cbsite) {

			// This is now an Ncomplex*NumColorspin*NumColorspin array
			float *coarse_link = u_coarse.GetSiteDirDataPtr(cb,cbsite,dir);
			int site=rb[cb].siteTable()[cbsite];

			for(int aggr_row=0; aggr_row < Naggr_c; ++aggr_row) {
				for(int aggr_col=0; aggr_col < Naggr_c; ++aggr_col ) {

					// This is an Ncolor_c x Ncolor_c matmul
					for(int matmul_row=0; matmul_row < Ncolor_c; ++matmul_row) {
						for(int matmul_col=0; matmul_col < Ncolor_c; ++matmul_col) {

							// Offset by the aggr_row and aggr_column
							int row = aggr_row*Ncolor_c + matmul_row;
							int col = aggr_col*Ncolor_c + matmul_col;

							//Index in coarse link
							int coarse_link_index = n_complex*(col+ Ncolorspin_c*row);

							// Init inner product
							coarse_link[ RE + coarse_link_index ] = 0;
							coarse_link[ IM + coarse_link_index ] = 0;

							// Inner product loop
							for(int k=0; k < Ncolor_c; ++k) {

								// [ V^H_upper   0      ] [  A_upper    B_upper ] = [ V^H_upper A_upper   V^H_upper B_upper  ]
								// [  0       V^H_lower ] [  A_lower    B_lower ]   [ V^H_lower A_lower   V^H_lower B_lower  ]
								//
								// NB: V^H_upper always multiplies an 'upper' (either A_upper or B_upper)
								//     V^H_lower always multiplies a 'lower' (either A_lower or B_lower)
								//
								// So there is no mixing of spins (ie: V^H_upper with B_lower etc)
								// So spins are decided by which portion of V^H we use, ie on aggr_row
								//
								// k / Nc maps to spin_component 0 or 1 in the aggregation
								// aggr_row*(Ns/2) offsets it to either upper or lower
								//
								int spin=k/Nc+aggr_row*(Ns/2);

								// k % Nc maps to color component (0,1,2)
								int color=k%Nc;

								// Right vector
								REAL right_r = out_vecs[col].elem(site).elem(spin).elem(color).real();
								REAL right_i = out_vecs[col].elem(site).elem(spin).elem(color).imag();

								// Left vector -- only Ncolor_c components with [ V^H_upper V^H_lower ]
								//
								// ie a compact storage
								// rather than:
								//
								// [ V^H_upper   0      ]
								// [  0       V^H_lower ]
								//
								// so index with row % Ncolor_c = matmul_row
								REAL left_r = in_vecs[matmul_row].elem(site).elem(spin).elem(color).real();
								REAL left_i = in_vecs[matmul_row].elem(site).elem(spin).elem(color).imag();

								// Accumulate inner product V^H_row A_column
								coarse_link[RE + coarse_link_index ] += left_r*right_r + left_i*right_i;
								coarse_link[IM + coarse_link_index ] += left_r*right_i - right_r*left_i;
							}

						}
					}
				}
			}

		}
	}

}


void clovTripleProduct12cx12Site(const QDPCloverTerm& clov, const LatticePropagator& in_prop, LatticePropagator& out_prop)
{
	LatticeFermion in, out;
	LatticePropagator prop_tmp = zero;
	for(int spin=0; spin < 4; ++spin ) {
			for(int color=0; color < 3; ++color ) {

				// Extract component into 'in'
				PropToFerm(in_prop,in,color,spin);

				// Apply Dlsash in that Direction
				clov.apply(out, in, 0,0) ; // isign doesnt matter as Hermitian, cb=0
				clov.apply(out, in, 0,1) ; // isign doesnt matter as Hermitian, cb=1

				// Place back into prop
				FermToProp(out, prop_tmp, color, spin);
			} // color
		} // spin
	out_prop=adj(in_prop)*prop_tmp;
}

void clovTripleProductSite(const QDPCloverTerm& clov,const multi1d<LatticeFermion>& in_vecs, CoarseClover& cl_coarse)
{
	int Ncolor_c = cl_coarse.GetNumColor();
	int Nchiral_c = cl_coarse.GetNumChiral();


	// in vecs has size Ncolor_c = Ncolorspin_c/2
	// But this mixes both upper and lower spins
	// Once we deal with those separately we will need Ncolorspin_c results
	// And we will need to apply the 'DslashDir' separately to each aggregate

	assert( in_vecs.size() == Ncolor_c );
	assert( Nchiral_c == 2);

	// out_vecs is the result of applying clover term to in_vecs
	// NOTE!!!: Unlike with Dslash where (1 +/- gamma_mu) mixes the upper and lower spin components
	// Clover *does not* do this. In this chiral basis that we use Clover is block diagonal
	// So it acts independently on upper and lower spin components.
	// This means Ncolor vectors are sufficient. The upper components will hold the results of
	// clover_term applied to the upper components while the lower components will hold the results of
	// clover_term applied to the lower components in the same way in_vector combines upper and lower
	// components.
	multi1d<LatticeFermion> out_vecs( Ncolor_c );

	// Zero the output
	for(int j=0; j < Ncolor_c; ++j) {
		out_vecs[j]=zero;
	}

	// for each in-vector pull out respectively the lower and upper spins
	// multiply by clover and store in out_vecs. There will be Ncolor_c*Nchiral_c output
	// vectors
	for(int j=0; j < Ncolor_c; ++j) {

		// Clover term is block diagonal
		// So I can apply it once, the upper and lower spin components will
		// be acted on independently. No need to separate the aggregates before
		// applying

		LatticeFermion tmp=zero;
		for(int cb=0; cb < 2; ++cb) {
			clov.apply(out_vecs[j], in_vecs[j], 0, cb);

		}
	}

	// So now we should have 12 out vecs Which contain the upper and lower aggregates
	// The first 6 should have upper aggregation components, the second 6 should have the lower ones.
	IndexType coarse_cbsites = cl_coarse.GetInfo().GetNumCBSites();


	// Technically these outer loops should be over all the blocks.
	for(int cb=0; cb < 2; ++cb) {
		for(int cbsite=0; cbsite < coarse_cbsites; ++cbsite) {
			for(int chiral = 0; chiral < Nchiral_c; ++chiral ) {

				// This is now an Ncomplex*Ncolor_c x Ncomplex*Ncolor_c
				float *coarse_clov = cl_coarse.GetSiteChiralDataPtr(cb,cbsite, chiral);
				int site=rb[cb].siteTable()[cbsite];

				// This is an Ncolor_c x Ncolor_c matmul
				for(int matmul_row=0; matmul_row < Ncolor_c; ++matmul_row) {
					for(int matmul_col=0; matmul_col < Ncolor_c; ++matmul_col) {


						//Index in coarse link
						int coarse_clov_index = n_complex*(matmul_col+ Ncolor_c*matmul_row);

						// Init inner product
						coarse_clov[ RE + coarse_clov_index ] = 0;
						coarse_clov[ IM + coarse_clov_index ] = 0;

						// Inner product loop
						for(int k=0; k < Ncolor_c; ++k) {

							// [ V^H_upper   0      ] [  A_upper    B_upper ] = [ V^H_upper A_upper   V^H_upper B_upper  ]
							// [  0       V^H_lower ] [  A_lower    B_lower ]   [ V^H_lower A_lower   V^H_lower B_lower  ]
							//
							// But
							//  [ A_upper B_upper ] = [ Clov_upper      0      ] [ V_upper        0    ] = [ A_upper    0    ]
							//  [ A_lower B_lower ]   [      0      Clov_lower ] [    0       V_lower  ]   [   0     B_lower ]
							//
							// So really I need to just evaluate:  V^H_upper A_upper and V^H_lower B_lower
							//
							//
							int spin=k/Nc + chiral * (Ns/2);  // Upper or lower spin depending on chiral

							// k % Nc maps to color component (0,1,2)
							int color=k%Nc;

							// Right vector - chiral*Ncolor_c selects A_upper ( chiral=0 ) or B_lower (chiral=1)

							// NB: Out vecs has only NColor members
							REAL right_r = out_vecs[matmul_col].elem(site).elem(spin).elem(color).real();
							REAL right_i = out_vecs[matmul_col].elem(site).elem(spin).elem(color).imag();

							// Left vector -- only Ncolor_c components with [ V^H_upper V^H_lower ]
							//
							// ie a compact storage
							// rather than:
							//
							// [ V^H_upper   0      ]
							// [	  0       V^H_lower ]
							//
							// so index with row % Ncolor_c = matmul_row
							REAL left_r = in_vecs[matmul_row ].elem(site).elem(spin).elem(color).real();
							REAL left_i = in_vecs[matmul_row ].elem(site).elem(spin).elem(color).imag();

							// Accumulate inner product V^H_row A_column
							coarse_clov[RE + coarse_clov_index ] += left_r*right_r + left_i*right_i;
							coarse_clov[IM + coarse_clov_index ] += left_r*right_i - right_r*left_i;
						}

					}
				}
			}

		}
	}

}

#if 0
void RestrictSpinor( const multi1d<LatticeFermion>& V, const LatticeFermion& ferm_in, CoarseSpinor& out)
{
	int num_cbsites=out.GetInfo().GetNumCBSites();
	int num_colorspin=out.GetNumColorSpin();
	int num_color = out.GetNumColor();

	for(int cb=0; cb < 2; ++cb) {
		for(int cbsite=0; cbsite < num_cbsites; ++cbsite ) {

			for(int coarse_row=0; coarse_row < num_colorspin; ++coarse_row) {
				int coarse_row_color =
			}

		}
	}
}
#endif

TEST(TestInterface, TestQDPSpinorToCoarseSpinor)
{
	IndexArray latdims={{2,2,2,2}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	LatticeInfo info(latdims, 2, 6, NodeInfo());
	CoarseSpinor s_coarse(info);

	LatticeFermion in,out;
	gaussian(in);
	gaussian(out); // Make sure it is different from in

	QDPSpinorToCoarseSpinor(in,s_coarse);
	CoarseSpinorToQDPSpinor(s_coarse,out);

	LatticeFermion diff;
	diff = in -out;
	Double diff_norm = norm2(diff);
	Double rel_diff_norm = diff_norm/norm2(in);
	QDPIO::cout << "Diff Norm = " << sqrt(diff_norm) << std::endl;
	QDPIO::cout << "Relative Diff Norm = " << sqrt(rel_diff_norm) << std::endl;

}

TEST(TestInterface, TestQDPPropagatorToCoarsePropagator)
{
	IndexArray latdims={{2,2,2,2}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	LatticeInfo info(latdims, 2, 6, NodeInfo());
	CoarseGauge u_coarse(info);

	LatticePropagator in,out;

	for(int mu=0; mu < 2*n_dim; ++mu) {
		gaussian(in);
		gaussian(out); // Make sure it is different from in

		QDPPropToCoarseGaugeLink(in,u_coarse,mu);
		CoarseGaugeLinkToQDPProp(u_coarse,out,mu);

		LatticePropagator diff;
		diff = in -out;
		Double diff_norm = norm2(diff);
		Double rel_diff_norm = diff_norm/norm2(in);
		QDPIO::cout << "Dir: " << mu << " Diff Norm = " << sqrt(diff_norm) << std::endl;
		QDPIO::cout << "Dir: " << mu << " Relative Diff Norm = " << sqrt(rel_diff_norm) << std::endl;
	}
}



TEST(TestCoarseQDPXX, TestCoarseQDPXXDslash)
{
	IndexArray latdims={{2,2,2,2}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	multi1d<LatticeColorMatrix> u(Nd);

	QDPIO::cout << "Generating Random Gauge with Gaussian Noise" << std::endl;
	for(int mu=0; mu < Nd; ++mu) {
//		u[mu] = 1;
		gaussian(u[mu]);
		reunit(u[mu]);
	}

	// Generate the 'vectors' of which there are to be 12. Funnily this fits nicely into a propagator
	// Later on would be better to have this be a general unitary matrix per site.
	QDPIO::cout << "Generating Eye" << std::endl;
	LatticePropagator eye=1;

	Double eye_norm = norm2(eye);
	Double eye_norm_per_site = eye_norm/Layout::vol();
	QDPIO::cout << "Eye Norm Per Site " << eye_norm_per_site << std::endl;


	multi1d<LatticePropagator> dslash_links(8);

	for(int mu=0; mu < 8; ++mu) {
		QDPIO::cout << "Attempting Triple Product in direction: " << mu << std::endl;
		dslashTripleProduct12x12SiteDir(mu, u, eye, dslash_links[mu]);
	}


	// Next step should be to copy this into the fields needed for gauge and clover ops
	LatticeInfo info(latdims, 2, 6, NodeInfo());
	CoarseGauge u_coarse(info);
	for(int mu=0; mu < 8; ++mu) {
		QDPPropToCoarseGaugeLink(dslash_links[mu],u_coarse, mu);
	}

	QDPIO::cout << "Coarse Gauge Field initialized " << std::endl;


	LatticeFermion psi, d_psi, m_psi;
	gaussian(psi);

	m_psi = zero;

	// Apply Dslash to both CBs, isign=1
	// Result in m_psiu
	dslash(m_psi, u, psi, 1, 0);
	dslash(m_psi, u, psi, 1, 1);

	// CoarsSpinors
	CoarseSpinor coarse_s_in(info);
	CoarseSpinor coarse_s_out(info);

	// Import psi
	QDPSpinorToCoarseSpinor(psi, coarse_s_in);


	// Create A coarse operator
	int n_smt = 1;
	CoarseDiracOp D_op_coarse(info, n_smt);

	// Apply Coarse Op Dslash in Threads
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		D_op_coarse.Dslash(coarse_s_out, u_coarse, coarse_s_in, 0, tid);
		D_op_coarse.Dslash(coarse_s_out, u_coarse, coarse_s_in, 1, tid);
	}

	// Export Coarse spinor to QDP++ spinors.
	LatticeFermion coarse_d_psi = zero;
	CoarseSpinorToQDPSpinor(coarse_s_out, coarse_d_psi);

	// Find the difference between regular dslash and 'coarse' dslash
	LatticeFermion diff = m_psi - coarse_d_psi;

	QDPIO::cout << "Norm Diff[0] = " << sqrt(norm2(diff, rb[0])) << std::endl;
	QDPIO::cout << "Norm Diff[1] = " << sqrt(norm2(diff, rb[1])) 	<< std::endl;
	QDPIO::cout << "Norm Diff = " << sqrt(norm2(diff)) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[0] = " << sqrt(norm2(diff, rb[0])/norm2(psi,rb[0])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[1] = " << sqrt(norm2(diff, rb[1])/norm2(psi,rb[1])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff = " << sqrt(norm2(diff)/norm2(psi)) << std::endl;

}

TEST(TestCoarseQDPXX, TestCoarseQDPXXDslash2)
{
	IndexArray latdims={{2,2,2,2}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	multi1d<LatticeColorMatrix> u(Nd);

	QDPIO::cout << "Generating Random Gauge with Gaussian Noise" << std::endl;
	for(int mu=0; mu < Nd; ++mu) {
		//u[mu] = 1;
		gaussian(u[mu]);
		reunit(u[mu]);
	}

	multi1d<LatticeFermion> in_vecs(Nc*Ns/2);     // In terms of vectors
	multi1d<LatticePropagator> dslash_links(8); // In terms of propagators

	QDPIO::cout << "Generating Eye" << std::endl;
	LatticePropagator eye=1;


	// Pack 'Eye' vectors into to the in_vecs;
	for(int spin=0; spin < Ns/2; ++spin) {
		for(int color =0; color < Nc; ++color) {
			LatticeFermion upper = zero;
			LatticeFermion lower = zero;

			PropToFerm(eye, lower, color, spin);
			PropToFerm(eye, upper, color, spin+Ns/2);
			in_vecs[color + Nc*spin] = upper + lower;
		}
	}

	QDPIO::cout << "Printing in-vecs:" << std::endl;
	for(int spin=0; spin < 4; ++spin ) {
		for(int color=0; color < 3; ++color ) {
			for(int j=0; j < Nc*Ns/2; ++j) {
				printf("( %1.2lf, %1.2lf ) ", in_vecs[j].elem(0).elem(spin).elem(color).real(),
									in_vecs[j].elem(0).elem(spin).elem(color).imag() );
			}
			printf("\n");

		}
	}

	// Generate the Triple product into dslash_links[mu]
	for(int mu=0; mu < 8; ++mu) {
			QDPIO::cout << "Attempting Triple Product in direction: " << mu << std::endl;
			dslashTripleProduct12x12SiteDir(mu, u, eye, dslash_links[mu]);
	}



	// Next step should be to copy this into the fields needed for gauge and clover ops
	LatticeInfo info(latdims, 2, 6, NodeInfo());
	CoarseGauge u_coarse(info);


	// Generate the triple products directly into the u_coarse
	for(int mu=0; mu < 8; ++mu) {
		QDPIO::cout << "Attempting Triple Product in direction: " << mu << std::endl;
		dslashTripleProductSiteDir(mu, u, in_vecs, u_coarse);
	}

	for(int row=0; row < Ns*Nc; ++row) {
		int spin_row = row/Nc;
		int color_row = row % Nc;

		for(int column=0; column < Nc*Ns; ++column) {
			int spin_column = column / Nc;
			int color_column = column % Nc;

			printf("( %1.2lf, %1.2lf ) ", dslash_links[0].elem(0).elem(spin_row,spin_column).elem(color_row,color_column).real(),
					dslash_links[0].elem(0).elem(spin_row, spin_column).elem(color_row,color_column).imag() );

		}
		printf("\n");
	}
	printf("\n");

	float *coarse_link = u_coarse.GetSiteDirDataPtr(0,0,0);

	for(int row=0; row < Ns*Nc; ++row) {

		for(int column=0; column < Nc*Ns; ++column) {

			int coarse_link_index = n_complex*(column + Ns*Nc*row);
			printf(" ( %1.2lf, %1.2lf ) ", coarse_link[ RE+coarse_link_index], coarse_link[ IM + coarse_link_index]);
		}
		printf("\n");
	}


	QDPIO::cout << "Coarse Gauge Field initialized " << std::endl;


	LatticeFermion psi, d_psi, m_psi;
	gaussian(psi);

	m_psi = zero;

	// Apply Dslash to both CBs, isign=1
	// Result in m_psiu
	dslash(m_psi, u, psi, 1, 0);
	dslash(m_psi, u, psi, 1, 1);

	// CoarsSpinors
	CoarseSpinor coarse_s_in(info);
	CoarseSpinor coarse_s_out(info);

	// Import psi
	QDPSpinorToCoarseSpinor(psi, coarse_s_in);


	// Create A coarse operator
	int n_smt = 1;
	CoarseDiracOp D_op_coarse(info, n_smt);

	// Apply Coarse Op Dslash in Threads
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		D_op_coarse.Dslash(coarse_s_out, u_coarse, coarse_s_in, 0, tid);
		D_op_coarse.Dslash(coarse_s_out, u_coarse, coarse_s_in, 1, tid);
	}

	// Export Coarse spinor to QDP++ spinors.
	LatticeFermion coarse_d_psi = zero;
	CoarseSpinorToQDPSpinor(coarse_s_out, coarse_d_psi);

	// Find the difference between regular dslash and 'coarse' dslash
	LatticeFermion diff = m_psi - coarse_d_psi;

	QDPIO::cout << "Norm Diff[0] = " << sqrt(norm2(diff, rb[0])) << std::endl;
	QDPIO::cout << "Norm Diff[1] = " << sqrt(norm2(diff, rb[1])) 	<< std::endl;
	QDPIO::cout << "Norm Diff = " << sqrt(norm2(diff)) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[0] = " << sqrt(norm2(diff, rb[0])/norm2(psi,rb[0])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[1] = " << sqrt(norm2(diff, rb[1])/norm2(psi,rb[1])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff = " << sqrt(norm2(diff)/norm2(psi)) << std::endl;

}

TEST(TestCoarseQDPXX, TestCoarseQDPXXClov)
{
	IndexArray latdims={{2,2,2,2}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	multi1d<LatticeColorMatrix> u(Nd);

	QDPIO::cout << "Generating Random Gauge with Gaussian Noise" << std::endl;
	for(int mu=0; mu < Nd; ++mu) {
//		u[mu] = 1;
		gaussian(u[mu]);
		reunit(u[mu]);
	}

	// Now need to make a clover op
	 CloverFermActParams clparam;
	 AnisoParam_t aniso;

	  // Aniso prarams
	 aniso.anisoP=true;
	 aniso.xi_0 = 1.5;
	 aniso.nu = 0.95;
	 aniso.t_dir = 3;

	  // Set up the Clover params
	 clparam.anisoParam = aniso;

	  // Some mass
	 clparam.Mass = Real(0.1);

	 // Some random clover coeffs
	  clparam.clovCoeffR=Real(1.35);
	  clparam.clovCoeffT=Real(0.8);
	  QDPCloverTerm clov_qdp;
	  clov_qdp.create(u,clparam);



	// Generate the 'vectors' of which there are to be 12. Funnily this fits nicely into a propagator
	// Later on would be better to have this be a general unitary matrix per site.
	QDPIO::cout << "Generating Eye" << std::endl;
	LatticePropagator eye=1;

	Double eye_norm = norm2(eye);
	Double eye_norm_per_site = eye_norm/Layout::vol();
	QDPIO::cout << "Eye Norm Per Site " << eye_norm_per_site << std::endl;

	LatticePropagator tprod_result;

	clovTripleProduct12cx12Site(clov_qdp, eye, tprod_result);

	QDPIO::cout << "Checking Triple product result PropClover is still block diagonal" << std::endl;

	for(int spin_row=0; spin_row < 2; ++spin_row) {
		for(int spin_col=2; spin_col < 4; ++spin_col) {
			for(int col_row = 0; col_row < 3; ++col_row ) {
				for(int col_col = 0; col_col < 3; ++col_col ) {
					float re = tprod_result.elem(0).elem(spin_row,spin_col).elem(col_row,col_col).real();
					float im = tprod_result.elem(0).elem(spin_row,spin_col).elem(col_row,col_col).imag();

					ASSERT_FLOAT_EQ(re,0);
					ASSERT_FLOAT_EQ(im,0);

				}
			}
		}
	}

	for(int spin_row=2; spin_row < 4; ++spin_row) {
		for(int spin_col=0; spin_col < 2; ++spin_col) {
			for(int col_row = 0; col_row < 3; ++col_row ) {
				for(int col_col = 0; col_col < 3; ++col_col ) {
					float re = tprod_result.elem(0).elem(spin_row,spin_col).elem(col_row,col_col).real();
					float im = tprod_result.elem(0).elem(spin_row,spin_col).elem(col_row,col_col).imag();

					ASSERT_FLOAT_EQ(re,0);
					ASSERT_FLOAT_EQ(im,0);

				}
			}
		}
	}

	LatticeFermion orig;
	gaussian(orig);
	LatticeFermion orig_res=zero;


	// orig_res = A orig
	clov_qdp.apply(orig_res, orig, 0, 0);
	clov_qdp.apply(orig_res, orig, 0, 1);


	LatticeFermion diff = zero;
	{
		QDPIO::cout << "Checking Triple product result PropClover can be multiplied with Fermion" << std::endl;

		LatticeFermion tprod_res_ferm = zero;

		// Just multiply by propgatator
		tprod_res_ferm = tprod_result*orig;

		diff = tprod_res_ferm - orig_res;
		QDPIO::cout << "Diff Norm = " << sqrt(norm2(diff)) << std::endl;
	}


	QDPIO::cout << "Importing Triple product result PropClover into CoarseClover " << std::endl;
	LatticeInfo info(latdims, 2, 6, NodeInfo());
	CoarseClover c_clov(info);
	QDPPropToCoarseClover(tprod_result, c_clov);

	CoarseSpinor s_in(info);
	QDPSpinorToCoarseSpinor(orig,s_in);

	CoarseSpinor s_out(info);

	int n_smt = 1;
	CoarseDiracOp D(info,n_smt);

#pragma omp parallel
	{
		int tid = omp_get_thread_num();

		D.CloverApply(s_out, c_clov, s_in,0,tid);
		D.CloverApply(s_out, c_clov, s_in,1,tid);


	}

	LatticeFermion coarse_res;
	CoarseSpinorToQDPSpinor(s_out,coarse_res);
	diff = orig_res - coarse_res;

	QDPIO::cout << "Norm Diff[0] = " << sqrt(norm2(diff, rb[0])) << std::endl;
	QDPIO::cout << "Norm Diff[1] = " << sqrt(norm2(diff, rb[1])) 	<< std::endl;
	QDPIO::cout << "Norm Diff = " << sqrt(norm2(diff)) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[0] = " << sqrt(norm2(diff, rb[0])/norm2(orig,rb[0])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[1] = " << sqrt(norm2(diff, rb[1])/norm2(orig,rb[1])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff = " << sqrt(norm2(diff)/norm2(orig)) << std::endl;



}

TEST(TestCoarseQDPXX, TestCoarseQDPXXClov2)
{
	IndexArray latdims={{2,2,2,2}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	multi1d<LatticeColorMatrix> u(Nd);

	QDPIO::cout << "Generating Random Gauge with Gaussian Noise" << std::endl;
	for(int mu=0; mu < Nd; ++mu) {
//		u[mu] = 1;
		gaussian(u[mu]);
		reunit(u[mu]);
	}

	// Now need to make a clover op
	 CloverFermActParams clparam;
	 AnisoParam_t aniso;

	  // Aniso prarams
	 aniso.anisoP=true;
	 aniso.xi_0 = 1.5;
	 aniso.nu = 0.95;
	 aniso.t_dir = 3;

	  // Set up the Clover params
	 clparam.anisoParam = aniso;

	  // Some mass
	 clparam.Mass = Real(0.1);

	 // Some random clover coeffs
	  clparam.clovCoeffR=Real(1.35);
	  clparam.clovCoeffT=Real(0.8);
	  QDPCloverTerm clov_qdp;
	  clov_qdp.create(u,clparam);



	// Generate the 'vectors' of which there are to be 12. Funnily this fits nicely into a propagator
	// Later on would be better to have this be a general unitary matrix per site.
	QDPIO::cout << "Generating Eye" << std::endl;
	LatticePropagator eye=1;

	Double eye_norm = norm2(eye);
	Double eye_norm_per_site = eye_norm/Layout::vol();
	QDPIO::cout << "Eye Norm Per Site " << eye_norm_per_site << std::endl;

	// Pack 'Eye' vectors into to the in_vecs;
	// This is similar to the case when we will have noise filled vectors
	multi1d<LatticeFermion> in_vecs(Nc*Ns/2);

	for(int spin=0; spin < Ns/2; ++spin) {
		for(int color =0; color < Nc; ++color) {
			LatticeFermion upper = zero;
			LatticeFermion lower = zero;

			PropToFerm(eye, lower, color, spin);
			PropToFerm(eye, upper, color, spin+Ns/2);
			in_vecs[color + Nc*spin] = upper + lower;
		}
	}


	LatticePropagator tprod_result;
	clovTripleProduct12cx12Site(clov_qdp, eye, tprod_result);
#if 0

	QDPIO::cout << "Checking Triple product result PropClover is still block diagonal" << std::endl;

	for(int spin_row=0; spin_row < 2; ++spin_row) {
		for(int spin_col=2; spin_col < 4; ++spin_col) {
			for(int col_row = 0; col_row < 3; ++col_row ) {
				for(int col_col = 0; col_col < 3; ++col_col ) {
					float re = tprod_result.elem(0).elem(spin_row,spin_col).elem(col_row,col_col).real();
					float im = tprod_result.elem(0).elem(spin_row,spin_col).elem(col_row,col_col).imag();

					ASSERT_FLOAT_EQ(re,0);
					ASSERT_FLOAT_EQ(im,0);

				}
			}
		}
	}

	for(int spin_row=2; spin_row < 4; ++spin_row) {
		for(int spin_col=0; spin_col < 2; ++spin_col) {
			for(int col_row = 0; col_row < 3; ++col_row ) {
				for(int col_col = 0; col_col < 3; ++col_col ) {
					float re = tprod_result.elem(0).elem(spin_row,spin_col).elem(col_row,col_col).real();
					float im = tprod_result.elem(0).elem(spin_row,spin_col).elem(col_row,col_col).imag();

					ASSERT_FLOAT_EQ(re,0);
					ASSERT_FLOAT_EQ(im,0);

				}
			}
		}
	}

#endif

	QDPIO::cout << "Importing Triple product result PropClover into CoarseClover " << std::endl;


	// Now test the new packer.
	LatticeInfo info(latdims, 2, 6, NodeInfo());
	CoarseClover c_clov(info);
	clovTripleProductSite(clov_qdp,in_vecs, c_clov);

	// Now create a LatticeFermion and apply both the QDP++ and the Coarse Clover
	LatticeFermion orig;
	gaussian(orig);
	LatticeFermion orig_res=zero;


	// orig_res = A orig
	for(int cb=0; cb < 2; ++cb) {
		clov_qdp.apply(orig_res, orig, 0, cb);
	}

	// Convert original spinor to a coarse spinor
	CoarseSpinor s_in(info);
	QDPSpinorToCoarseSpinor(orig,s_in);
	CoarseSpinor s_out(info);

	int n_smt = 1;
	CoarseDiracOp D(info,n_smt);

#pragma omp parallel
	{
		int tid = omp_get_thread_num();

		D.CloverApply(s_out, c_clov, s_in,0,tid);
		D.CloverApply(s_out, c_clov, s_in,1,tid);
	}

	LatticeFermion coarse_res;
	CoarseSpinorToQDPSpinor(s_out,coarse_res);

	LatticeFermion diff = orig_res - coarse_res;

	QDPIO::cout << "Norm Diff[0] = " << sqrt(norm2(diff, rb[0])) << std::endl;
	QDPIO::cout << "Norm Diff[1] = " << sqrt(norm2(diff, rb[1])) 	<< std::endl;
	QDPIO::cout << "Norm Diff = " << sqrt(norm2(diff)) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[0] = " << sqrt(norm2(diff, rb[0])/norm2(orig,rb[0])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff[1] = " << sqrt(norm2(diff, rb[1])/norm2(orig,rb[1])) << std::endl;
	QDPIO::cout << "Rel. Norm Diff = " << sqrt(norm2(diff)/norm2(orig)) << std::endl;



}


int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

