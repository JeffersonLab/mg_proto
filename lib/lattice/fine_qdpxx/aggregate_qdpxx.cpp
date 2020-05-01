/*
 * aggregate_qdpxx.cpp
 *
 *  Created on: Dec 9, 2016
 *      Author: bjoo
 */

#include "lattice/fine_qdpxx/aggregate_qdpxx.h"
#include "lattice/fine_qdpxx/transf.h"

#include "lattice/constants.h"

using namespace QDP;


namespace MG {


// Apply a single direction of Dslash
void DslashDirQDPXX(LatticeFermion& out, const multi1d<LatticeColorMatrix>& u, const LatticeFermion& in, int dir)
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
		MasterLog(ERROR,"Unknown direction. You oughtnt call this\n");
		QDP_abort(1);
	}
}

// In the full blown version, instead of the 'site' we will sum over sites in
// a block
void axAggrQDPXX(const double alpha, LatticeFermion& v, int site, int aggr)
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
void caxpyAggrQDPXX(const std::complex<double>& alpha, const LatticeFermion& x, LatticeFermion& y, int site, int aggr)
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
double norm2AggrQDPXX(const LatticeFermion& v, int site, int aggr)
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
innerProductAggrQDPXX(const LatticeFermion& left, const LatticeFermion& right, int site, int aggr)
{
	std::complex<double> iprod = {0,0};

	for(int spin=aggr*(Ns/2); spin < (aggr+1)*(Ns/2); ++spin) {
		for(int color=0; color < Nc; ++color) {
			REAL left_r = left.elem(site).elem(spin).elem(color).real();
			REAL left_i = left.elem(site).elem(spin).elem(color).imag();

			REAL right_r = right.elem(site).elem(spin).elem(color).real();
			REAL right_i = right.elem(site).elem(spin).elem(color).imag();

			std::complex<double> left_elem = { left_r, left_i };
			std::complex<double> right_elem = { right_r, right_i };

			iprod += conj(left_elem)*right_elem;
		}
	}
	return iprod;
}

void orthonormalizeAggregatesQDPXX(multi1d<LatticeFermion>& vecs)
{


	for(int aggr=0; aggr < 2; ++aggr) {

		MasterLog(DEBUG, "Orthonormalizing Aggregate: %d\n",aggr);

			// This will be over blocks...
			for(int site=QDP::all.start(); site <= QDP::all.end(); ++site) {


				// do vecs[0] ... vecs[N]
				for(int curr_vec=0; curr_vec < vecs.size(); curr_vec++) {

					// orthogonalize against previous vectors
					// if curr_vec == 0 this will be skipped
					for(int prev_vec=0; prev_vec < curr_vec; prev_vec++) {

						std::complex<double> iprod = innerProductAggrQDPXX( vecs[prev_vec], vecs[curr_vec], site, aggr);
						std::complex<double> minus_iprod=std::complex<double>(-real(iprod), -imag(iprod) );

						// curr_vec <- curr_vec - <curr_vec|prev_vec>*prev_vec = -iprod*prev_vec + curr_vec
						caxpyAggrQDPXX( minus_iprod, vecs[prev_vec], vecs[curr_vec], site, aggr);
					}

					// Normalize current vector
					double inv_norm = ((double)1)/sqrt(norm2AggrQDPXX(vecs[curr_vec], site, aggr));

					// vecs[curr_vec] = inv_norm * vecs[curr_vec]
					axAggrQDPXX(inv_norm, vecs[curr_vec], site, aggr);
				}


			}	// site
	}// aggregates
}

//  prop_out(x) = prop_in^\dagger(x) ( 1 +/- gamma_mu ) U mu(x) prop_in(x + mu)
//
//  This routine is used to test Coarse Dslash
//  Initially, prop_in should be just I_{12x12}
//  However, any unitary matrix on the sites (orthonormal basis) would do
void dslashTripleProduct12x12SiteDirQDPXX(int dir, const multi1d<LatticeColorMatrix>& u, const LatticePropagator& in_prop, LatticePropagator& out_prop)
{
	LatticeFermion in, out;
	LatticePropagator prop_tmp=zero;
	// Loop through spins and colors

	for(int spin=0; spin < 4; ++spin ) {
		for(int color=0; color < 3; ++color ) {

			// Extract component into 'in'
			PropToFerm(in_prop,in,color,spin);

			// Apply Dlsash in that Direction
			DslashDirQDPXX(out, u, in, dir);

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


void extractAggregateQDPXX(LatticeFermion& target, const LatticeFermion& src, int aggr )
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
void dslashTripleProductSiteDirQDPXX(int dir, const multi1d<LatticeColorMatrix>& u, const multi1d<LatticeFermion>& in_vecs, CoarseGauge& u_coarse)
{


	int Ncolor_c = u_coarse.GetNumColor();
	int Ncolorspin_c = u_coarse.GetNumColorSpin();
	int Naggr_c = Ncolorspin_c/Ncolor_c;

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
			extractAggregateQDPXX(tmp, in_vecs[j], aggr);
			DslashDirQDPXX(out_vecs[aggr*Ncolor_c+j], u, tmp, dir);
		}
	}


	// So now we should have 12 out vecs Which contain the upper and lower aggregates
	// The first 6 should have upper aggregation components, the second 6 should have the lower ones.
	IndexType coarse_cbsites = u_coarse.GetInfo().GetNumCBSites();


	// Technically these outer loops should be over all the blocks.
	for(int cb=0; cb < 2; ++cb) {
		for(int cbsite=0; cbsite < coarse_cbsites; ++cbsite) {

			// This is now an Ncomplex*NumColorspin*NumColorspin array
			// THis is based on a block being a site. In that case there is no
			// induced local term and filling out the connection is fine.

			float *coarse_link = u_coarse.GetSiteDirDataPtr(cb,cbsite,dir);
			int site=rb[cb].siteTable()[cbsite];

			// FIXME Switch row and column
			for(int aggr_row=0; aggr_row < Naggr_c; ++aggr_row) {
				for(int aggr_col=0; aggr_col < Naggr_c; ++aggr_col ) {

					// This is an Ncolor_c x Ncolor_c matmul
					for(int matmul_row=0; matmul_row < Ncolor_c; ++matmul_row) {
						for(int matmul_col=0; matmul_col < Ncolor_c; ++matmul_col) {

							// Offset by the aggr_row and aggr_column
							int row = aggr_row*Ncolor_c + matmul_row;
							int col = aggr_col*Ncolor_c + matmul_col;

							//Index in coarse link
							// FIXME: Switch Row and column
							int coarse_link_index = n_complex*(row + Ncolorspin_c*col);

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


void clovTripleProduct12cx12SiteQDPXX(const QDPCloverTerm& clov, const LatticePropagator& in_prop, LatticePropagator& out_prop)
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

void clovTripleProductSiteQDPXX(const QDPCloverTerm& clov,const multi1d<LatticeFermion>& in_vecs, CoarseGauge& cl_coarse)
{
	int Ncolor_c = cl_coarse.GetNumColor();
	int Nchiral_c = 2;
	int Ncolorspin_c = Ncolor_c*Nchiral_c;


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
				float *coarse_clov = cl_coarse.GetSiteDiagDataPtr(cb,cbsite);
				int site=rb[cb].siteTable()[cbsite];

				// This is an Ncolor_c x Ncolor_c matmul
				// FIXME: Switch row and column
				for(int matmul_row=0; matmul_row < Ncolor_c; ++matmul_row) {
					for(int matmul_col=0; matmul_col < Ncolor_c; ++matmul_col) {

						int c_offset = (chiral == 0) ? 0 : Ncolor_c;

						//Index in coarse link
						// FIXME: store matrix in row major order: Have rows run faster
						int coarse_clov_index = n_complex*(matmul_row+c_offset + Ncolorspin_c*(matmul_col+c_offset) );

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

// Later on there will be Coarse to Coarse maybe?
//
void restrictSpinorQDPXXFineToCoarse( const multi1d<LatticeFermion>& v, const LatticeFermion& ferm_in, CoarseSpinor& out)
{
	int num_coarse_cbsites=out.GetInfo().GetNumCBSites();
	int num_coarse_color = out.GetNumColor();
	assert(out.GetNCol() == 1);

	assert( v.size() == num_coarse_color );

	// This will be a loop over blocks
	for(int cb=0; cb < 2; ++cb) {
		for(int cbsite=0; cbsite < num_coarse_cbsites; ++cbsite ) {

			float* site_spinor = out.GetSiteDataPtr(0,cb,cbsite);
			int qdpsite = rb[cb].siteTable()[cbsite];

			for(int chiral = 0; chiral < 2; ++chiral ) {
			for(int coarse_color=0; coarse_color  < num_coarse_color; coarse_color++) {
				int coarse_colorspin = coarse_color + chiral * num_coarse_color;
				site_spinor[ RE + n_complex*coarse_colorspin ] = 0;
				site_spinor[ IM + n_complex*coarse_colorspin ] = 0;


				for(int spin=0; spin < Ns/2; ++spin ) {
					for(int color=0; color < Nc; ++color ) {

						int targ_spin = spin + chiral*(Ns/2); // Offset by whether upper/lower

						REAL left_r = v[ coarse_color ].elem(qdpsite).elem(targ_spin).elem(color).real();
						REAL left_i = v[ coarse_color ].elem(qdpsite).elem(targ_spin).elem(color).imag();

						REAL right_r = ferm_in.elem(qdpsite).elem(targ_spin).elem(color).real();
						REAL right_i = ferm_in.elem(qdpsite).elem(targ_spin).elem(color).imag();

						// It is V_j^H  ferm_in so conj(left)*right.
						site_spinor[ RE + n_complex*coarse_colorspin ] += left_r * right_r + left_i * right_i;
						site_spinor[ IM + n_complex*coarse_colorspin ] += left_r * right_i - right_r * left_i;

					}
				}

			}
			}
		}
	}
}

// Later on there may be Coarse To Coarse Maybe
void prolongateSpinorCoarseToQDPXXFine( const multi1d<LatticeFermion>& v, const CoarseSpinor& coarse_in, LatticeFermion& fine_out)
{
	int num_coarse_cbsites=coarse_in.GetInfo().GetNumCBSites();
	int num_coarse_color = coarse_in.GetNumColor();
	assert(coarse_in.GetNCol() == 1);

	assert( v.size() == num_coarse_color );

	// Two ways to look at this. One way is that we need to prolongate the coarse blocks.
	// For now each block is a site. I'll figure out how to deal with sites in the blocks later
	//
	for(int cb=0; cb < 2; ++cb) {
		for(int cbsite=0; cbsite < num_coarse_cbsites; ++cbsite ) {

			const float *coarse_spinor = coarse_in.GetSiteDataPtr(0,cb,cbsite);
			int qdpsite = rb[cb].siteTable()[cbsite];

			for(int fine_spin=0; fine_spin < Ns; ++fine_spin) {

				int chiral = fine_spin < (Ns/2) ? 0 : 1;

				for(int fine_color=0; fine_color < Nc; fine_color++ ) {

					fine_out.elem(qdpsite).elem(fine_spin).elem(fine_color).real() = 0;
					fine_out.elem(qdpsite).elem(fine_spin).elem(fine_color).imag() = 0;


					for(int coarse_color = 0; coarse_color < num_coarse_color; coarse_color++) {

						REAL left_r = v[coarse_color].elem(qdpsite).elem(fine_spin).elem(fine_color).real();
						REAL left_i = v[coarse_color].elem(qdpsite).elem(fine_spin).elem(fine_color).imag();

						int colorspin = coarse_color + chiral*num_coarse_color;
						REAL right_r = coarse_spinor[ RE + n_complex*colorspin];
						REAL right_i = coarse_spinor[ IM + n_complex*colorspin];

						// V_j | out  (rather than V^{H}) so needs regular complex mult?
						fine_out.elem(qdpsite).elem(fine_spin).elem(fine_color).real() += left_r * right_r - left_i * right_i;
						fine_out.elem(qdpsite).elem(fine_spin).elem(fine_color).imag() += left_i * right_r + left_r * right_i;
					}
				}
			}
		}
	}

}
}

