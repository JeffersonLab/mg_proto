/*
 * qphix_aggregate.cpp
 *
 *  Created on: Oct 20, 2017
 *      Author: bjoo
 */
#include <lattice/qphix/qphix_aggregate.h>
#include <lattice/fine_qdpxx/aggregate_block_qdpxx.h>
#include <lattice/qphix/qphix_blas_wrappers.h>
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/geometry_utils.h"
#include "lattice/coarse/coarse_l1_blas.h"
#include "lattice/coarse/aggregate_block_coarse.h"
#include <cassert>
#include<omp.h>



namespace MG {


// Implementation -- where possible call the site versions
//! v *= alpha (alpha is real) over and aggregate in a block, v is a QDP++ Lattice Fermion
template<typename QT>
inline
void axBlockAggrT(const double alpha, QT& v, const Block& block, int aggr)
{
  auto block_sitelist = block.getCBSiteList();
  int num_sites = block.getNumSites();
  const LatticeInfo& info = v.GetInfo();

  const int num_color = info.GetNumColors();

  const int min_spin = 2*aggr;
  const int max_spin = min_spin + 2;

//#pragma omp parallel for
  for(int site=0; site < num_sites; ++site) {

    const CBSite& cbsite = block_sitelist[site];
    for(int spin=min_spin; spin < max_spin; ++spin) {
      for(int color=0; color < 3; color++) {
        v(cbsite.cb, cbsite.site, spin,color, RE) *= alpha;
        v(cbsite.cb, cbsite.site, spin,color, IM) *= alpha;
      } // color
    } // spin
  } // site
}


void axBlockAggr(const double alpha, QPhiXSpinor& v, const Block& block, int aggr)
{
  axBlockAggrT(alpha,v,block,aggr);
}

void axBlockAggr(const double alpha, QPhiXSpinorF& v, const Block& block, int aggr)
{
  axBlockAggrT(alpha,v,block,aggr);
}


//! y += alpha * x (alpha is complex) over aggregate in a block, x, y are QDP++ LatticeFermions;
template<typename QS>
inline
void caxpyBlockAggrT(const std::complex<double>& alpha, const QS& x, QS& y,  const Block& block, int aggr)
{

  auto block_sitelist = block.getCBSiteList();
  int num_sites = block.getNumSites();

  AssertCompatible( x.GetInfo(), y.GetInfo() );

  const LatticeInfo& info = y.GetInfo();

  const int min_spin = 2*aggr;
  const int max_spin = min_spin + 2;

// #pragma omp parallel for
  for(int site=0; site < num_sites; ++site) {
    const CBSite& cbsite = block_sitelist[site];

    for(int spin = min_spin; spin < max_spin; ++spin) {
      for(int color=0; color < 3; ++color ) {
        double ar = real(alpha);
        double ai = imag(alpha);
        double xr = x(cbsite.cb, cbsite.site, spin,color, RE);
        double xi = x(cbsite.cb, cbsite.site, spin,color, IM);
        y(cbsite.cb, cbsite.site, spin, color, RE) += ar*xr - ai*xi;
        y(cbsite.cb, cbsite.site, spin, color, IM) += ar*xi + ai*xr;
      }
    }
  }

}

void caxpyBlockAggr(const std::complex<double>& alpha, const QPhiXSpinorF& x, QPhiXSpinorF& y,  const Block& block, int aggr)
{
  caxpyBlockAggrT(alpha,x,y,block,aggr);
}
void caxpyBlockAggr(const std::complex<double>& alpha, const QPhiXSpinor& x, QPhiXSpinor& y,  const Block& block, int aggr)
{
  caxpyBlockAggrT(alpha,x,y,block,aggr);
}

//! return || v ||^2 over an aggregate in a block, v is a QDP++ LatticeFermion
template<typename QS>
inline
double norm2BlockAggrT(const QS& v, const Block& block, int aggr)
{
  auto block_sitelist = block.getCBSiteList();
  int num_sites = block.getNumSites();

  const LatticeInfo& info = v.GetInfo();
  const int min_spin = 2*aggr;
  const int max_spin = min_spin+2;

  double block_sum=0;


//#pragma omp parallel for reduction(+:block_sum)
  for(int site=0; site < num_sites; ++site) {
    const CBSite& cbsite = block_sitelist[site];
    for(int spin=min_spin; spin < max_spin; ++spin) {
      for(int color=0; color < 3; ++color) {
        double vr = v(cbsite.cb, cbsite.site, spin, color, RE);
        double vi = v(cbsite.cb, cbsite.site, spin, color, IM);

        block_sum += vr*vr + vi*vi;
      }
    }
  }

  return block_sum;
}

double norm2BlockAggr(const QPhiXSpinor& v, const Block& block, int aggr)
{
  return norm2BlockAggrT(v,block,aggr);
}

double norm2BlockAggr(const QPhiXSpinorF& v, const Block& block, int aggr)
{
  return norm2BlockAggrT(v,block,aggr);
}

//! return < left | right > = sum left^\dagger_i * right_i for an aggregate, over a block
template<typename QS>
inline
std::complex<double>
innerProductBlockAggrT(const QS& left, const QS& right, const Block& block, int aggr)
{

  auto block_sitelist = block.getCBSiteList();
  int num_sites = block.getNumSites();

  AssertCompatible( left.GetInfo(), right.GetInfo() );

  const LatticeInfo& info = right.GetInfo();
  const int min_spin = 2*aggr;
  const int max_spin = min_spin + 2;

  double real_part=0;
  double imag_part=0;

//#pragma omp parallel for reduction(+:real_part) reduction(+:imag_part)
  for(int site=0; site < num_sites; ++site) {

    const CBSite& cbsite = block_sitelist[site];
    for(int spin=min_spin; spin < max_spin; ++spin) {
       for(int color=0; color < 3; ++color ) {

         double left_r = left(cbsite.cb, cbsite.site, spin, color, RE);
         double left_i = left(cbsite.cb, cbsite.site, spin, color, IM);

         double right_r = right(cbsite.cb, cbsite.site, spin, color, RE);
         double right_i = right(cbsite.cb, cbsite.site, spin, color, IM);

         real_part +=  (left_r*right_r) + (left_i*right_i);
         imag_part +=  (left_r*right_i) - (left_i*right_r);
       }
    }
  }

  std::complex<double> ret_val(real_part,imag_part);
  return ret_val;
}

std::complex<double>
innerProductBlockAggr(const QPhiXSpinor& left, const QPhiXSpinor& right, const Block& block, int aggr)
{
  return innerProductBlockAggrT(left,right,block,aggr);
}
std::complex<double>
innerProductBlockAggr(const QPhiXSpinorF& left, const QPhiXSpinorF& right, const Block& block, int aggr)
{
  return innerProductBlockAggrT(left,right,block,aggr);
}

//! Extract the spins belonging to a given aggregate from QDP++ source vector src, into QDP++ target vector target
template<typename QS>
inline
void extractAggregateBlockT(QS& target, const QS& src, const Block& block, int aggr )
{

  auto block_sitelist = block.getCBSiteList();
  int num_sites = block.getNumSites();


  const LatticeInfo& info = src.GetInfo();
  AssertCompatible( info, target.GetInfo() );

  const int min_spin = 2*aggr;
  const int max_spin = min_spin + 2;

//#pragma omp parallel for
  for(int site=0; site < num_sites; ++site) {
    const CBSite& cbsite = block_sitelist[site];
    for(int spin=min_spin; spin < max_spin; ++spin) {
      for(int color=0; color < 3; ++color ) {
        target(cbsite.cb, cbsite.site, spin, color, RE) = src(cbsite.cb, cbsite.site, spin, color, RE);
        target(cbsite.cb, cbsite.site, spin, color, IM) = src(cbsite.cb, cbsite.site, spin, color, IM);

      }
    }
  }
}

void extractAggregateBlock(QPhiXSpinor& target, const QPhiXSpinor& src, const Block& block, int aggr )
{
  extractAggregateBlockT(target,src,block,aggr);
}
void extractAggregateBlock(QPhiXSpinorF& target, const QPhiXSpinorF& src, const Block& block, int aggr )
{
  extractAggregateBlockT(target,src,block,aggr);
}

//! Extract the spins belonging to a given aggregate from QDP++ source vector src, into QDP++ target vector target
template<typename QS>
inline
void extractAggregateT(QS& target, const QS& src, int aggr )
{


  const LatticeInfo& info = src.GetInfo();
  AssertCompatible( info, target.GetInfo() );
  const int min_spin = 2*aggr;
  const int max_spin = min_spin + 2;

  const int num_cbsites = info.GetNumCBSites();

//#pragma omp parallel for collapse(2)
  for(int cb =0; cb < n_checkerboard; ++cb) {
    for(int site=0; site < num_cbsites; ++site) {

      for(int spin=min_spin; spin < max_spin; ++spin) {
        for(int color=0; color < 3; ++color ) {
          target(cb,site,spin,color,RE) = src(cb,site,spin,color,RE);
          target(cb,site,spin,color,IM) = src(cb,site,spin,color,IM);
        }
      }
    }
  }

}


void extractAggregate(QPhiXSpinor& target, const QPhiXSpinor& src, int aggr )
{
  extractAggregateT(target,src,aggr);
}

void extractAggregate(QPhiXSpinorF& target, const QPhiXSpinorF& src, int aggr ) {
  extractAggregateT(target,src,aggr);
}


//! Orthonormalize vecs over the spin aggregates within the sites
template<typename QS>
inline
void orthonormalizeBlockAggregatesT(std::vector<std::shared_ptr<QS>>& vecs,
    const std::vector<Block>& block_list)
{
  int num_blocks = block_list.size();

#pragma omp parallel for collapse(2)
  for(int aggr=0; aggr < 2; ++aggr) {

    for(int block_id=0; block_id < num_blocks; block_id++) {

      const Block& block = block_list[block_id];

      // This will be over blocks...
      // do vecs[0] ... vecs[N]
      for(IndexType curr_vec=0; curr_vec < static_cast<IndexType>(vecs.size()); curr_vec++) {

        // orthogonalize against previous vectors
        // if curr_vec == 0 this will be skipped
        for(int prev_vec=0; prev_vec < curr_vec; prev_vec++) {

          std::complex<double> iprod = innerProductBlockAggr( *(vecs[prev_vec]), *(vecs[curr_vec]), block, aggr);
          std::complex<double> minus_iprod=std::complex<double>(-real(iprod), -imag(iprod) );

          // curr_vec <- curr_vec - <curr_vec|prev_vec>*prev_vec = -iprod*prev_vec + curr_vec
          caxpyBlockAggr( minus_iprod, *(vecs[prev_vec]), *(vecs[curr_vec]), block, aggr);

        }

        // Normalize current vector
        double inv_norm = ((double)1)/sqrt(norm2BlockAggr(*(vecs[curr_vec]), block, aggr));

        // vecs[curr_vec] = inv_norm * vecs[curr_vec]
        axBlockAggr(inv_norm, *(vecs[curr_vec]), block, aggr);
      }


    } // block
  }// aggregates
}

void orthonormalizeBlockAggregates(std::vector<std::shared_ptr<QPhiXSpinor>>& vecs,
    const std::vector<Block>& block_list)
{
  orthonormalizeBlockAggregatesT(vecs,block_list);
}

void orthonormalizeBlockAggregates(std::vector<std::shared_ptr<QPhiXSpinorF>>& vecs,
    const std::vector<Block>& block_list)
{
  orthonormalizeBlockAggregatesT(vecs,block_list);
}

//! 'Restrict' a QDP++ spinor to a CoarseSpinor with the same geometry
template<typename QS>
void restrictSpinorT( const std::vector<Block>& blocklist, const std::vector< std::shared_ptr<QS> >& fine_vecs,
		const QS& fine_in, CoarseSpinor& out)
{

	const int num_coarse_cbsites = out.GetInfo().GetNumCBSites();
	const int num_coarse_color = out.GetNumColor();

	// Sanity check. The number of sites in the coarse spinor
	// Has to equal the number of blocks
	//  assert( n_checkerboard*num_coarse_cbsites == static_cast<const int>(blocklist.size()) );

	// The number of vectors has to eaqual the number of coarse colors
	assert( fine_vecs.size() == num_coarse_color );

	// This will be a loop over blocks

#pragma omp parallel for collapse(2)
	for(int block_cb = 0; block_cb < n_checkerboard; ++block_cb ) {
		for(int block_cbsite = 0 ; block_cbsite < num_coarse_cbsites; ++block_cbsite) {

			for(int chiral = 0; chiral < 2; ++chiral ) {
				for(int coarse_color=0; coarse_color  < num_coarse_color; coarse_color++) {

					int block_idx = block_cbsite + block_cb*num_coarse_cbsites;

					// The coarse site spinor is where we will write the result
					float* coarse_site_spinor = out.GetSiteDataPtr(block_cb,block_cbsite);

					// Identify the current block
					const Block& block = blocklist[block_idx];

					// Get the list of fine sites in the blocks
					auto block_sitelist = block.getCBSiteList();
					auto num_sites_in_block = block_sitelist.size();

					// Our loop is over coarse_colors and chiralities -- to fill out the colorspin components
					// However, each colorspin component will involve a site loop, and we can compute the contributions
					// to both chiralities of the color component from a vector in a single loop. So I put the fine site
					// loop outside of the chirality loop.
					//
					// An optimization/stabilization will be to  accumulate these loops in double since they are
					// over potentially large number of sites (e.g. 4^4)

					// Remember that coarse color picks the vector so have this outermost now,
					// Since we will be working in a vector at a time
					// Now loop over the chiral components. These are local in a site at the level of spin

					int coarse_colorspin = coarse_color + chiral * num_coarse_color;

					float sum_cspin_re=0;
					float sum_cspin_im=0;
					// Now aggregate over all the sites in the block -- this will be over a single vector...
					// NB: The loop indices may be later rerolled, e.g. if we can restrict multiple vectors at once
					// Then having the coarse_color loop inner will be better.
					for( IndexType fine_site_idx = 0; fine_site_idx < static_cast<IndexType>(num_sites_in_block); fine_site_idx++ ) {

						// Find the fine site
						const CBSite& fine_cbsite = block_sitelist[fine_site_idx];
						const int fine_site = (rb[ fine_cbsite.cb ].siteTable())[fine_cbsite.site ];




						// Aggregate the spins for the site.
						for(int spin=0; spin < Ns/2; ++spin ) {
							for(int color=0; color < Nc; ++color ) {
								int targ_spin = spin + chiral*(Ns/2); // Offset by whether upper/lower

								REAL left_r = (*(fine_vecs[ coarse_color ]))(fine_cbsite.cb, fine_cbsite.site, targ_spin,color,RE);
								REAL left_i = (*(fine_vecs[ coarse_color ]))(fine_cbsite.cb, fine_cbsite.site, targ_spin,color,IM);

								REAL right_r = fine_in(fine_cbsite.cb,fine_cbsite.site,targ_spin,color,RE);
								REAL right_i = fine_in(fine_cbsite.cb,fine_cbsite.site,targ_spin,color,IM);

								// It is V_j^H  ferm_in so conj(left)*right.
								sum_cspin_re += left_r * right_r + left_i * right_i;
								sum_cspin_im += left_r * right_i - right_r * left_i;

							} // color
						}  // spin aggregates
					} // fine site
					coarse_site_spinor[ RE + n_complex*coarse_colorspin ] = sum_cspin_re;
					coarse_site_spinor[ IM + n_complex*coarse_colorspin ] = sum_cspin_im;
				} // coarse color
			} // chiral
		} // block_cbsites
	} // block_cb

}

//! 'Restrict' a QDP++ spinor to a CoarseSpinor with the same geometry
template<typename QS>
void restrictSpinor2T( const std::vector<Block>& blocklist, const std::vector< std::shared_ptr<QS> >& fine_vecs,
		const QS& fine_in, CoarseSpinor& out)
{

	const int num_coarse_cbsites = out.GetInfo().GetNumCBSites();
	const int num_coarse_color = out.GetNumColor();
	const int num_coarse_colorspin = out.GetNumColorSpin();

	// Sanity check. The number of sites in the coarse spinor
	// Has to equal the number of blocks
	//  assert( n_checkerboard*num_coarse_cbsites == static_cast<const int>(blocklist.size()) );

	// The number of vectors has to eaqual the number of coarse colors
	assert( fine_vecs.size() == num_coarse_color );

	// This will be a loop over blocks
#pragma omp parallel for collapse(2)
	for(int block_cb = 0; block_cb < n_checkerboard; ++block_cb ) {
		for(int block_cbsite = 0 ; block_cbsite < num_coarse_cbsites; ++block_cbsite) {

			// Identify the current block.
			int block_idx = block_cbsite + block_cb*num_coarse_cbsites;
			const Block& block = blocklist[block_idx];

			// Get the list of fine sites in the blocks
			auto block_sitelist = block.getCBSiteList();
			auto num_sites_in_block = block_sitelist.size();

			// The coarse site spinor is where we will write the result
			std::complex<float>* coarse_site_spinor = reinterpret_cast<std::complex<float>*>(out.GetSiteDataPtr(block_cb,block_cbsite));

			// The accumulation goes here
			std::vector< std::complex<float> > site_accum(num_coarse_colorspin);

			// The vector-s components go here
			std::vector< std::complex<float> > vs(num_coarse_colorspin);

			// The psi-s get broadcast here.
			std::vector< std::complex<float> > psi(num_coarse_colorspin);

			// Zero the accumulated coarse site
#pragma omp simd
			for(int i=0; i < num_coarse_colorspin; ++i) {
				site_accum[i] = std::complex<float>(0,0);
			}

			// Loop through the fine sites in the block
			for( IndexType fine_site_idx = 0; fine_site_idx < static_cast<IndexType>(num_sites_in_block); fine_site_idx++ ) {

				// Find the fine site
				const CBSite& fine_cbsite = block_sitelist[fine_site_idx];
				const int fine_site = (rb[ fine_cbsite.cb ].siteTable())[fine_cbsite.site ];

				// for each site we will loop over Ns/2 * Ncolor
				for(int spin=0; spin < 2; ++spin) {
					for(int color=0; color < 3; ++color) {
						// copy upper psis
						std::complex<float> psi_upper( fine_in(fine_cbsite.cb, fine_cbsite.site, spin, color, RE),
								fine_in(fine_cbsite.cb, fine_cbsite.site, spin, color, IM ));

						std::complex<float> psi_lower( fine_in(fine_cbsite.cb, fine_cbsite.site, spin+2, color, RE),
								fine_in(fine_cbsite.cb, fine_cbsite.site, spin+2, color, IM ));

						// Broadcast the psi-s
#pragma omp simd
						for(int colorspin=0; colorspin < num_coarse_color; ++colorspin ) {
							psi[colorspin] = psi_upper;
						}

#pragma omp simd
						for(int colorspin=0; colorspin < num_coarse_color; ++colorspin) {
							psi[num_coarse_color + colorspin] = psi_lower;
						}

						// For now, this is a gather, because fine_vecs are not contiguous
						// Later if we replace this with a data structure that runs
						//  restrictor[block_cb][num_blocks][sites_in_block][spin][color][chiral][Ns/2][Nc][n_vecs]
						// it ought to become a stream

						for(int colorspin=0; colorspin < num_coarse_color; ++colorspin ) {
							vs[colorspin] = std::complex<float>(
									(*(fine_vecs[ colorspin ]))(fine_cbsite.cb, fine_cbsite.site, spin,color,RE),
									-(*(fine_vecs[ colorspin ]))(fine_cbsite.cb, fine_cbsite.site, spin,color,IM) );
						}

						for(int colorspin=0; colorspin < num_coarse_color; ++colorspin ) {
							vs[colorspin+num_coarse_color] = std::complex<float>(
									(*(fine_vecs[ colorspin ]))(fine_cbsite.cb, fine_cbsite.site, spin+2,color,RE),
									-(*(fine_vecs[ colorspin ]))(fine_cbsite.cb, fine_cbsite.site, spin+2,color,IM) );
						}

#pragma omp simd
						for(int colorspin=0; colorspin < num_coarse_colorspin; colorspin++) {
							site_accum[colorspin] += vs[colorspin]*psi[colorspin];
						}

					} // color
				} // spin
			} // fine sites in block

#pragma omp simd
			for(int colorspin=0; colorspin <  num_coarse_colorspin; ++colorspin) {
				coarse_site_spinor[colorspin] = site_accum[colorspin];
			}
		}// block CBSITE
	} // block CB

}


void restrictSpinor( const std::vector<Block>& blocklist, const std::vector< std::shared_ptr<QPhiXSpinor> >& fine_vecs,
    const QPhiXSpinor& fine_in, CoarseSpinor& coarse_out)
{
  restrictSpinorT(blocklist, fine_vecs, fine_in,coarse_out);
}

void restrictSpinor( const std::vector<Block>& blocklist, const std::vector< std::shared_ptr<QPhiXSpinorF> >& fine_vecs,
    const QPhiXSpinorF& fine_in, CoarseSpinor& coarse_out)
{
  restrictSpinorT(blocklist, fine_vecs, fine_in,coarse_out);
}

void restrictSpinor2( const std::vector<Block>& blocklist, const std::vector< std::shared_ptr<QPhiXSpinor> >& fine_vecs,
    const QPhiXSpinor& fine_in, CoarseSpinor& coarse_out)
{
  restrictSpinor2T(blocklist, fine_vecs, fine_in,coarse_out);
}

void restrictSpinor2( const std::vector<Block>& blocklist, const std::vector< std::shared_ptr<QPhiXSpinorF> >& fine_vecs,
    const QPhiXSpinorF& fine_in, CoarseSpinor& coarse_out)
{
  restrictSpinor2T(blocklist, fine_vecs, fine_in,coarse_out);
}

//! 'Prolongate' a CoarseSpinor to a QDP++ Fine Spinor
template<typename QS>
void prolongateSpinorT(const std::vector<Block>& blocklist,
    const std::vector<std::shared_ptr<QS> >& fine_vecs,
    const CoarseSpinor& coarse_in, QS& fine_out)
{
  // Prolongate in here
  IndexType num_coarse_cbsites=coarse_in.GetInfo().GetNumCBSites();

  // assert( num_coarse_cbsites == static_cast<IndexType>(blocklist.size()/2) );

  IndexType num_coarse_color = coarse_in.GetNumColor();
  assert( static_cast<IndexType>(fine_vecs.size()) == num_coarse_color);

  // NB: Parallelism wise, this is a scatter. Because we are visiting each block
  // and keeping it fixed we write out all the fine sites in the block which will not
  // be contiguous. One potential optimization is to turn this into a gather...
  // Then we would need to loop through all the fine sites in order. Our current blocklist
  // Only contains coarse site -> list of fine sites
  // The inverse mapping of fine site-> block (many fine sites to same block)
  // does not exist. We could create it ...

  // Loop over the coarse sites (blocks)
  // Do this with checkerboarding, because of checkerboarded index for
#pragma omp parallel for collapse(2)
  for(int block_cb = 0; block_cb < n_checkerboard; ++block_cb ) {
    for(int block_cbsite = 0 ; block_cbsite < num_coarse_cbsites; ++block_cbsite) {


      for(int fine_spin=0; fine_spin < Ns; ++fine_spin) {
        for(int fine_color=0; fine_color < Nc; fine_color++ ) {

          int block_idx = block_cbsite + block_cb*num_coarse_cbsites;
          const float *coarse_spinor = coarse_in.GetSiteDataPtr(block_cb,block_cbsite);

          // Get the list of sites in the block
          auto fine_sitelist = blocklist[block_idx].getCBSiteList();
          auto num_fine_sitelist = fine_sitelist.size();

          int chiral = fine_spin < (Ns/2) ? 0 : 1;

          for(int fine_site_idx = 0; fine_site_idx < num_fine_sitelist; ++fine_site_idx) {

            const CBSite& fine_cbsite = fine_sitelist[fine_site_idx];
            int qdpsite = (rb[fine_cbsite.cb].siteTable())[ fine_cbsite.site ] ;

           // fine_out(fine_cbsite.cb, fine_cbsite.site,fine_spin,fine_color,RE) = 0;
           // fine_out(fine_cbsite.cb, fine_cbsite.site,fine_spin,fine_color,IM) = 0;
            float csum_re = 0;
            float csum_im = 0;

#pragma omp simd reduction(+:csum_re,csum_im)
            for(int coarse_color = 0; coarse_color < num_coarse_color; coarse_color++) {

              REAL left_r = (*fine_vecs[coarse_color])(fine_cbsite.cb,fine_cbsite.site,fine_spin,fine_color,RE);
              REAL left_i = (*fine_vecs[coarse_color])(fine_cbsite.cb,fine_cbsite.site,fine_spin,fine_color,IM);

              int colorspin = coarse_color + chiral*num_coarse_color;
              REAL right_r = coarse_spinor[ RE + n_complex*colorspin];
              REAL right_i = coarse_spinor[ IM + n_complex*colorspin];

              // V_j | out  (rather than V^{H}) so needs regular complex mult?
              csum_re += left_r * right_r - left_i * right_i;
              csum_im += left_i * right_r + left_r * right_i;
            }
            fine_out(fine_cbsite.cb, fine_cbsite.site,fine_spin,fine_color,RE) = csum_re;
            fine_out(fine_cbsite.cb, fine_cbsite.site,fine_spin,fine_color,IM) = csum_im;

          }
        }
      }
    }
  }
}



void prolongateSpinor(const std::vector<Block>& blocklist,
    const std::vector<std::shared_ptr<QPhiXSpinor> >& fine_vecs,
    const CoarseSpinor& coarse_in, QPhiXSpinor& fine_out)
{

  prolongateSpinorT(blocklist,fine_vecs, coarse_in, fine_out);
}

void prolongateSpinor(const std::vector<Block>& blocklist,
    const std::vector<std::shared_ptr<QPhiXSpinorF> >& fine_vecs,
    const CoarseSpinor& coarse_in, QPhiXSpinorF& fine_out)
{
  prolongateSpinorT(blocklist,fine_vecs, coarse_in,fine_out);
}





//! Coarsen one direction of a 'dslash' link
template<typename DiracOperator, typename QS>
inline
void dslashTripleProductDirT(const DiracOperator& D_op,
		const std::vector<Block>& blocklist, int dir,
		const std::vector<std::shared_ptr<QS> >& in_vecs,
		CoarseGauge& u_coarse)
{
	// Dslash triple product in here
	// Dslash triple product in here


	int num_coarse_colors = u_coarse.GetNumColor();
	int num_coarse_colorspin = u_coarse.GetNumColorSpin();

	int num_coarse_cbsites = u_coarse.GetInfo().GetNumCBSites();
	const int n_chiral = 2;
	const int num_spincolor_per_chiral = (Nc*Ns)/n_chiral;
	const int num_spin_per_chiral = Ns/n_chiral;

	const LatticeInfo& c_info=u_coarse.GetInfo();
	IndexArray coarse_dims = c_info.GetLatticeDimensions();

	const LatticeInfo& fine_info = in_vecs[0]->GetInfo();


	// in vecs has size Ncolor_c = num_coarse_colorspin/2
	// But this mixes both upper and lower spins
	// Once we deal with those separately we will need num_coarse_colorspin results
	// And we will need to apply the 'DslashDir' separately to each aggregate

	assert( in_vecs.size() == num_coarse_colors);
	std::vector<std::shared_ptr<QS> > out_vecs(num_coarse_colorspin);
	for (int j = 0; j < num_coarse_colorspin; ++j) {
		out_vecs[j] = std::make_shared<QS>(fine_info);
		ZeroVec(*(out_vecs[j]));
	}


	// Apply DslashDir to each aggregate separately.
	// DslashDir may mix spins with (1 +/- gamma_mu)
	for(int j=0; j < num_coarse_colors; ++j) {
		for(int aggr=0; aggr < n_chiral; ++aggr) {
			QS tmp(fine_info); ZeroVec(tmp);
			extractAggregate(tmp, *(in_vecs[j]), aggr);
			D_op.DslashDir(*(out_vecs[aggr*num_coarse_colors+j]), tmp, dir);
		}
	}

	// Loop over the coarse sites (blocks)
#pragma omp parallel for collapse(2)
	for(int coarse_cb=0; coarse_cb < n_checkerboard; ++coarse_cb) {
		for(int coarse_cbsite =0; coarse_cbsite < num_coarse_cbsites; ++coarse_cbsite) {



			float *coarse_link = u_coarse.GetSiteDirDataPtr(coarse_cb,coarse_cbsite, dir);



			int block_idx = coarse_cbsite + coarse_cb*num_coarse_cbsites;
			const Block& block = blocklist[ block_idx ]; // Get the block


			//  --------------------------------------------
			//  Now do the faces of the Block
			//  ---------------------------------------------
			auto face_sitelist = block.getFaceList(dir);
			auto  num_sites = face_sitelist.size();



			for(IndexType fine_site_idx = 0; fine_site_idx < static_cast<IndexType>(num_sites); ++fine_site_idx) {

				CBSite& fine_cbsite = face_sitelist[ fine_site_idx ];
				int qdp_site = rb[ fine_cbsite.cb ].siteTable()[ fine_cbsite.site ] ;
				multi1d<int> qdp_coords = Layout::siteCoords(Layout::nodeNumber(), qdp_site);

				// Offset by the aggr_row and aggr_column
				for(int aggr_row=0; aggr_row < n_chiral; ++aggr_row) {
					for(int aggr_col=0; aggr_col <n_chiral; ++aggr_col ) {
						for(int matmul_row=0; matmul_row < num_coarse_colors; ++matmul_row) {
							for(int matmul_col=0; matmul_col < num_coarse_colors; ++matmul_col) {

								int row = aggr_row*num_coarse_colors + matmul_row;
								int col = aggr_col*num_coarse_colors + matmul_col;

								//Index in coarse link
								int coarse_link_index = n_complex*(row+ num_coarse_colorspin*col);

								double tmp_sum_re = 0;
								double tmp_sum_im = 0;

								// Inner product loop
								for(int k=0; k < num_spincolor_per_chiral; ++k) {



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
									float right_r = (*(out_vecs[col]))(fine_cbsite.cb,fine_cbsite.site,spin,color,RE);
									float right_i = (*(out_vecs[col]))(fine_cbsite.cb,fine_cbsite.site,spin,color,IM);

									// Left vector -- only num_coarse_colors components with [ V^H_upper V^H_lower ]
									//
									// ie a compact storage
									// rather than:
									//
									// [ V^H_upper   0      ]
									// [  0       V^H_lower ]
									//
									// so index with row % num_coarse_colors = matmul_row
									float left_r = (*(in_vecs[matmul_row]))(fine_cbsite.cb,fine_cbsite.site,spin,color,RE);
									float left_i = (*(in_vecs[matmul_row]))(fine_cbsite.cb,fine_cbsite.site,spin,color,IM);

									// Accumulate inner product V^H_row A_column
									tmp_sum_re += (left_r*right_r + left_i*right_i);
									tmp_sum_im += (left_r*right_i - right_r*left_i);
								} // k
								coarse_link[RE + coarse_link_index ] += tmp_sum_re;
								coarse_link[IM + coarse_link_index ] += tmp_sum_im;
							} // matmul_col
						} // matmul_row
					} // aggr_col
				} // aggr_row
			} // fine_site_idx

		} // cbsite
	}// cb

	//  --------------------------------------------
	//  Now do the Not Faces faces
	//  ---------------------------------------------
#pragma omp parallel for collapse (2)
	for(int coarse_cb=0; coarse_cb < n_checkerboard; ++coarse_cb) {
		for(int coarse_cbsite =0; coarse_cbsite < num_coarse_cbsites; ++coarse_cbsite) {

			int block_idx = coarse_cbsite + coarse_cb*num_coarse_cbsites;


			// Get a Block Index

			const Block& block = blocklist[ block_idx ]; // Get the block


			auto not_face_sitelist = block.getNotFaceList(dir);
			auto  num_sites = not_face_sitelist.size();


			// Get teh coarse site for writing
			// Thiis is fixed
			float *coarse_link = u_coarse.GetSiteDiagDataPtr(coarse_cb,coarse_cbsite);

			for(IndexType fine_site_idx = 0; fine_site_idx < static_cast<IndexType>(num_sites); ++fine_site_idx) {

				CBSite& fine_cbsite = not_face_sitelist[ fine_site_idx ];
				int qdp_site = rb[ fine_cbsite.cb ].siteTable()[ fine_cbsite.site ] ;
				multi1d<int> qdp_coords = Layout::siteCoords(Layout::nodeNumber(), qdp_site);

				// Inner product loop
				for(int aggr_row=0; aggr_row < n_chiral; ++aggr_row) {
					for(int aggr_col=0; aggr_col <n_chiral; ++aggr_col ) {
						for(int matmul_row=0; matmul_row < num_coarse_colors; ++matmul_row) {
							for(int matmul_col=0; matmul_col < num_coarse_colors; ++matmul_col) {

								// Offset by the aggr_row and aggr_column
								int row = aggr_row*num_coarse_colors + matmul_row;
								int col = aggr_col*num_coarse_colors + matmul_col;

								//Index in coarse link
								int coarse_link_index = n_complex*(row + num_coarse_colorspin*col);
								double tmp_sum_re =0;
								double tmp_sum_im =0;

								for(int k=0; k < num_spincolor_per_chiral; ++k) {


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
									float right_r = (*(out_vecs[col]))(fine_cbsite.cb,fine_cbsite.site,spin,color,RE);
									float right_i = (*(out_vecs[col]))(fine_cbsite.cb,fine_cbsite.site,spin,color,IM);
									// Left vector -- only num_coarse_colors components with [ V^H_upper V^H_lower ]
									//
									// ie a compact storage
									// rather than:
									//
									// [ V^H_upper   0      ]
									// [  0       V^H_lower ]
									//
									// so index with row % num_coarse_colors = matmul_row
									float left_r = (*(in_vecs[matmul_row]))(fine_cbsite.cb,fine_cbsite.site,spin,color,RE);
									float left_i = (*(in_vecs[matmul_row]))(fine_cbsite.cb,fine_cbsite.site,spin,color,IM);

									// Accumulate inner product V^H_row A_column
									tmp_sum_re += (left_r*right_r + left_i*right_i);
									tmp_sum_im += (left_r*right_i - right_r*left_i);
								} // k

								// Accumulate inner product V^H_row A_column
								coarse_link[RE + coarse_link_index ] += tmp_sum_re;
								coarse_link[IM + coarse_link_index ] += tmp_sum_im;

							} // matmul_col
						} // matmul_row
					} // aggr_col
				} // aggr_row
			} // fine_site_idx

		} // coarse cbsite
	} //coarse_cb

}

//! Coarsen one direction of a 'dslash' link
void dslashTripleProductDir(const QPhiXWilsonCloverLinearOperator& D_op,
    const std::vector<Block>& blocklist, int dir,
    const std::vector<std::shared_ptr<QPhiXSpinor> >& in_vecs,
    CoarseGauge& u_coarse)
{
  dslashTripleProductDirT(D_op,blocklist,dir,in_vecs,u_coarse);
}

void dslashTripleProductDir(const QPhiXWilsonCloverLinearOperatorF& D_op,
    const std::vector<Block>& blocklist, int dir,
    const std::vector<std::shared_ptr<QPhiXSpinorF> >& in_vecs,
    CoarseGauge& u_coarse)
{
  dslashTripleProductDirT(D_op,blocklist,dir,in_vecs,u_coarse);
}


//! Coarsen one direction of a 'dslash' link
void dslashTripleProductDir(const QPhiXWilsonCloverEOLinearOperator& D_op,
    const std::vector<Block>& blocklist, int dir,
    const std::vector<std::shared_ptr<QPhiXSpinor> >& in_vecs,
    CoarseGauge& u_coarse)
{
  dslashTripleProductDirT(D_op,blocklist,dir,in_vecs,u_coarse);
}

void dslashTripleProductDir(const QPhiXWilsonCloverEOLinearOperatorF& D_op,
    const std::vector<Block>& blocklist, int dir,
    const std::vector<std::shared_ptr<QPhiXSpinorF> >& in_vecs,
    CoarseGauge& u_coarse)
{
  dslashTripleProductDirT(D_op,blocklist,dir,in_vecs,u_coarse);
}

//! Coarsen the clover term (1 block = 1 site )
//! Coarsen the clover term (1 block = 1 site )
template<typename DiracOpType, typename QS>
inline
void clovTripleProductT(const DiracOpType& D_op,
    const std::vector<Block>& blocklist,
    const std::vector<std::shared_ptr<QS> >& in_vecs,
    CoarseGauge& gauge_clover)
{
  // Clover Triple product in here
  int num_coarse_colors = gauge_clover.GetNumColor();
  int num_chiral_components = 2;
  int num_coarse_colorspin = num_coarse_colors*num_chiral_components;
  int num_coarse_cbsites = gauge_clover.GetInfo().GetNumCBSites();
  const int num_spincolor_per_chiral = (Nc*Ns)/num_chiral_components;
  const int num_spin_per_chiral = Ns/num_chiral_components;

  // in vecs has size num_coarse_colors = Ncolorspin_c/2
  // But this mixes both upper and lower spins
  // Once we deal with those separately we will need Ncolorspin_c results
  // And we will need to apply the 'DslashDir' separately to each aggregate

  assert( in_vecs.size() == num_coarse_colors );
  assert( num_chiral_components == 2);

  // out_vecs is the result of applying clover term to in_vecs
  // NOTE!!!: Unlike with Dslash where (1 +/- gamma_mu) mixes the upper and lower spin components
  // Clover *does not* do this. In this chiral basis that we use Clover is block diagonal
  // So it acts independently on upper and lower spin components.
  // This means Ncolor vectors are sufficient. The upper components will hold the results of
  // clover_term applied to the upper components while the lower components will hold the results of
  // clover_term applied to the lower components in the same way in_vector combines upper and lower
  // components.
  const LatticeInfo& fine_info = in_vecs[0]->GetInfo();

  std::vector<std::shared_ptr<QS> > out_vecs(num_coarse_colorspin);
   for (int j = 0; j < num_coarse_colorspin; ++j) {
     out_vecs[j] = std::make_shared<QS>(fine_info);
     ZeroVec(*(out_vecs[j]));
   }

  // for each in-vector pull out respectively the lower and upper spins
  // multiply by clover and store in out_vecs. There will be num_coarse_colors*num_chiral_components output
  // vectors
  for(int j=0; j < num_coarse_colors; ++j) {

    // Clover term is block diagonal
    // So I can apply it once, the upper and lower spin components will
    // be acted on independently. No need to separate the aggregates before
    // applying
    D_op.M_ee(*(out_vecs[j]), *(in_vecs[j]), LINOP_OP);
    D_op.M_oo(*(out_vecs[j]), *(in_vecs[j]), LINOP_OP);

  }


  // Technically these outer loops should be over all the blocks.
#pragma omp parallel for collapse(2)
  for(int coarse_cb=0; coarse_cb < n_checkerboard; ++coarse_cb) {
    for(int coarse_cbsite=0; coarse_cbsite < num_coarse_cbsites; ++coarse_cbsite) {


    	 int block_idx = coarse_cbsite + coarse_cb*num_coarse_cbsites;
    	 const Block& block = blocklist[block_idx];
    	 auto block_sitelist = block.getCBSiteList();
    	 auto num_block_sites = block.getNumSites();


    	 float *coarse_clov = gauge_clover.GetSiteDiagDataPtr(coarse_cb,coarse_cbsite);
    	 //for(int i=0; i < 2*num_coarse_colorspin*num_coarse_colorspin; i++) {
    		// coarse_clov[i] = 0;
    	// }

    	 for(IndexType fine_site_idx=0; fine_site_idx < static_cast<IndexType>(num_block_sites); ++fine_site_idx) {
    		 // Inner product loop

    		 const CBSite& fine_cbsite = block_sitelist[ fine_site_idx ];
    		 int site = rb[ fine_cbsite.cb ].siteTable()[ fine_cbsite.site ];

    		 for(int chiral =0; chiral < num_chiral_components; ++chiral) {
    			 for(int matmul_row=0; matmul_row < num_coarse_colors; ++matmul_row) {
    				 for(int matmul_col=0; matmul_col < num_coarse_colors; ++matmul_col) {

    					 int row_col_min = (chiral == 0) ? 0 : num_coarse_colors;
    					 int outrow = matmul_row + row_col_min;
    					 int outcol = matmul_col + row_col_min;
    					 int coarse_clov_index = n_complex*(outrow+ num_coarse_colorspin*outcol);

    					 double tmp_sum_re =0;
    					 double tmp_sum_im =0;

    					 for(int k=0; k < num_spincolor_per_chiral; ++k) {

    						 // [ 		V^H_upper   0      ] [  A_upper    B_upper ] = [ V^H_upper A_upper   V^H_upper B_upper  ]
    						 // [  0       V^H_lower ] [  A_lower    B_lower ]   [ V^H_lower A_lower   V^H_lower B_lower  ]
    						 //
    						 // But
    						 //  [ A_upper B_upper ] = [ Clov_upper      0      ] [ V_upper        0    ] = [ A_upper    0    ]
    						 //  [ A_lower B_lower ]   [      0      Clov_lower ] [    0       V_lower  ]   [   0     B_lower ]
    						 //
    						 // So really I need to just evaluate:  V^H_upper A_upper and V^H_lower B_lower
    						 //
    						 //
    						 int spin=k/Nc + chiral * num_spin_per_chiral;  // Upper or lower spin depending on chiral

    						 // k % Nc maps to color component (0,1,2)
    						 int color=k%Nc;

    						 // Right vector - chiral*num_coarse_colors selects A_upper ( chiral=0 ) or B_lower (chiral=1)

    						 // NB: Out vecs has only NColor members
    						 float right_r = (*(out_vecs[matmul_col]))(fine_cbsite.cb, fine_cbsite.site,spin,color,RE);
    						 float right_i = (*(out_vecs[matmul_col]))(fine_cbsite.cb, fine_cbsite.site,spin,color,IM);

    						 // Left vector -- only num_coarse_colors components with [ V^H_upper V^H_lower ]
    						 //
    						 // ie a compact storage
    						 // rather than:
    						 //
    						 // [ V^H_upper   0      ]
    						 // [    0       V^H_lower ]
    						 //
    						 // so index with row % num_coarse_colors = matmul_row
    						 float left_r = (*(in_vecs[matmul_row ]))(fine_cbsite.cb, fine_cbsite.site, spin, color, RE);
    						 float left_i = (*(in_vecs[matmul_row ]))(fine_cbsite.cb, fine_cbsite.site, spin, color, IM);

    						 // Accumulate inner product V^H_row A_column
    						 tmp_sum_re += (double)(left_r*right_r + left_i*right_i);
    						 tmp_sum_im += (double)(left_r*right_i - right_r*left_i);
    					 } // k

    					 coarse_clov[RE + coarse_clov_index ] += tmp_sum_re;
    					 coarse_clov[IM + coarse_clov_index ] += tmp_sum_im;
    				 } // matmul col
    			 } // matmul row
    		 }// chiral
    	 } // fine_site_idx
    } // coarse_site
  } // coarse_cb

}


void clovTripleProduct(const QPhiXWilsonCloverLinearOperator& D_op,
    const std::vector<Block>& blocklist,
    const std::vector<std::shared_ptr<QPhiXSpinor> >& in_vecs,
    CoarseGauge& gauge_clover)
{
  clovTripleProductT(D_op, blocklist, in_vecs, gauge_clover);
}

void clovTripleProduct(const QPhiXWilsonCloverLinearOperatorF& D_op,
    const std::vector<Block>& blocklist,
    const std::vector<std::shared_ptr<QPhiXSpinorF> >& in_vecs,
    CoarseGauge& gauge_clover)
{
  clovTripleProductT(D_op,blocklist, in_vecs, gauge_clover);
}

void clovTripleProduct(const QPhiXWilsonCloverEOLinearOperator& D_op,
    const std::vector<Block>& blocklist,
    const std::vector<std::shared_ptr<QPhiXSpinor> >& in_vecs,
    CoarseGauge& gauge_clover)
{
  clovTripleProductT(D_op, blocklist, in_vecs, gauge_clover);
}

void clovTripleProduct(const QPhiXWilsonCloverEOLinearOperatorF& D_op,
    const std::vector<Block>& blocklist,
    const std::vector<std::shared_ptr<QPhiXSpinorF> >& in_vecs,
    CoarseGauge& gauge_clover)
{
  clovTripleProductT(D_op,blocklist, in_vecs, gauge_clover);
}

}
