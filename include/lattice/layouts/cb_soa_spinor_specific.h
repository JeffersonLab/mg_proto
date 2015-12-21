/*
 * cb_soa_spinor_specific.h
 *
 *  Created on: Nov 10, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_CB_SOA_SPINOR_SPECIFIC_H_
#define INCLUDE_LATTICE_CB_SOA_SPINOR_SPECIFIC_H_

#include "lattice/constants.h"
#include "lattice/layouts/cb_soa_spinor_layout.h"
#include "lattice/layout_container.h"
#include "lattice/contiguous_blas.h"

// This is using C++ random number engine for testing
#include <random>


namespace MGGeometry {

	template<typename T, size_t blocksize = MG_DEFAULT_ALIGNMENT>
	void Zero(GenericLayoutContainer<T,CBSOASpinorLayout<T>>& spinor)
	{
		const IndexType num_spins = spinor.GetLatticeInfo().GetNumSpins();
		const IndexType num_colors = spinor.GetLatticeInfo().GetNumColors();
		const IndexType num_cb = n_checkerboard;
		const IndexType num_cbsites = spinor.GetLatticeInfo().GetNumCBSites();

		/* Problem: We can distribute threading and vectors in various ways.
		 * We can definitely compute in parallel the NormSq over each spin and color component in
		 * each checkerboard. However, in the case of the fine lattice this does not have a lot of
		 * parallelism. 4*3*2 = 24 way. However, on the fine lattice there may be a lot of local sites,
		 * e.g. 4^4. We want to vectorize over these, but also feed additional threads.
		 *
		 * In single precision, for a vector length of 16 we need 8 complexes.
		 * So	 the idea is to chunk the number of checkerboarded sites into blocks
		 * the l	ength of which corresponds to a cache line, which is at least 1 vector.
		 * We can then parallelize also over the chunks.
		 */

		const IndexType sites_per_block = blocksize / sizeof(std::complex<T>);

		// This many sites
		const IndexType num_blocks =
				(num_cbsites % sites_per_block == 0) ?
						(num_cbsites / sites_per_block) :
						(num_cbsites / sites_per_block) + 1;

#pragma omp parallel for collapse(4)
		for (IndexType spin = 0; spin < num_spins; ++spin) {
			for (IndexType color = 0; color < num_colors; ++color) {
				for (IndexType cb = 0; cb < num_cb; ++cb) {
					for (IndexType block = 0; block < num_blocks; ++block) {

						/* Work out the start and end sites taking into account that the last iteration may not be full */
						IndexType start_site = block * sites_per_block;
						IndexType end_site =
								(block + 1) * sites_per_block > num_cbsites ?
										num_cbsites : (block + 1) * sites_per_block;

						T* data = & spinor.Index(cb,start_site,spin,color,0);
#pragma omp simd
						for(IndexType site=0; site < 2*(end_site-start_site); ++site) {
							data[site] = 0;
						}

					}
				}
			}
		}
	}

	template<typename T, size_t blocksize = MG_DEFAULT_ALIGNMENT>
	void Fill(const std::complex<T>& scalar, GenericLayoutContainer<T,CBSOASpinorLayout<T>>& spinor)
	{
		const IndexType num_spins = spinor.GetLatticeInfo().GetNumSpins();
		const IndexType num_colors = spinor.GetLatticeInfo().GetNumColors();
		const IndexType num_cb = n_checkerboard;
		const IndexType num_cbsites = spinor.GetLatticeInfo().GetNumCBSites();

		/* Problem: We can distribute threading and vectors in various ways.
		 * We can definitely compute in parallel the NormSq over each spin and color component in
		 * each checkerboard. However, in the case of the fine lattice this does not have a lot of
		 * parallelism. 4*3*2 = 24 way. However, on the fine lattice there may be a lot of local sites,
		 * e.g. 4^4. We want to vectorize over these, but also feed additional threads.
		 *
		 * In single precision, for a vector length of 16 we need 8 complexes.
		 * So	 the idea is to chunk the number of checkerboarded sites into blocks
		 * the l	ength of which corresponds to a cache line, which is at least 1 vector.
		 * We can then parallelize also over the chunks.
		 */

		const IndexType sites_per_block = blocksize / sizeof(std::complex<T>);

		const IndexType num_blocks =
				(num_cbsites % sites_per_block == 0) ?
						(num_cbsites / sites_per_block) :
						(num_cbsites / sites_per_block) + 1;


#pragma omp parallel for collapse(4)
		for (IndexType spin = 0; spin < num_spins; ++spin) {
			for (IndexType color = 0; color < num_colors; ++color) {
				for (IndexType cb = 0; cb < num_cb; ++cb) {
					for (IndexType block = 0; block < num_blocks; ++block) {

						/* Work out the start and end sites taking into account that the last iteration may not be full */
						IndexType start_site = block * sites_per_block;
						IndexType end_site =
								(block + 1) * sites_per_block > num_cbsites ?
										num_cbsites : (block + 1) * sites_per_block;

						std::complex<T>* data = reinterpret_cast<std::complex<double>*>(& spinor.Index(cb,start_site,spin,color,0));
#pragma omp simd
						for(IndexType site=0; site < (end_site-start_site); ++site) {
							data[site] = scalar;
						}

					}
				}
			}
		}
	}

	template<typename T,
		     typename Generator,
			 size_t blocksize = MG_DEFAULT_ALIGNMENT>
	void FillGaussian(GenericLayoutContainer<T,CBSOASpinorLayout<T>>& spinor, Generator& g) {


		std::normal_distribution<> gaussian_distribution;

		const IndexType num_spins = spinor.GetLatticeInfo().GetNumSpins();
		const IndexType num_colors = spinor.GetLatticeInfo().GetNumColors();
		const IndexType num_cb = n_checkerboard;
		const IndexType num_cbsites = spinor.GetLatticeInfo().GetNumCBSites();

		/* Problem: We can distribute threading and vectors in various ways.
		 * We can definitely compute in parallel the NormSq over each spin and color component in
		 * each checkerboard. However, in the case of the fine lattice this does not have a lot of
		 * parallelism. 4*3*2 = 24 way. However, on the fine lattice there may be a lot of local sites,
		 * e.g. 4^4. We want to vectorize over these, but also feed additional threads.
		 *
		 * In single precision, for a vector lenght of 16 we need 8 complexes.
		 * So the idea is to chunk the number of checkerboarded sites into blocks
		 * the length of which corresponds to a cache line, which is at least 1 vector.
		 * We can then parallelize also over the chunks.
		 */


		/* In this layout a site is a complex. So the number of sites in the block is the block size
		 * in bytes divided by the size of a complex
		 */
		const IndexType sites_per_block = blocksize/sizeof(std::complex<T>);

		/* Compute the number of blocks. If there is a remainder (the % op != 0) add an extra block. */
		const IndexType num_blocks = (num_cbsites%sites_per_block == 0) ? (num_cbsites / sites_per_block) : (num_cbsites/sites_per_block)+1;

#pragma omp parallel for collapse(4)
		for(IndexType spin=0; spin < num_spins; ++spin) {
			for(IndexType color=0; color < num_colors; ++color) {
				for(IndexType cb=0; cb < num_cb; ++cb)  {
					for(IndexType block=0; block < num_blocks; ++block) {


						/* Find the start site of the block */
						IndexType start_site = block*sites_per_block;

						/* Find the end site of the block. In the last block, we may not fill the whole block */
						IndexType end_site = (block+1)*sites_per_block > num_cbsites ? num_cbsites : (block+1)*sites_per_block;

						for(IndexType site=start_site; site < end_site; ++site) {
							spinor.Index(cb,site,spin,color,RE) = gaussian_distribution(g);
							spinor.Index(cb,site,spin,color,IM) = gaussian_distribution(g);
						}
					}
				}
			}
		}

	}


	template<typename T, size_t blocksize = MG_DEFAULT_ALIGNMENT>
	double NormSq(const GenericLayoutContainer<T,CBSOASpinorLayout<T>>& spinor) {



		const IndexType num_spins = spinor.GetLatticeInfo().GetNumSpins();
		const IndexType num_colors = spinor.GetLatticeInfo().GetNumColors();
		const IndexType num_cb = n_checkerboard;
		const IndexType num_cbsites = spinor.GetLatticeInfo().GetNumCBSites();

		/* Problem: We can distribute threading and vectors in various ways.
		 * We can definitely compute in parallel the NormSq over each spin and color component in
		 * each checkerboard. However, in the case of the fine lattice this does not have a lot of
		 * parallelism. 4*3*2 = 24 way. However, on the fine lattice there may be a lot of local sites,
		 * e.g. 4^4. We want to vectorize over these, but also feed additional threads.
		 *
		 * In single precision, for a vector lenght of 16 we need 8 complexes.
		 * So the idea is to chunk the number of checkerboarded sites into blocks
		 * the length of which corresponds to a cache line, which is at least 1 vector.
		 * We can then parallelize also over the chunks.
		 */


		/* In this layout a site is a complex. So the number of sites in the block is the block size
		 * in bytes divided by the size of a complex
		 */
		const IndexType sites_per_block = blocksize/sizeof(std::complex<T>);

		/* Compute the number of blocks. If there is a remainder (the % op != 0) add an extra block. */
		const IndexType num_blocks = (num_cbsites%sites_per_block == 0) ? (num_cbsites / sites_per_block) : (num_cbsites/sites_per_block)+1;


		double ret_val = 0;
#pragma omp parallel for collapse(4) reduction(+:ret_val)
		for(IndexType spin=0; spin < num_spins; ++spin) {
			for(IndexType color=0; color < num_colors; ++color) {
				for(IndexType cb=0; cb < num_cb; ++cb)  {
					for(IndexType block=0; block < num_blocks; ++block) {


						/* Find the start site of the block */
						IndexType start_site = block*sites_per_block;

						/* Find the end site of the block. In the last block, we may not fill the whole block */
						IndexType end_site = (block+1)*sites_per_block > num_cbsites ? num_cbsites : (block+1)*sites_per_block;

						/* Find the address of the data in the block -- NB: This should be MG_DEFAULT_ALIGMENT aligned
						 * since the blocks are multiples of MG_DEFAULT_ALIGMENT bytes, and each cb is also aligned on cbsite=0*/
						const T* data = & spinor.Index(cb,start_site,spin,color,0);

						/* Call the vectorized NormSq in contiguous_blas.h. ret_val is  a reduction. */

						ret_val += NormSq( data,n_complex*(end_site-start_site));
					}
				}
			}
		}

		return ret_val;
	}


	template<typename T, size_t blocksize=MG_DEFAULT_ALIGNMENT>
	std::complex<double>
	InnerProduct(const GenericLayoutContainer<T,CBSOASpinorLayout<T>>& left, GenericLayoutContainer<T, CBSOASpinorLayout<T>>& right) {

//			FIXME: I should assert that the spinors are conformant in terms of their colors, spins and number of sites
		AssertCompatible( left.GetLatticeInfo(), right.GetLatticeInfo() );

		const IndexType num_spins = left.GetLatticeInfo().GetNumSpins();
		const IndexType num_colors = left.GetLatticeInfo().GetNumColors();
		const IndexType num_cb = n_checkerboard;
		const IndexType num_cbsites = left.GetLatticeInfo().GetNumCBSites();

		/* Problem: We can distribute threading and vectors in various ways.
		 * We can definitely compute in parallel the NormSq over each spin and color component in
		 * each checkerboard. However, in the case of the fine lattice this does not have a lot of
		 * parallelism. 4*3*2 = 24 way. However, on the fine lattice there may be a lot of local sites,
		 * e.g. 4^4. We want to vectorize over these, but also feed additional threads.
		 *
		 * In single precision, for a vector length of 16 we need 8 complexes.
		 * So the idea is to chunk the number of checkerboarded sites into blocks
		 * the length of which corresponds to a cache line, which is at least 1 vector.
		 * We can then parallelize also over the chunks.
		 */

		const IndexType sites_per_block = blocksize/sizeof(std::complex<T>);


		// This many sites
		const IndexType num_blocks = (num_cbsites%sites_per_block == 0) ? (num_cbsites / sites_per_block) : (num_cbsites/sites_per_block)+1;


		/* I bet I can't do an openmp reduction over a std::complex<> so let us not even try */

		double ret_val_re=0;
		double ret_val_im=0;

#pragma omp parallel for collapse(4) reduction(+:ret_val_re) reduction(+:ret_val_im)
		for(IndexType spin=0; spin < num_spins; ++spin) {
			for(IndexType color=0; color < num_colors; ++color) {
				for(IndexType cb=0; cb < num_cb; ++cb)  {
					for(IndexType block=0; block < num_blocks; ++block) {

						/* Work out the start and end sites taking into account that the last iteration may not be full */
						IndexType start_site = block*sites_per_block;
						IndexType end_site = (block+1)*sites_per_block > num_cbsites ? num_cbsites : (block+1)*sites_per_block;
						const T* l_data = & (left.Index(cb,start_site,spin,color,0));
						const T* r_data = & (right.Index(cb,start_site,spin,color,0));

						// Vectorized Inner over block
						std::complex<double> priv_sum = InnerProduct(l_data, r_data, (end_site-start_site));
						ret_val_re += priv_sum.real();
						ret_val_im += priv_sum.imag();

					}
				}
			}
		}

		// Turn ret_val re and ret_val im into a complex and return;
		return std::complex<double>{ret_val_re,ret_val_im};


	}

	/* Scale a spinor */
	template<typename T, size_t blocksize=MG_DEFAULT_ALIGNMENT >
	void
	VScale(const T& scalar, GenericLayoutContainer<T,CBSOASpinorLayout<T>>& vector)
	{
		const IndexType num_spins = vector.GetLatticeInfo().GetNumSpins();
		const IndexType num_colors = vector.GetLatticeInfo().GetNumColors();
		const IndexType num_cb = n_checkerboard;
		const IndexType num_cbsites = vector.GetLatticeInfo().GetNumCBSites();

		/* Problem: We can distribute threading and vectors in various ways.
		 * We can definitely compute in parallel the NormSq over each spin and color component in
		 * each checkerboard. However, in the case of the fine lattice this does not have a lot of
		 * parallelism. 4*3*2 = 24 way. However, on the fine lattice there may be a lot of local sites,
		 * e.g. 4^4. We want to vectorize over these, but also feed additional threads.
		 *
		 * In single precision, for a vector length of 16 we need 8 complexes.
		 * So	 the idea is to chunk the number of checkerboarded sites into blocks
		 * the l	ength of which corresponds to a cache line, which is at least 1 vector.
		 * We can then parallelize also over the chunks.
		 */

		const IndexType sites_per_block = blocksize/sizeof(std::complex<T>);

		// This many sites
		const IndexType num_blocks = (num_cbsites%sites_per_block == 0) ? (num_cbsites / sites_per_block) : (num_cbsites/sites_per_block)+1;

#pragma omp parallel for collapse(4)
		for(IndexType spin=0; spin < num_spins; ++spin) {
			for(IndexType color=0; color < num_colors; ++color) {
				for(IndexType cb=0; cb < num_cb; ++cb)  {
					for(IndexType block=0; block < num_blocks; ++block) {

						/* Work out the start and end sites taking into account that the last iteration may not be full */
						IndexType start_site = block*sites_per_block;
						IndexType end_site = (block+1)*sites_per_block > num_cbsites ? num_cbsites : (block+1)*sites_per_block;
						T* data = & (vector.Index(cb,start_site,spin,color,0));
						VScale(scalar, data, n_complex*(end_site-start_site));
					}
				}
			}
		}
	}





	/* Scale a spinor */
	template<typename T, size_t blocksize=MG_DEFAULT_ALIGNMENT >
	void
	MCaxpy(	GenericLayoutContainer<T,CBSOASpinorLayout<T>>& y,
			const std::complex<T>& scalar,
			const GenericLayoutContainer<T,CBSOASpinorLayout<T>>& x) {

		AssertCompatible( y.GetLatticeInfo(), x.GetLatticeInfo() );

		const IndexType num_spins = y.GetLatticeInfo().GetNumSpins();
		const IndexType num_colors = y.GetLatticeInfo().GetNumColors();
		const IndexType num_cb = n_checkerboard;
		const IndexType num_cbsites = y.GetLatticeInfo().GetNumCBSites();

		/* Problem: We can distribute threading and vectors in various ways.
		 * We can definitely compute in parallel the NormSq over each spin and color component in
		 * each checkerboard. However, in the case of the fine lattice this does not have a lot of
		 * parallelism. 4*3*2 = 24 way. However, on the fine lattice there may be a lot of local sites,
		 * e.g. 4^4. We want to vectorize over these, but also feed additional threads.
		 *
		 * In single precision, for a vector length of 16 we need 8 complexes.
		 * So	 the idea is to chunk the number of checkerboarded sites into blocks
		 * the l	ength of which corresponds to a cache line, which is at least 1 vector.
		 * We can then parallelize also over the chunks.
		 */

		const IndexType sites_per_block = blocksize/sizeof(std::complex<T>);

		// This many sites
		const IndexType num_blocks = (num_cbsites%sites_per_block == 0) ?
				(num_cbsites / sites_per_block) : (num_cbsites/sites_per_block)+1;

#pragma omp parallel for collapse(4)
		for(IndexType spin=0; spin < num_spins; ++spin) {
			for(IndexType color=0; color < num_colors; ++color) {
				for(IndexType cb=0; cb < num_cb; ++cb)  {
					for(IndexType block=0; block < num_blocks; ++block) {

						/* Work out the start and end sites taking into account that the last iteration may not be full */
						IndexType start_site = block*sites_per_block;
						IndexType end_site = (block+1)*sites_per_block > num_cbsites ? num_cbsites : (block+1)*sites_per_block;
						T* ydata =& (y.Index(cb,start_site,spin,color,0));
						const T* xdata = & (x.Index(cb,start_site,spin,color,0));
						MCaxpy(ydata,scalar, xdata, (end_site-start_site));
					}
				}
			}
		}
	}

	template<typename T>
	void GramSchmidt(std::vector<GenericLayoutContainer<T, CBSOASpinorLayout<T>>>& vectors)
	{
		auto num_vectors = vectors.size();

		// Dumb?
		if ( num_vectors == 0 ) {
			MGUtils::MasterLog(MGUtils::ERROR, "GramSchmidt: Attempting to Orthogonalize zero vectors ");
		}

		// Gram-Schmidt-ery
		double norm_sq = NormSq( vectors[0] );
		T invnorm = 1/sqrt(norm_sq);
		VScale(invnorm, vectors[0]);

		for(IndexType i=1; i < num_vectors; ++i) {
			for(IndexType j=0; j < i; ++j) {
				std::complex<double> iprod=InnerProduct(vectors[i],vectors[j]);
				std::complex<T> iprod_T = { iprod.real(), iprod.imag() };
				MCaxpy(vectors[i], iprod_T, vectors[j]);
			}
			//Now normalize
			norm_sq = NormSq( vectors[i] );
			invnorm = 1/sqrt(norm_sq);
			VScale(invnorm, vectors[i]);
		}
	}

}




#endif /* INCLUDE_LATTICE_CB_SOA_SPINOR_SPECIFIC_H_ */
