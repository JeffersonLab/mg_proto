/*
 * coarse_transfer.h
 *
 *  Created on: Mar 16, 2018
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_COARSE_COARSE_TRANSFER_H_
#define INCLUDE_LATTICE_COARSE_COARSE_TRANSFER_H_

#include <MG_config.h>

#if defined MG_USE_AVX512
#include <cstdio>
#include <immintrin.h>
#endif

#include "MG_config.h"
#include <lattice/qphix/qphix_veclen.h>
#include <lattice/coarse/coarse_types.h>
#include <lattice/cmat_mult.h>
#include "lattice/qphix/qphix_types.h"
#include <lattice/coarse/block.h>
#include <utils/print_utils.h>

#include <vector>
#include <memory>

namespace MG {

	class CoarseTransfer {
		public:

			CoarseTransfer(const std::vector<Block>& blocklist,
					const std::vector<std::shared_ptr<CoarseSpinor> >& vectors, int r_threads_per_block = 1)
				: _n_blocks(blocklist.size()),  _n_vecs(vectors.size()), _blocklist(blocklist), _vecs(vectors),
				_r_threads_per_block(r_threads_per_block) {


					if ( blocklist.size() > 0 ) {
						_sites_per_block = blocklist[0].getNumSites();
					}
					else {
						MasterLog(ERROR, "Cannot create Restrictor Array for blocklist of size 0");
					}


					if( _vecs.size() == 0 ) {
						MasterLog(ERROR, "Attempting to create transfer operator without any vectors.");
					}

					const LatticeInfo& fine_info = vectors[0]->GetInfo();

					num_fine_color = fine_info.GetNumColors();
					int num_coarse_color = _n_vecs;

					int num_coarse_colorspin = 2*num_coarse_color;
					int num_fine_colorspin = 2* num_fine_color;

					int num_complex = _n_blocks*_sites_per_block*num_fine_colorspin*num_coarse_colorspin*n_complex;
					_data = (float *)MemoryAllocate(num_complex*sizeof(float));

					int num_coarse_cbsites = _n_blocks/2;


					MasterLog(INFO, "Importing Vectors");
#pragma omp parallel for collapse(2)
					for(int block_cb = 0; block_cb < n_checkerboard; ++block_cb ) {
						for(int block_cbsite = 0 ; block_cbsite < num_coarse_cbsites; ++block_cbsite) {

							// Identify the current block.
							int block_idx = block_cbsite + block_cb*num_coarse_cbsites;
							const Block& block = blocklist[block_idx];

							// Get the list of fine sites in the blocks
							auto block_sitelist = block.getCBSiteList();
							auto num_sites_in_block = block_sitelist.size();
							const int offset = n_complex*num_fine_color;

							// Loop through the fine sites in the block
							for( IndexType fine_site_idx = 0; fine_site_idx < static_cast<IndexType>(num_sites_in_block); fine_site_idx++ ) {

								// Find the fine site
								const CBSite& fine_cbsite = block_sitelist[fine_site_idx];


								// Copy components  into  (n_num_fine_color x num_coarse_colorspin )
								for(int v=0; v < _n_vecs; ++v) {

									const float* vsite = vectors[v]->GetSiteDataPtr(0,fine_cbsite.cb, fine_cbsite.site);



									for(int chiral =0 ; chiral < 2 ; ++chiral ) {
										for(int color=0; color < num_fine_color; ++color) {

											(*this).index(block_idx,fine_site_idx,color,chiral,v,RE)
												= vsite[ chiral*offset +color * n_complex + RE];

											(*this).index(block_idx,fine_site_idx,color,chiral,v,IM)
												= -vsite[ chiral*offset + color * n_complex + IM ];
										}
									}
								}

							}
						}
					}

					// Set up the reverse map
					MasterLog(INFO, "Creating Reverse Map");

					for(int cb=0; cb < n_checkerboard; ++cb) {
						reverse_map[cb].resize(fine_info.GetNumCBSites());
						reverse_transfer_row[cb].resize(fine_info.GetNumCBSites());
					}

#pragma omp parallel for collapse(2)
					for(int block_cb = 0; block_cb < n_checkerboard; ++block_cb ) {
						for(int block_cbsite = 0 ; block_cbsite < num_coarse_cbsites; ++block_cbsite) {

							// Identify the current block.
							int block_idx = block_cbsite + block_cb*num_coarse_cbsites;
							const Block& block = blocklist[block_idx];

							// Get the list of fine sites in the blocks
							auto block_sitelist = block.getCBSiteList();
							auto num_sites_in_block = block_sitelist.size();

							// Loop through the fine sites in the block
							for( IndexType fine_site_idx = 0; fine_site_idx < static_cast<IndexType>(num_sites_in_block); fine_site_idx++ ) {
								const CBSite& cbsite = block_sitelist[fine_site_idx];
								reverse_map[cbsite.cb][cbsite.site].cb = block_cb;
								reverse_map[cbsite.cb][cbsite.site].site = block_cbsite;
								reverse_transfer_row[cbsite.cb][cbsite.site] = fine_site_idx;

							}
						}
					}

#pragma omp parallel
					{
						int tid = omp_get_thread_num();
						int n_threads = omp_get_num_threads();
						if ( tid == 0 ) {
							_n_threads = n_threads;
						}
#pragma omp barrier
					}
				}

			inline
				float& index(int block, int blocksite,  int color, int chiral, int vec, int REIM)
				{
					return _data[ REIM + n_complex*(vec
							+ _n_vecs*(chiral + 2*(color +num_fine_color*(blocksite + _sites_per_block*block)))) ];
				}

			inline
				const
				float& index(int block, int blocksite, int color, int chiral, int vec, int REIM) const
				{
					return _data[ REIM + n_complex*(vec
							+ _n_vecs*(chiral + 2*(color +num_fine_color*(blocksite + _sites_per_block*block)))) ];
				}


			inline
				const
				float* indexPtr(int block, int blocksite,  int color) const
				{
					return &(_data[  2*n_complex*_n_vecs*(color +num_fine_color*(blocksite + _sites_per_block*block)) ]);
				}

			void R_op(int num_coarse_color, const CoarseSpinor& fine_in, int source_cb, CoarseSpinor& out) const
			{
				assert(num_coarse_color == out.GetNumColor());
				assert(fine_in.GetNCol() == out.GetNCol());
				IndexType ncol = fine_in.GetNCol();

				const int num_coarse_cbsites = out.GetInfo().GetNumCBSites();

				const int num_coarse_colorspin = 2*num_coarse_color;

				// Sanity check. The number of sites in the coarse spinor
				// Has to equal the number of blocks
				//  assert( n_checkerboard*num_coarse_cbsites == static_cast<const int>(blocklist.size()) );

				// The number of vectors has to eaqual the number of coarse colors
				assert( _n_vecs == num_coarse_color );
				assert( num_coarse_cbsites == _n_blocks/2);

				ZeroVec(out);
				// This will be a loop over blocks
#pragma omp parallel for collapse(2) schedule(static)
				for(int block_cb = 0; block_cb < n_checkerboard; ++block_cb ) {
					for(int block_cbsite = 0 ; block_cbsite < num_coarse_cbsites; ++block_cbsite) {
						// Identify the current block.
						int block_idx = block_cbsite + block_cb*num_coarse_cbsites;
						const Block& block = _blocklist[block_idx];

						// Get the list of fine sites in the blocks
						auto block_sitelist = block.getCBSiteList();
						auto num_sites_in_block = block_sitelist.size();

						// The coarse site spinor is where we will write the result
						std::complex<float>* coarse_site_spinor = reinterpret_cast<std::complex<float>*>(out.GetSiteDataPtr(0,block_cb,block_cbsite));

						// Loop through the fine sites in the block
						for( IndexType fine_site_idx = 0; fine_site_idx < static_cast<IndexType>(num_sites_in_block); fine_site_idx++ ) {

							// Find the fine site
							const CBSite& fine_cbsite = block_sitelist[fine_site_idx];

							if( source_cb < 0 || fine_cbsite.cb == source_cb ) {
								// Get the pointer to the fine site data. Should be n_complex*num_fine_color*2 floats or 2*num_fine_color complexes
								// If num_fine_color is a multiple of 8 this should be properly 64 byte aligned.

								const std::complex<float>* fine_cbsite_data =  reinterpret_cast<const std::complex<float>*>(fine_in.GetSiteDataPtr(0,fine_cbsite.cb,
											fine_cbsite.site));
								const std::complex<float>* v = reinterpret_cast<const std::complex<float>*>((*this).indexPtr(block_idx, fine_site_idx,0));

								// Do the upper chiral component
								CMatMultCoeffAddNaive(1.0, coarse_site_spinor, num_coarse_color*2, 1.0, v, num_coarse_color*2, fine_cbsite_data, num_fine_color*2,
								                      num_coarse_color, num_fine_color, ncol);

								// Do the lower chiral component
								CMatMultCoeffAddNaive(1.0, &coarse_site_spinor[num_coarse_color], num_coarse_color*2, 1.0, &v[num_coarse_color], num_coarse_color*2, &fine_cbsite_data[num_fine_color], num_fine_color*2,
								                      num_coarse_color, num_fine_color, ncol);

							}
						} // fine sites in block

					}// block CBSITE
				} // block CB
			}

			void R(const CoarseSpinor& fine_in, CoarseSpinor& out) const
			{
				R_op(out.GetNumColor(),fine_in,-1,out);
			}

			void R(const CoarseSpinor& fine_in, int source_cb, CoarseSpinor& out) const
			{
				R_op(out.GetNumColor(),fine_in,source_cb,out);
			}

			void P_op(int num_coarse_color, const CoarseSpinor& coarse_in, int target_cb, CoarseSpinor& fine_out) const
			{

				const LatticeInfo& fine_info = fine_out.GetInfo();
				const LatticeInfo& coarse_info = coarse_in.GetInfo();
				assert(coarse_in.GetNCol() == fine_out.GetNCol());
				IndexType ncol = coarse_in.GetNCol();

				assert( num_coarse_color == coarse_info.GetNumColors());
				assert( num_fine_color == fine_info.GetNumColors());

				const int num_fine_cbsites = fine_info.GetNumCBSites();
				ZeroVec(fine_out);
				int ncb = target_cb >= 0 ? 1 : n_checkerboard;
#pragma omp parallel for collapse(2) schedule(static)
				for(int cbi =0; cbi < ncb; ++cbi) {
					for(int fsite=0; fsite < num_fine_cbsites; ++fsite) {

						int cb=target_cb >= 0 ? target_cb : cbi;
						int block_cb = reverse_map[cb][fsite].cb;
						int block_cbsite = reverse_map[cb][fsite].site;


						// These two to index the V-s
						int block_idx = block_cbsite + block_cb * coarse_info.GetNumCBSites();
						int fine_site_idx = reverse_transfer_row[cb][fsite];

						std::complex<float>* fine_site_spinor = reinterpret_cast<std::complex<float>*>(fine_out.GetSiteDataPtr(0,cb,fsite));
						const std::complex<float>* coarse_site_spinor =
							reinterpret_cast<const std::complex<float>*>(coarse_in.GetSiteDataPtr(0,block_cb,block_cbsite));
						const std::complex<float>* v =
							reinterpret_cast<const std::complex<float>*>((*this).indexPtr(block_idx, fine_site_idx,0));

						// Do the upper chiral component
						CMatAdjMultCoeffAddNaive(1.0, fine_site_spinor, num_fine_color*2, 1.0, v, num_coarse_color*2, coarse_site_spinor, num_coarse_color*2,
								num_coarse_color, num_fine_color, ncol);

						// Do the lower chiral component
						CMatAdjMultCoeffAddNaive(1.0, &fine_site_spinor[num_fine_color], num_fine_color*2, 1.0, &v[num_coarse_color], num_coarse_color*2, &coarse_site_spinor[num_coarse_color], num_coarse_color*2,
								num_coarse_color, num_fine_color, ncol);


					} // fsite
				} // cbi
			} // function


			void P(const CoarseSpinor& coarse_in, CoarseSpinor& fine_out) const
			{
				P_op(coarse_in.GetNumColor(),coarse_in,-1,fine_out);
			}

			void P(const CoarseSpinor& coarse_in, int target_cb, CoarseSpinor& fine_out) const
			{
				P_op(coarse_in.GetNumColor(),coarse_in,target_cb,fine_out);
			}

			~CoarseTransfer()
			{
				MemoryFree(_data);
			}

		private:


			int _n_blocks;
			int _sites_per_block;
			int _n_vecs;
			int num_fine_color;
			const std::vector<Block>& _blocklist;
			std::vector<CBSite> reverse_map[2];
			std::vector<int> reverse_transfer_row[2];
			float* _data;
			const std::vector<std::shared_ptr<CoarseSpinor> >& _vecs;
			int _n_threads;
			int _r_threads_per_block;
	};




} // namespace




#endif /* INCLUDE_LATTICE_COARSE_COARSE_TRANSFER_H_ */
