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
#include "lattice/qphix/qphix_types.h"
#include <lattice/coarse/block.h>
#include <utils/print_utils.h>

#include <vector>
#include <memory>

namespace MG {

#ifndef MAX_VECS
#define MAX_VECS 64
#endif



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

						const float* vsite = vectors[v]->GetSiteDataPtr(fine_cbsite.cb, fine_cbsite.site);



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

#ifndef MG_USE_AVX512
  template< int num_coarse_color>
	void R_op(const CoarseSpinor& fine_in, CoarseSpinor& out) const
	{
	  assert(num_coarse_color == out.GetNumColor());

	  const int num_coarse_cbsites = out.GetInfo().GetNumCBSites();

	  const int num_coarse_colorspin = 2*num_coarse_color;

	  // Sanity check. The number of sites in the coarse spinor
	  // Has to equal the number of blocks
	  //  assert( n_checkerboard*num_coarse_cbsites == static_cast<const int>(blocklist.size()) );

	  // The number of vectors has to eaqual the number of coarse colors
	  assert( _n_vecs == num_coarse_color );
	  assert( num_coarse_cbsites == _n_blocks/2);

	  // This will be a loop over blocks
#pragma omp parallel for collapse(2)
	  for(int block_cb = 0; block_cb < n_checkerboard; ++block_cb ) {
	    for(int block_cbsite = 0 ; block_cbsite < num_coarse_cbsites; ++block_cbsite) {

	      // Identify the current block.
	      int block_idx = block_cbsite + block_cb*num_coarse_cbsites;
	      const Block& block = _blocklist[block_idx];

	      // Get the list of fine sites in the blocks
	      auto block_sitelist = block.getCBSiteList();
	      auto num_sites_in_block = block_sitelist.size();

	      // The coarse site spinor is where we will write the result
	      std::complex<float>* coarse_site_spinor = reinterpret_cast<std::complex<float>*>(out.GetSiteDataPtr(block_cb,block_cbsite));

	      // The accumulation goes here
	      std::complex<float> site_accum[ 2*MAX_VECS ] __attribute__((aligned(64)));

	      const int offset = num_coarse_color;
	      // Zero the accumulated coarse site

#pragma omp simd safelen(16) simdlen(16) aligned(site_accum: 64)
	      for(int i=0; i < 2*num_coarse_color; ++i) {
	    	  site_accum[i] = 0;
	      }

	      // Loop through the fine sites in the block
	      for( IndexType fine_site_idx = 0; fine_site_idx < static_cast<IndexType>(num_sites_in_block); fine_site_idx++ ) {

	    	  // Find the fine site
	    	  const CBSite& fine_cbsite = block_sitelist[fine_site_idx];

	    	  // Get the pointer to the fine site data. Should be n_complex*num_fine_color*2 floats or 2*num_fine_color complexes
	    	  // If num_fine_color is a multiple of 8 this should be properly 64 byte aligned.

	    	  const std::complex<float>* fine_cbsite_data =  reinterpret_cast<const std::complex<float>*>(fine_in.GetSiteDataPtr(fine_cbsite.cb,
	    			  fine_cbsite.site));

	    	  // for each site we will loop over Ns/2 * Ncolor

	    	  for(int f_color=0; f_color < num_fine_color; ++f_color) {


	    		  std::complex<float>  psi_upper(fine_cbsite_data[f_color]);
	    		  std::complex<float>  psi_lower(fine_cbsite_data[f_color + num_fine_color]);

	    		  const std::complex<float>* v = reinterpret_cast<const std::complex<float>*>((*this).indexPtr(block_idx, fine_site_idx,f_color));


	    			  // Accumulate the upper chiral compponent
#pragma simd safelen(8) simdlen(16) aligned(site_accum, v:64)
	    			  for(int c_color=0; c_color < num_coarse_color; c_color++) {
	    				  site_accum[c_color] += v[c_color]*psi_upper;
	    			  }

	    			  // Accumulate the lower chiral  componente
#pragma simd safelen(8) simdlen(16) aligned(site_accum, v:64)
	    			  for(int c_color=0; c_color < num_coarse_color; c_color++) {
	    				  site_accum[c_color+offset] += v[c_color+offset]*psi_lower;
	    			  }

	    		  } // f_color

	      } // fine sites in block

	      // Stream out
#pragma simd safelen(16) simdlen(16) aligned(coarse_site_spinor, site_accum:64)
	      for(int colorspin=0; colorspin <  2*num_coarse_color; ++colorspin) {
	    	  coarse_site_spinor[colorspin] = site_accum[colorspin];
	      }
	    }// block CBSITE
	  } // block CB
	}
#else
  template<int num_coarse_color>
  void R_op(const CoarseSpinor& fine_in, CoarseSpinor& out) const
  {

	  assert(num_coarse_color == out.GetNumColor());

	  const int num_coarse_cbsites = out.GetInfo().GetNumCBSites();

	  constexpr  int num_coarse_colorspin = 2*num_coarse_color;
	  constexpr  int n_floats = 4*num_coarse_color;

	  // Sanity check. The number of sites in the coarse spinor
	  // Has to equal the number of blocks
	  //  assert( n_checkerboard*num_coarse_cbsites == static_cast<const int>(blocklist.size()) );

	  // The number of vectors has to eaqual the number of coarse colors
	  assert( _n_vecs == num_coarse_color );
	  assert( num_coarse_cbsites == _n_blocks/2);


	  // Threasd can accumulate in here
	  float site_accum[ n_floats*_n_threads] __attribute__((aligned(64)));

    //block lock
    int mutex[_n_threads];

#pragma omp parallel shared(site_accum,mutex)
	  {
		  int tid = omp_get_thread_num();
		  int r_block_threads = _n_threads/ _r_threads_per_block;
		  int block_tid = tid / _r_threads_per_block;
		  int site_tid =  tid % _r_threads_per_block;

      //set lock
      mutex[site_tid +  _r_threads_per_block*block_tid] = 0;
      
		  // Zero this buffer - so that if there are too many threads
		  // in the block, their contribtion will give zero
#pragma omp simd simdlen(16) safelen(16) aligned(site_accum:64)
		  for(int i=0; i < n_floats; ++i) {
			  site_accum[i+n_floats*(site_tid + _r_threads_per_block*block_tid)]= 0;
		  }

      if ( block_tid < _n_blocks ) {

        for(int  block_idx = block_tid; block_idx < _n_blocks; block_idx += r_block_threads) {

          int block_cb = block_idx /num_coarse_cbsites;
          int block_cbsite = block_idx % num_coarse_cbsites;
          const Block& block = _blocklist[block_idx];

          // Get the list of fine sites in the blocks
          auto block_sitelist = block.getCBSiteList();
          auto num_sites_in_block = block_sitelist.size();

          // The coarse site spinor is where we will write the result
          float* coarse_site_spinor = out.GetSiteDataPtr(block_cb,block_cbsite);

          // Zero the result
          if( site_tid  == 0 ) {
#pragma omp simd safelen(16) simdlen(16) aligned(coarse_site_spinor:64)
            for(int i=0; i < n_floats; ++i) {
              coarse_site_spinor[i] = 0;
            }
          } // no need to barrier here as only site_tid == 0 will write this again

          int sa_offset = n_floats*(site_tid  +_r_threads_per_block*block_tid);

          // A thread may reuse this -- so re-zero it
#pragma omp simd safelen(16) simdlen(16) aligned(site_accum:64)
          for(int i=0; i < n_floats; ++i) {
            site_accum[i+sa_offset] = 0;
          }


          const int coffset = 2*num_coarse_color;
          const int foffset = 2*num_fine_color;


          for( IndexType fine_site_idx = site_tid;
          fine_site_idx < static_cast<IndexType>(num_sites_in_block);
          fine_site_idx += _r_threads_per_block ) {
            
            const CBSite& fine_cbsite = block_sitelist[fine_site_idx];
            const float *fine_data  = fine_in.GetSiteDataPtr(fine_cbsite.cb, fine_cbsite.site);

            for(int color=0; color < num_fine_color; ++color) {


              __m512 psi_upper_re = _mm512_set1_ps(  fine_data[ RE+ 2*color]  );
              __m512 psi_upper_im = _mm512_set1_ps(  fine_data[ IM + 2*color]  );

              __m512 psi_lower_re = _mm512_set1_ps( fine_data[RE + 2*color +foffset ] );
              __m512 psi_lower_im = _mm512_set1_ps( fine_data[IM + 2*color +foffset ] );

              const float* v = ((*this).indexPtr(block_idx, fine_site_idx,color));


              for(int i=0; i < 2*num_coarse_color; i +=16) {
                __m512 v_vec = _mm512_load_ps( &v[i] );
                __m512 accum_vec = _mm512_load_ps( &site_accum[i+sa_offset]);

                __m512 v_perm= _mm512_shuffle_ps(v_vec,v_vec, 0xb1);
                __m512 t = _mm512_fmaddsub_ps(  v_perm, psi_upper_im, accum_vec);
                accum_vec = _mm512_fmaddsub_ps( v_vec, psi_upper_re, t );
                _mm512_store_ps( &site_accum[i+sa_offset], accum_vec);
              }

              int soffset = coffset + sa_offset;

              for(int i=0; i <2*num_coarse_color; i+=16) {
                __m512 v_vec = _mm512_load_ps( &v[i+coffset] );
                __m512 accum_vec = _mm512_load_ps( &site_accum[i+soffset]);

                __m512 v_perm= _mm512_shuffle_ps(v_vec,v_vec, 0xb1);
                __m512 t = _mm512_fmaddsub_ps(  v_perm, psi_lower_im, accum_vec);
                accum_vec = _mm512_fmaddsub_ps( v_vec, psi_lower_re, t );
                _mm512_store_ps( &site_accum[i+soffset], accum_vec);
              }
            }// color
          } // fine sites in block
          mutex[site_tid +  _r_threads_per_block*block_tid] = 1;

          //#pragma omp barrier
          if( site_tid ==0 ) {
            int count = _r_threads_per_block;
            while(count>0){
              std::cout << site_tid << "," << block_tid << " " << count << std::endl;
              for(int s=0; s < _r_threads_per_block; ++s) {
                if(mutex[s +  _r_threads_per_block*block_tid] == 1){
                  mutex[s +  _r_threads_per_block*block_tid] = 0;
                  count--;
                  int soffset =n_floats*(s  + _r_threads_per_block*block_tid);
#pragma simd safelen(16) simdlen(16) aligned(coarse_site_spinor, site_accum:64)
                  for(int colorspin=0; colorspin <  n_floats; ++colorspin) {
                    coarse_site_spinor[colorspin] += site_accum[colorspin+soffset];
                  }
                }
              }
            }
          }
          //						  int soffset =n_floats*(s  + _r_threads_per_block*block_tid);
          //
          //
          //						  }
          //					  } //s
          //				  } // site_tid

        } // block idx
      }
      //		  else {
      //#pragma omp barrier
      //		  }
    } // parallel

  }
#endif

  void R(const CoarseSpinor& fine_in, CoarseSpinor& out) const
  {
    const  int num_color = out.GetNumColor();
    if( num_color == 8 ) {
      R_op<8>(fine_in,out);
    }
    else if ( num_color == 16 ) {
      R_op<16>(fine_in,out);
    }
    else if ( num_color == 24 ) {
      R_op<24>(fine_in,out);
    }
    else if ( num_color == 32 ) {
      R_op<32>(fine_in,out);
    }
    else if ( num_color == 40 ) {
       R_op<40>(fine_in,out);
     }
    else if ( num_color == 48 ) {
       R_op<48>(fine_in,out);
     }
    else if ( num_color == 56 ) {
       R_op<56>(fine_in,out);
     }
    else if ( num_color == 64 ) {
       R_op<64>(fine_in,out);
     }
    else {
      MasterLog(ERROR, "Unhandled dispatch size %d. Num coarse color must be divisible by 8 and <=64", num_color);
    }
    return;
  }

#ifndef MG_USE_AVX512

  template<int num_coarse_color>
  void P_op(const CoarseSpinor& coarse_in, CoarseSpinor& fine_out) const
  {

    const LatticeInfo& fine_info = fine_out.GetInfo();
    const LatticeInfo& coarse_info = coarse_in.GetInfo();

    assert( num_coarse_color == coarse_info.GetNumColors());
    assert( num_fine_color == fine_info.GetNumColors());

    const int num_fine_cbsites = fine_info.GetNumCBSites();

#pragma omp parallel for collapse(2)
    for(int cb =0; cb < n_checkerboard; ++cb) {
    	for(int fsite=0; fsite < num_fine_cbsites; ++fsite) {



    		int block_cb = reverse_map[cb][fsite].cb;
    		int block_cbsite = reverse_map[cb][fsite].site;


    		// These two to index the V-s
    		int block_idx = block_cbsite + block_cb * coarse_info.GetNumCBSites();
    		int fine_site_idx = reverse_transfer_row[cb][fsite];

    		std::complex<float>* fine_site_data = reinterpret_cast<std::complex<float>*>(fine_out.GetSiteDataPtr(cb,fsite));
    		const std::complex<float>* coarse_site_spinor =
    				reinterpret_cast<const std::complex<float>*>(coarse_in.GetSiteDataPtr(block_cb,block_cbsite));

    		for(int fcolor=0; fcolor < num_fine_color; ++fcolor ) {

    			// v is 2 x num_coarse_color complexes.
    			const std::complex<float>* v =
    					reinterpret_cast<const std::complex<float>*>((*this).indexPtr(block_idx, fine_site_idx,fcolor));

    			std::complex<float> reduce_upper(0,0);
    			std::complex<float> reduce_lower(0,0);


#pragma omp simd safelen(8) simdlen(16) aligned(v,coarse_site_spinor:64)
    			for(int i=0; i < num_coarse_color; ++i) {
    				reduce_upper +=conj( v[i]) * coarse_site_spinor[i];
    			}
#pragma omp simd safelen(8) simdlen(16) aligned(v,coarse_site_spinor:64)
    			for(int i=0; i < num_coarse_color; ++i) {
    				reduce_lower += conj(v[i+num_coarse_color]) * coarse_site_spinor[i+num_coarse_color];
    			}

    			fine_site_data[fcolor] = reduce_upper;
    			fine_site_data[fcolor+num_fine_color] = reduce_lower;

    		} // fcolor
    	} // fsite

    }  // cb
  } // function


#else

  template<int num_coarse_color>
    void P_op(const CoarseSpinor& coarse_in, CoarseSpinor& fine_out) const
    {

      const LatticeInfo& fine_info = fine_out.GetInfo();
      const LatticeInfo& coarse_info = coarse_in.GetInfo();

      assert( num_coarse_color == coarse_info.GetNumColors());
      assert( num_fine_color == fine_info.GetNumColors());

      const int num_fine_cbsites = fine_info.GetNumCBSites();
      const int coffset = 2*num_coarse_color;
      const int foffset = 2*num_fine_color;

  #pragma omp parallel for collapse(2)
      for(int cb =0; cb < n_checkerboard; ++cb) {
      	for(int fsite=0; fsite < num_fine_cbsites; ++fsite) {



      		int block_cb = reverse_map[cb][fsite].cb;
      		int block_cbsite = reverse_map[cb][fsite].site;

      		int block_idx = block_cbsite + block_cb * coarse_info.GetNumCBSites();
      		int fine_site_idx = reverse_transfer_row[cb][fsite];

     		float* fine_site_data =(float *)fine_out.GetSiteDataPtr(cb,fsite);
      		const float* coarse_site_spinor = coarse_in.GetSiteDataPtr(block_cb,block_cbsite);
      		float fine_site_tmp[2*MAX_VECS] __attribute__((aligned(64)));


      		for(int fcolor=0; fcolor < num_fine_color; ++fcolor ) {

      			// v is 2 x num_coarse_color complexes.
      			const float* v =
      					reinterpret_cast<const float*>((*this).indexPtr(block_idx, fine_site_idx,fcolor));

      			float reduce_upper_re=0;
      			float reduce_upper_im=0;

      			float reduce_lower_re=0;
      			float reduce_lower_im=0;


      			float vec_re[16] __attribute__((aligned(64)));
      			float vec_im[16] __attribute__((aligned(64)));

      			const __m512 sign = _mm512_set_ps(-1,1,-1,1,-1,1,-1,1, -1,1,-1,1,-1,1,-1,1);
      			__m512 sum_r_vec __attribute((aligned(64)));
      			__m512 sum_i_vec __attribute((aligned(64)));


      			sum_r_vec = _mm512_setzero_ps();
      			sum_i_vec = _mm512_setzero_ps();



      			for(int i=0; i < 2*num_coarse_color; i+=16) {

      				__m512 v_vec = _mm512_load_ps( &(v[i]) );
      				__m512 c_vec = _mm512_load_ps( &(coarse_site_spinor[i]));

      				sum_r_vec = _mm512_fmadd_ps(v_vec, c_vec, sum_r_vec);
      				__m512 vtmp = _mm512_shuffle_ps(c_vec, c_vec,0xb1);
      				sum_i_vec = _mm512_fmadd_ps( v_vec,vtmp,sum_i_vec);
      			}

      			sum_i_vec = _mm512_mul_ps(sum_i_vec,sign);
      			_mm512_store_ps(vec_re,sum_r_vec);
      			_mm512_store_ps(vec_im,sum_i_vec);

#pragma omp simd safelen(16) simdlen(16) aligned(vec_re:64)
      			for(int i=0; i < 16; ++i) {
      				reduce_upper_re +=vec_re[i];
      			}

#pragma omp simd safelen(16) simdlen(16) aligned(vec_im:64)
      			for(int i=0; i < 16; ++i) {
      				reduce_upper_im +=vec_im[i];
      			}

      			fine_site_tmp[RE + 2*fcolor] = reduce_upper_re;
      			fine_site_tmp[IM  + 2*fcolor] = reduce_upper_im;

      			sum_r_vec = _mm512_setzero_ps();
      			sum_i_vec = _mm512_setzero_ps();

      			for(int i=0; i < 2*num_coarse_color; i+=16) {
      				__m512 v_vec = _mm512_load_ps( &v[i + coffset] );
      				__m512 c_vec = _mm512_load_ps( &coarse_site_spinor[i+coffset]);

      				sum_r_vec = _mm512_fmadd_ps(v_vec, c_vec, sum_r_vec);
      				sum_i_vec = _mm512_fmadd_ps( v_vec, _mm512_shuffle_ps(c_vec,c_vec,0xb1),sum_i_vec);
      			}
      			sum_i_vec = _mm512_mul_ps(sum_i_vec,sign);
      			_mm512_store_ps(vec_re,sum_r_vec);
      			_mm512_store_ps(vec_im,sum_i_vec);

#pragma omp simd safelen(16) simdlen(16) aligned(vec_re:64)
      			for(int i=0; i < 16; ++i) {
      				reduce_lower_re +=vec_re[i];
      			}

#pragma omp simd safelen(16) simdlen(16) aligned(vec_im:64)
      			for(int i=0; i < 16; ++i) {
      				reduce_lower_im +=vec_im[i];
      			}

      			fine_site_tmp[ RE +  2*fcolor + foffset]  = reduce_lower_re;
      			fine_site_tmp[ IM  +  2*fcolor + foffset ] = reduce_lower_im;
      		} // fcolor

#pragma simd safelen(16) simdlen(16) aligned(fine_site_tmp,fine_site_data:64)
      		for(int i=0; i < 4*num_fine_color; i++) {
      			fine_site_data[i] = fine_site_tmp[i];
      		}

      	} // fsite
      } // cb
    } // funciton
  #endif



  void P(const CoarseSpinor& coarse_in, CoarseSpinor& fine_out) const
  {
	  int num_color = coarse_in.GetNumColor();

	  if( num_color == 8 ) {
		  P_op<8>(coarse_in,fine_out);
	  }
	  else  if( num_color == 16 ) {
		  P_op<16>(coarse_in,fine_out);
	  }
	  else  if( num_color == 24 ) {
		  P_op<24>(coarse_in,fine_out);
	  }
	  else if ( num_color == 32 ) {
		  P_op<32>(coarse_in,fine_out);
	  }
	  else if ( num_color == 40 ) {
		  P_op<40>(coarse_in,fine_out);
	  }
	  else if ( num_color == 48 ) {
		  P_op<48>(coarse_in,fine_out);
	  }
	  else if ( num_color == 56 ) {
		  P_op<56>(coarse_in,fine_out);
	  }
	  else if ( num_color == 64 ) {
		  P_op<64>(coarse_in,fine_out);
	  }
	  else {
		  MasterLog(ERROR, "Unhandled dispatch size %d. Num vecs must be divisible by 8 and <= 64", num_color);
	  }
	  return;
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
