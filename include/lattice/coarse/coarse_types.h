/*
 * coarse_types.h
 *
 *  Created on: Jan 21, 2016
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_COARSE_COARSE_TYPES_H_
#define INCLUDE_LATTICE_COARSE_COARSE_TYPES_H_

#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "utils/memory.h"
#include "utils/print_utils.h"


using namespace MG;

namespace MG {

	/** Coarse Spinor
	 *  \param LatticeInfo
	 *
	 *  Basic Coarse Spinor. Holds memory for two checkerboards of sites
	 *  The checheckerboards are contiguous for now.
	 *  Regular site ordering: ie <cb><sites>< Nspin*Ncolor >< n_complex = fastest >
	 *
	 *
	 *  Destruction frees memory
	 *
	 */
	class CoarseSpinor {
	public:
		CoarseSpinor(const LatticeInfo& lattice_info) : _lattice_info(lattice_info), data{nullptr,nullptr},
				_n_color(lattice_info.GetNumColors()),
				_n_spin(lattice_info.GetNumSpins()),
				_n_colorspin(lattice_info.GetNumColors()*lattice_info.GetNumSpins()),
				_n_site_offset(n_complex*_n_colorspin)
		{
#if 1
			// Check That we have 2 spins
			if( lattice_info.GetNumSpins() != 2 ) {
				MasterLog(ERROR, "Attempting to Create CoarseSpinor with num_spins != 2");
			}
#endif

			// Allocate Data
			IndexType num_floats_per_cb = _lattice_info.GetNumCBSites()*_n_site_offset;

			/* Non-Contiguout allocation */
			data[0] = (float *)MG::MemoryAllocate(num_floats_per_cb*sizeof(float), MG::REGULAR);
			data[1] = (float *)MG::MemoryAllocate(num_floats_per_cb*sizeof(float), MG::REGULAR);

			/* Offset the checkerboard */
			//data[1] = (data[0] + num_floats_per_cb);
		}

		/** GetCBData
		 *
		 * 	Returns a pointer to the data for cb
		 */
		inline
		float* GetCBDataPtr(IndexType cb)
		{
			return data[cb];
		}

		/** GetSiteData
		 *
		 *  Returns a pointer to the data for a site in a cb
		 *  This is essentially a float array of size _n_site_offset
		 *  or it can be reinterpreted as _n_colorspin complexes
		 */
		inline
		float* GetSiteDataPtr(IndexType cb, IndexType site)
		{
			return &data[cb][site*_n_site_offset];
		}

		inline
		const float* GetSiteDataPtr(IndexType cb, IndexType site) const
			{
				return &data[cb][site*_n_site_offset];
			}

		~CoarseSpinor()
		{
			MemoryFree(data[0]);
			MemoryFree(data[1]);
			data[0] = nullptr;
			data[1] = nullptr;
		}

		inline
		IndexType GetNumColorSpin() const {
				return _n_colorspin;
		}

		inline
		IndexType GetNumColor() const {
				return _n_color;
		}

		inline
		IndexType GetNumSpin() const {
				return _n_spin;
		}

		inline
		const LatticeInfo& GetInfo() const {
			return _lattice_info;
		}


	private:
		const LatticeInfo& _lattice_info;
		float* data[2];  // Even and odd checkerboards

		const IndexType _n_color;
		const IndexType _n_spin;
		const IndexType _n_colorspin;
		const IndexType _n_site_offset;

	};


	class CoarseGauge {
	public:
		CoarseGauge(const LatticeInfo& lattice_info) : _lattice_info(lattice_info), data{nullptr,nullptr},
				_n_color(lattice_info.GetNumColors()),
				_n_spin(lattice_info.GetNumSpins()),
				_n_colorspin(lattice_info.GetNumColors()*lattice_info.GetNumSpins()),
				_n_link_offset(n_complex*_n_colorspin*_n_colorspin),
				_n_site_offset((2*n_dim+1)*_n_link_offset)

		{
			// Check That we have 2 spins
			if( lattice_info.GetNumSpins() != 2 ) {
				MasterLog(ERROR, "Attempting to Create CoarseSpinor with num_spins != 2");
			}


			// Allocate Data
			IndexType num_floats_per_cb = _lattice_info.GetNumCBSites()*_n_site_offset;

			/* Contiguout allocation */
			data[0] = (float *)MG::MemoryAllocate(num_floats_per_cb*sizeof(float), MG::REGULAR);
			data[1] = (float *)MG::MemoryAllocate(num_floats_per_cb*sizeof(float), MG::REGULAR);

		}

		/** GetCBData
		 *
		 * 	Returns a pointer to the data for cb
		 */
		inline
		float* GetCBDataPtr(IndexType cb)
		{
			return data[cb];
		}

		/** GetSiteData
		 *
		 *  Returns a pointer to the data for a site in a cb
		 *  This is essentially a float array of size _n_site_offset
		 *  or it can be reinterpreted as _n_colorspin complexes
		 */
		inline
		float* GetSiteDataPtr(IndexType cb, IndexType site)
		{
			return &data[cb][site*_n_site_offset];
		}

		/** GetSiteData
				 *
				 *  Returns a pointer to the data for a site in a cb
				 *  This is essentially a float array of size _n_site_offset
				 *  or it can be reinterpreted as _n_colorspin complexes
				 */
				inline
				const float* GetSiteDataPtr(IndexType cb, IndexType site) const
				{
					return &data[cb][site*_n_site_offset];
				}
				/** GetSiteDirData
				 *
				 *  Returns a pointer to the link in direction mu
				 *  Conventions are:
				 *  	mu=0 - X forward
				 *  	mu=1 - X backward
				 *  	mu=2 - Y forwad
				 *  	mu=3 - Y backward
				 *  	mu=4 - Z forward
				 *  	mu=5 - Z backward
				 *      mu=6 - T forward
				 *      mu=7 - T backward
				 *      mu=8 - Local Term.
				 */
				inline
				float *GetSiteDirDataPtr(IndexType cb, IndexType site, IndexType mu)
				{
					return &data[cb][site*_n_site_offset + mu*_n_link_offset];
				}

				inline
				const float *GetSiteDirDataPtr(IndexType cb, IndexType site, IndexType mu) const
				{
					return &data[cb][site*_n_site_offset + mu*_n_link_offset];
				}

		~CoarseGauge()
		{
			MemoryFree(data[0]);
			MemoryFree(data[1]);
			data[0] = nullptr;
			data[1] = nullptr;
		}

		inline
		IndexType GetNumColorSpin() const {
				return _n_colorspin;
		}

		inline
		IndexType GetNumColor() const {
				return _n_color;
		}

		inline
		IndexType GetNumSpin() const {
				return _n_spin;
		}

		inline
		IndexType GetLinkOffset() const {
			return _n_link_offset;
		}

		inline
		const LatticeInfo& GetInfo() const {
			 return _lattice_info;
		}
	private:
		const LatticeInfo& _lattice_info;
		float* data[2];  // Even and odd checkerboards

		const IndexType _n_color;
		const IndexType _n_spin;
		const IndexType _n_colorspin;
		const IndexType _n_link_offset;
		const IndexType _n_site_offset;
	};



}



#endif /* INCLUDE_LATTICE_COARSE_COARSE_TYPES_H_ */
