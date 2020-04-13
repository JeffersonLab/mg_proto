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
	 *  Regular site ordering: ie <cb><sites>< Nspin*Ncolor >< n_complex = fastest >
	 *
	 *
	 *  Destruction frees memory
	 *
	 */
	class CoarseSpinor {
	public:
		CoarseSpinor(const LatticeInfo& lattice_info, IndexType n_col=1) : _lattice_info(lattice_info), data{nullptr,nullptr},
				_n_color(lattice_info.GetNumColors()),
				_n_spin(lattice_info.GetNumSpins()),
				_n_colorspin(lattice_info.GetNumColors()*lattice_info.GetNumSpins()),
				_n_site_offset(n_complex*_n_colorspin*n_col),
				_n_xh( lattice_info.GetCBLatticeDimensions()[0] ),
				_n_x( lattice_info.GetLatticeDimensions()[0] ),
				_n_y( lattice_info.GetLatticeDimensions()[1] ),
				_n_z( lattice_info.GetLatticeDimensions()[2] ),
				_n_t( lattice_info.GetLatticeDimensions()[3] ),
				_n_col( n_col ),
				_n_col_offset(n_complex*_n_colorspin)
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
		float* GetSiteDataPtr(IndexType col, IndexType cb, IndexType site)
		{
			return &data[cb][site*_n_site_offset+col*_n_col_offset];
		}

		inline
		const float* GetSiteDataPtr(IndexType col, IndexType cb, IndexType site) const
			{
				return &data[cb][site*_n_site_offset+col*_n_col_offset];
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

		inline
		const IndexType& GetNxh() const { return _n_xh; }

		inline
		const IndexType& GetNx() const { return _n_x; }

		inline
		const IndexType& GetNy() const { return _n_y; }

		inline
		const IndexType& GetNz() const { return _n_z; }

		inline
		const IndexType& GetNt() const { return _n_t; }

		inline
		const IndexType& GetNCol() const { return _n_col; }

		inline
		const IndexType& GetSiteDataLD() const { return _n_col_offset; }

	private:
		const LatticeInfo& _lattice_info;
		float* data[2];  // Even and odd checkerboards

		const IndexType _n_color;
		const IndexType _n_spin;
		const IndexType _n_colorspin;
		const IndexType _n_site_offset;
		const IndexType _n_xh;
		const IndexType _n_x;
		const IndexType _n_y;
		const IndexType _n_z;
		const IndexType _n_t;
		const IndexType _n_col;
		const IndexType _n_col_offset;


	};




	class CoarseGauge {
	public:
		CoarseGauge(const LatticeInfo& lattice_info) : _lattice_info(lattice_info), data{nullptr,nullptr}, diag_data{nullptr,nullptr},
		invdiag_data{nullptr,nullptr}, AD_data{nullptr,nullptr}, DA_data{nullptr,nullptr},
				_n_color(lattice_info.GetNumColors()),
				_n_spin(lattice_info.GetNumSpins()),
				_n_colorspin(lattice_info.GetNumColors()*lattice_info.GetNumSpins()),
				_n_link_offset(n_complex*_n_colorspin*_n_colorspin),
				_n_site_offset((2*n_dim)*_n_link_offset),
				_n_xh( lattice_info.GetCBLatticeDimensions()[0] ),
				_n_x( lattice_info.GetLatticeDimensions()[0] ),
				_n_y( lattice_info.GetLatticeDimensions()[1] ),
				_n_z( lattice_info.GetLatticeDimensions()[2] ),
				_n_t( lattice_info.GetLatticeDimensions()[3] )
		{
			// Check That we have 2 spins
			if( lattice_info.GetNumSpins() != 2 ) {
				MasterLog(ERROR, "Attempting to Create CoarseSpinor with num_spins != 2");
			}


			// Allocate Data - data, AD data and DA data are the off-diagonal links - 8 links per site (use n_site_iffset)
			IndexType offdiag_num_floats_per_cb = _lattice_info.GetNumCBSites()*_n_site_offset;

			// diag_data and invdiag data hold the clover terms. These are 1 link per site (use _n_link_offset)
			IndexType diag_num_floats_per_cb = _lattice_info.GetNumCBSites()*_n_link_offset;

			/* Contiguous allocation */
			data[0] = (float *)MG::MemoryAllocate(offdiag_num_floats_per_cb*sizeof(float), MG::REGULAR);
			data[1] = (float *)MG::MemoryAllocate(offdiag_num_floats_per_cb*sizeof(float), MG::REGULAR);

			diag_data[0] = (float *)MG::MemoryAllocate(diag_num_floats_per_cb*sizeof(float), MG::REGULAR);
			diag_data[1] = (float *)MG::MemoryAllocate(diag_num_floats_per_cb*sizeof(float), MG::REGULAR);

			invdiag_data[0] = (float *)MG::MemoryAllocate(diag_num_floats_per_cb*sizeof(float), MG::REGULAR);
			invdiag_data[1] = (float *)MG::MemoryAllocate(diag_num_floats_per_cb*sizeof(float), MG::REGULAR);

			AD_data[0] = (float *)MG::MemoryAllocate(offdiag_num_floats_per_cb*sizeof(float), MG::REGULAR);
			AD_data[1] = (float *)MG::MemoryAllocate(offdiag_num_floats_per_cb*sizeof(float), MG::REGULAR);

			DA_data[0] = (float *)MG::MemoryAllocate(offdiag_num_floats_per_cb*sizeof(float), MG::REGULAR);
			DA_data[1] = (float *)MG::MemoryAllocate(offdiag_num_floats_per_cb*sizeof(float), MG::REGULAR);

		}

#if 0
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

		/** GetSiteEODataPtr
		 *
		 *  Returns a pointer to the eo_data for a site in a cb
		 *  This is essentially a float array of size _n_site_offset
		 *  or it can be reinterpreted as _n_colorspin complexes
		 */
		inline
		float* GetSiteADDataPtr(IndexType cb, IndexType site)
		{
			return &AD_data[cb][site*_n_site_offset];
		}

		inline
		float* GetSiteDADataPtr(IndexType cb, IndexType site)
		{
			return &DA_data[cb][site*_n_site_offset];
		}

		/** GetSiteData
		 *
		 *  Returns a const pointer to the data for a site in a cb
		 *  This is essentially a float array of size _n_site_offset
		 *  or it can be reinterpreted as _n_colorspin complexes
		 */
		inline
		const float* GetSiteDataPtr(IndexType cb, IndexType site) const
		{
			return &data[cb][site*_n_site_offset];
		}

		/** GetSiteEOData
		 *
		 *  Returns a const pointer to the eo_data for a site in a cb
		 *  This is essentially a float array of size _n_site_offset
		 *  or it can be reinterpreted as _n_colorspin complexes
		 */
		inline
		const float* GetSiteADDataPtr(IndexType cb, IndexType site) const
		{
			return &AD_data[cb][site*_n_site_offset];
		}

		inline
		const float* GetSiteDADataPtr(IndexType cb, IndexType site) const
		{
			return &DA_data[cb][site*_n_site_offset];
		}
#endif
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

		/** GetSiteDirADData
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
		 */
		inline
		float *GetSiteDirADDataPtr(IndexType cb, IndexType site, IndexType mu)
		{
			return &AD_data[cb][site*_n_site_offset + mu*_n_link_offset];
		}

		inline
		const float *GetSiteDirADDataPtr(IndexType cb, IndexType site, IndexType mu) const
		{
			return &AD_data[cb][site*_n_site_offset + mu*_n_link_offset];
		}

		/** GetSiteDirDAData
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
			 */
		inline
		float *GetSiteDirDADataPtr(IndexType cb, IndexType site, IndexType mu)
		{
			return &DA_data[cb][site*_n_site_offset + mu*_n_link_offset];
		}

		inline
		const float *GetSiteDirDADataPtr(IndexType cb, IndexType site, IndexType mu) const
		{
			return &DA_data[cb][site*_n_site_offset + mu*_n_link_offset];
		}


		inline
		float *GetSiteDiagDataPtr(IndexType cb, IndexType site)
		{
			return &diag_data[cb][site*_n_link_offset];
		}

		inline
		const float *GetSiteDiagDataPtr(IndexType cb, IndexType site) const
		{
			return &diag_data[cb][site*_n_link_offset];
		}

		inline
		float *GetSiteInvDiagDataPtr(IndexType cb, IndexType site)
		{
			return &invdiag_data[cb][site*_n_link_offset];

		}

		inline
		const float *GetSiteInvDiagDataPtr(IndexType cb, IndexType site) const
		{
			return &invdiag_data[cb][site*_n_link_offset];

		}


		~CoarseGauge()
		{
			MemoryFree(data[0]);
			MemoryFree(data[1]);
			MemoryFree(diag_data[0]);
			MemoryFree(diag_data[1]);
			MemoryFree(invdiag_data[0]);
			MemoryFree(invdiag_data[1]);
			MemoryFree(AD_data[0]);
			MemoryFree(AD_data[1]);
			MemoryFree(DA_data[0]);
			MemoryFree(DA_data[1]);

			data[0] = nullptr;
			data[1] = nullptr;
			diag_data[0] = nullptr;
			diag_data[1] = nullptr;
			invdiag_data[0] = nullptr;
			invdiag_data[1] = nullptr;
			AD_data[0] = nullptr;
			AD_data[1] = nullptr;
			DA_data[0] = nullptr;
			DA_data[1] = nullptr;
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
		IndexType GetSiteOffset() const {
			return _n_site_offset;
		}

		inline
		const LatticeInfo& GetInfo() const {
			 return _lattice_info;
		}

		inline
		const IndexType& GetNxh() const { return _n_xh; }

		inline
		const IndexType& GetNx() const { return _n_x; }

		inline
		const IndexType& GetNy() const { return _n_y; }

		inline
		const IndexType& GetNz() const { return _n_z; }

		inline
		const IndexType& GetNt() const { return _n_t; }

	private:
		const LatticeInfo& _lattice_info;
		float* data[2];        // Even and odd checkerboards off diagonal data (D)
		float* diag_data[2];   // Diagonal data (Clov, or A)
		float* invdiag_data[2]; // Inverse Clover (A^{-1})
		float* AD_data[2]; // holds A^{-1}_oo D_oe and A^{-1}_ee D_eo (AD)
		float* DA_data[2]; // holds D_oe A^{-1}_ee and D_eo A^{-1}_oo (DA)


		const IndexType _n_color;
		const IndexType _n_spin;
		const IndexType _n_colorspin;
		const IndexType _n_link_offset;
		const IndexType _n_site_offset;
		const IndexType _n_xh;
		const IndexType _n_x;
		const IndexType _n_y;
		const IndexType _n_z;
		const IndexType _n_t;

	};


	template<typename T>
	static size_t haloDatumSize(const LatticeInfo& info);

	template<>
	inline
	size_t haloDatumSize<CoarseSpinor>(const LatticeInfo& info)
	{
		return n_complex*info.GetNumColorSpins();
	}

	template<>
	inline
	size_t haloDatumSize<CoarseGauge>(const LatticeInfo& info)
	{
		return n_complex*info.GetNumColorSpins()*info.GetNumColorSpins();
	}

}



#endif /* INCLUDE_LATTICE_COARSE_COARSE_TYPES_H_ */
