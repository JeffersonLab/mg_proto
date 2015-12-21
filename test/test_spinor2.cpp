#include "gtest/gtest.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/layouts/compact_cb_aos_spinor_layout.h"
#include "lattice/lattice_spinor.h"
#include "lattice/block_operations.h"
#include "utils/print_utils.h"
#include "utils/memory.h"
#include <vector>

#include "test_env.h"
#include "mock_nodeinfo.h"
#include <cmath>

#include <random>
using namespace MGGeometry;
using namespace MGUtils;

/* Test the layout functions */
TEST(TestLayout, TestCBLayoutCreate)
{
	IndexArray latdims={{4,3,3,4}};
	LatticeInfo info(latdims);

	std::size_t before_alloc = GetCurrentRegularMemoryUsage();
	CompactCBAOSSpinorLayout<float> float_layout(info);

	MasterLog(DEBUG, "Float layout needs %u bytes", float_layout.GetDataInBytes());
	{
		MasterLog(DEBUG, "Memory USage before alloc = %u bytes", before_alloc);
		GenericLayoutContainer<float, CompactCBAOSSpinorLayout<float>> f_spinor(float_layout);
		std::size_t after_alloc = GetCurrentRegularMemoryUsage();
		MasterLog(DEBUG, "Memory Usage after alloc = %u bytes", after_alloc);

		std::size_t alloced = after_alloc - before_alloc;
		MasterLog(DEBUG, "Allocating spinor alloced %u bytes", alloced);

		// i) Check that the correct number of bytes was allocated
		// ii) Check that the correct number of bytes is what we think it ought to be
		ASSERT_EQ( float_layout.GetDataInBytes(), alloced);
		ASSERT_EQ( float_layout.GetDataInBytes(),
				float_layout.GetNumData()*sizeof(float));
	}
	// Destroy spinor on leaving the block
	// Check that the memory is back to where it was before.
	ASSERT_EQ( before_alloc, GetCurrentRegularMemoryUsage());
}

TEST(TestSpinor, TestCBLayout_site)
{
	IndexArray latdims={{4,3,3,4}};
	LatticeInfo info(latdims);
	CompactCBAOSSpinorLayout<float> float_layout(info);

	IndexType n_colors = info.GetNumColors();
	IndexType n_spins  = info.GetNumSpins();

#pragma omp parallel for shared(n_colors, n_spins)
	for(IndexType site=0; site < info.GetNumSites(); ++site) {


		IndexType site_index = site*n_colors*n_spins*n_complex;
		IndexType offset_in_site = 0;

		for(IndexType spin=0; spin < n_spins; ++spin) {
			for(IndexType color=0; color < n_colors; ++color) {
				for(IndexType reim=0; reim < n_complex; ++reim) {
					EXPECT_EQ( float_layout.ContainerIndex(site,spin,color,reim),
								site_index + offset_in_site );
					offset_in_site++;
				}
			}
		}

	}

}

TEST(TestSpinor, TestCBLayout_site2)
{
	IndexArray latdims={{4,3,3,4}};
	LatticeInfo info_even(latdims);
	CompactCBAOSSpinorLayout<float> float_layout(info_even);

	IndexType n_colors = info_even.GetNumColors();
	IndexType n_spins  = info_even.GetNumSpins();

	IndexType n_cb_sites = info_even.GetNumCBSites();
	for(IndexType cb=0; cb < n_checkerboard; ++cb ) {

#pragma omp parallel for shared(n_colors, n_spins,n_cb_sites)
		for(IndexType cbsite=0; cbsite < n_cb_sites; ++cbsite) {


		IndexType site_index = (cbsite + cb*n_cb_sites)*n_colors*n_spins*n_complex;
		IndexType offset_in_site = 0;

		for(IndexType spin=0; spin < n_spins; ++spin) {
			for(IndexType color=0; color < n_colors; ++color) {
				for(IndexType reim=0; reim < n_complex; ++reim) {
					EXPECT_EQ( float_layout.ContainerIndex(cb, cbsite, spin,color,reim),
								site_index + offset_in_site );
					offset_in_site++;
				}
			}
		}

	}
	}

}

TEST(TestSpinor, TestCBLayout_Coords)
{
	IndexArray latdims={{4,3,3,4}};
	LatticeInfo info_even(latdims);
	CompactCBAOSSpinorLayout<float> float_layout(info_even);



	IndexType n_colors = info_even.GetNumColors();
	IndexType n_spins  = info_even.GetNumSpins();
	IndexType n_cb_sites = info_even.GetNumCBSites();
	IndexArray cb_latdims = info_even.GetCBLatticeDimensions();

	IndexType tdims=latdims[T_DIR];
	IndexType zdims=latdims[Z_DIR];
	IndexType ydims=latdims[Y_DIR];
	IndexType xdims=latdims[X_DIR];
#pragma omp parallel for collapse(4)
	for(IndexType t=0; t < tdims; ++t) {
		for(IndexType z=0; z < zdims; ++z) {
			for(IndexType y=0; y < ydims; ++y) {
				for(IndexType x=0; x < xdims; ++x) {

					// Site index within a checkerboard
					IndexType site_index = CoordsToIndex({{x/2,y,z,t}}, cb_latdims);


					IndexType cb = (x + y + z + t) &1;
					// Add checkerbaord offset
					site_index += cb*n_cb_sites;

					// Multiply by sizeof site
					site_index *= (n_colors*n_spins*n_complex);

					IndexType offset_in_site = 0;
					for(IndexType spin=0; spin < n_spins; ++spin) {
						for(IndexType color=0; color < n_colors; ++color) {
							for(IndexType reim=0; reim < n_complex; ++reim) {
								EXPECT_EQ( float_layout.ContainerIndex({{x,y,z,t}}, spin,color,reim),
										site_index + offset_in_site );
								offset_in_site++;
							}
						}
					}
				}
			}
		}
	}

}

TEST(TestSpinor, TestCBLayout_CoordsOddnode)
{
	IndexArray latdims={{2,3,3,2}};
	MockNodeInfo nodeinfo( {{2,2,1,1}}, {{0,1,0,0}});
	LatticeInfo info_even(latdims, 4,3, nodeinfo);
	CompactCBAOSSpinorLayout<float> float_layout(info_even);



	IndexType n_colors = info_even.GetNumColors();
	IndexType n_spins  = info_even.GetNumSpins();
	IndexType n_cb_sites = info_even.GetNumCBSites();
	IndexArray cb_latdims = info_even.GetCBLatticeDimensions();

	IndexType tdims = latdims[T_DIR];
	IndexType zdims = latdims[Z_DIR];
	IndexType ydims = latdims[Y_DIR];
	IndexType xdims = latdims[X_DIR];

#pragma omp parallel for collapse(4)
	for(IndexType t=0; t < tdims; ++t) {
		for(IndexType z=0; z < zdims; ++z) {
			for(IndexType y=0; y < ydims; ++y) {
				for(IndexType x=0; x < xdims; ++x) {


					IndexType site_index = CoordsToIndex({{x/2,y,z,t}}, cb_latdims);
					IndexType cb = (x + y + z + t + info_even.GetCBOrigin()) &1 ;

					// Add checkerbaord offset
					site_index += cb*n_cb_sites;

					// Multiply by sizeof site
					site_index *= (n_colors*n_spins*n_complex);

					IndexType offset_in_site = 0;
					for(IndexType spin=0; spin < n_spins; ++spin) {
						for(IndexType color=0; color < n_colors; ++color) {
							for(IndexType reim=0; reim < n_complex; ++reim) {
								EXPECT_EQ( float_layout.ContainerIndex({{x,y,z,t}}, spin,color,reim),
										site_index + offset_in_site );
								offset_in_site++;
							}
						}
					}
				}
			}
		}
	}
}

//
TEST(TestLayout, TestCBSOALayoutCreate)
{
	// This should require padding
	// One CB is 2x3x5x2 = 60 sites = 120 floats = 480 bytes = 7.5 lines
	// So it will need padding for 1/2 a line = 32 bytes = 8 floats = 4 sites ( a site is a complex)
	//
	// So the allocated will be 64 sites per cb => 128 sites per spin/color => 1536 complex = 3072 floats
	IndexArray latdims={{4,3,5,2}};
	LatticeInfo info(latdims);
	CBSOASpinorLayout<float> float_layout(info);

	ASSERT_EQ(float_layout.GetNumData(), static_cast<IndexType>(3072));
}

TEST(TestLayout, TestCBSOALayoutSpinorCreate)
{
	IndexArray latdims={{4,3,5,2}};
	LatticeInfo info(latdims);
	CBSOASpinorLayout<float> float_layout(info);

	std::size_t before_alloc = GetCurrentRegularMemoryUsage();
	MasterLog(DEBUG, "Float layout needs %u bytes", float_layout.GetDataInBytes());
	{
		MasterLog(DEBUG, "Memory USage before alloc = %u bytes", before_alloc);
		GenericLayoutContainer<float, CBSOASpinorLayout<float> > f_spinor(float_layout);
		std::size_t after_alloc = GetCurrentRegularMemoryUsage();
		MasterLog(DEBUG, "Memory Usage after alloc = %u bytes", after_alloc);

		std::size_t alloced = after_alloc - before_alloc;
		MasterLog(DEBUG, "Allocating spinor alloced %u bytes", alloced);

		// i) Check that the correct number of bytes was allocated
		// ii) Check that the correct number of bytes is what we think it ought to be
		ASSERT_EQ( float_layout.GetDataInBytes(), alloced);
		ASSERT_EQ( float_layout.GetDataInBytes(),
				float_layout.GetNumData()*sizeof(float));
	}
	// Destroy spinor on leaving the block
	// Check that the memory is back to where it was before.
	ASSERT_EQ( before_alloc, GetCurrentRegularMemoryUsage());
}


TEST(TestSpinor, TestCBSOALayout_site)
{
	IndexArray latdims={{4,3,3,4}};
	LatticeInfo info(latdims);
	CBSOASpinorLayout<float> float_layout(info);

	IndexType n_colors = info.GetNumColors();
	IndexType n_spins  = info.GetNumSpins();
	// Stride is from start of 1 cb to the next
	IndexType cb_stride =  (float_layout.ContainerIndex(1,0,0,0,0) -  float_layout.ContainerIndex(0,0,0,0,0))/n_complex;
#pragma omp parallel for shared(n_colors, n_spins) collapse(2)
	for(IndexType spin=0; spin < n_spins; ++spin) {
		for(IndexType color=0; color < n_colors; ++color) {


			IndexType start_offset = float_layout.ContainerIndex(0,spin,color,0);

			for(IndexType site=0; site < info.GetNumSites(); ++site) {



        		IndexType cb = site/info.GetNumCBSites();
        		IndexType cbsite = site % info.GetNumCBSites();
        		IndexType site_index = n_complex*(cbsite + cb_stride *cb) + start_offset;
        		IndexType offset_in_site =0;
				for(IndexType reim=0; reim < n_complex; ++reim) {
					EXPECT_EQ( float_layout.ContainerIndex(site,spin,color,reim),
								site_index + offset_in_site );
					offset_in_site++;
				}
			}
		}

	}

}

TEST(TestSpinor, TestCBSOALayout_cbsite)
{
	IndexArray latdims={{4,3,3,4}};
	LatticeInfo info(latdims);
	CBSOASpinorLayout<float> float_layout(info);

	IndexType n_colors = info.GetNumColors();
	IndexType n_spins  = info.GetNumSpins();

	IndexType cb_stride = (float_layout.ContainerIndex(1,0,0,0,0)-float_layout.ContainerIndex(0,0,0,0,0))/n_complex;
#pragma omp parallel for shared(n_colors, n_spins) collapse(2)
	for(IndexType spin=0; spin < n_spins; ++spin) {
		for(IndexType color=0; color < n_colors; ++color) {

			IndexType start_offset = float_layout.ContainerIndex(0,spin,color,0);

			for(IndexType cb =0; cb < n_checkerboard; ++cb) {
				for(IndexType cbsite=0; cbsite < info.GetNumCBSites(); ++cbsite) {

					IndexType site_index = n_complex*(cbsite + cb_stride*cb) + start_offset;
					IndexType offset_in_site =0;
					for(IndexType reim=0; reim < n_complex; ++reim) {
						EXPECT_EQ( float_layout.ContainerIndex(cb, cbsite,spin,color,reim),
								site_index + offset_in_site );
						offset_in_site++;
					}
				}
			}
		}

	}

}

TEST(TestSpinor, TestCBSOALayout_Coords)
{
	IndexArray latdims={{4,3,3,4}};
	LatticeInfo info(latdims);
	CBSOASpinorLayout<float> float_layout(info);

	IndexType n_colors = info.GetNumColors();
	IndexType n_spins  = info.GetNumSpins();
	IndexArray cb_latdims = info.GetCBLatticeDimensions();

	IndexType tdims = latdims[T_DIR];
	IndexType zdims = latdims[Z_DIR];
	IndexType ydims = latdims[Y_DIR];
	IndexType xdims = latdims[X_DIR];


	IndexType cb_stride = (float_layout.ContainerIndex(1,0,0,0,0)-float_layout.ContainerIndex(0,0,0,0,0))/n_complex;

#pragma omp parallel for shared(n_colors, n_spins) collapse(6)
	for(IndexType spin=0; spin < n_spins; ++spin) {
		for(IndexType color=0; color < n_colors; ++color) {
			for(IndexType t=0; t < tdims; ++t) {
				for(IndexType z=0; z < zdims; ++z) {
					for(IndexType y=0; y < ydims; ++y) {
						for(IndexType x=0; x < xdims; ++x) {

							// Site index within a checkerboard
							IndexType cbsite = CoordsToIndex({{x/2,y,z,t}}, cb_latdims);
							IndexType cb = (x + y + z + t) &1;

							IndexType start_offset = (color+n_colors*spin)*n_checkerboard*cb_stride*n_complex;
							IndexType site_index = n_complex*(cbsite + cb_stride*cb) + start_offset;

							IndexType offset_in_site =0;
							for(IndexType reim=0; reim < n_complex; ++reim) {
								EXPECT_EQ( float_layout.ContainerIndex({{x,y,z,t}},spin,color,reim),
										site_index + offset_in_site );
								offset_in_site++;
							}
						}
					}
				}
			}
		}
	}

}

TEST(TestSpinor, TestCBSOALayout_CoordsOddNode)
{
	IndexArray latdims={{2,3,3,2}};
	MockNodeInfo nodeinfo( {{2,2,1,1}}, {{0,1,0,0}});
	LatticeInfo info(latdims,4,3,nodeinfo);

	CBSOASpinorLayout<float> float_layout(info);

	IndexType n_colors = info.GetNumColors();
	IndexType n_spins  = info.GetNumSpins();
	IndexArray cb_latdims = info.GetCBLatticeDimensions();

	IndexType tdims = latdims[T_DIR];
	IndexType zdims = latdims[Z_DIR];
	IndexType ydims = latdims[Y_DIR];
	IndexType xdims = latdims[X_DIR];

    IndexType cb_stride = (float_layout.ContainerIndex(1,0,0,0,0)-float_layout.ContainerIndex(0,0,0,0,0))/n_complex;

#pragma omp parallel for shared(n_colors, n_spins) collapse(6)
	for(IndexType spin=0; spin < n_spins; ++spin) {
		for(IndexType color=0; color < n_colors; ++color) {
			for(IndexType t=0; t < tdims; ++t) {
				for(IndexType z=0; z < zdims; ++z) {
					for(IndexType y=0; y < ydims; ++y) {
						for(IndexType x=0; x < xdims; ++x) {

							// Site index within a checkerboard
							IndexType cbsite = CoordsToIndex({{x/2,y,z,t}}, cb_latdims);
							IndexType cb = (x + y + z + t + info.GetCBOrigin() ) &1;

							IndexType start_offset = n_complex*cb_stride*n_checkerboard*(color+n_colors*spin);
							IndexType site_index = n_complex*(cbsite + cb_stride*cb) + start_offset;


							for(IndexType reim=0; reim < n_complex; ++reim) {

								EXPECT_EQ( float_layout.ContainerIndex({{x,y,z,t}},spin,color,reim),
										site_index + reim);

							}
						}
					}
				}
			}
		}
	}

}

TEST(TestSpinor, TestCBSOALayoutCBAligned)
{
	IndexArray latdims={{4,3,5,2}};
	LatticeInfo info(latdims);
	CBSOASpinorLayout<float> float_layout(info);
	GenericLayoutContainer<float, CBSOASpinorLayout<float>> my_spinor(float_layout);

	IndexType n_colors = info.GetNumColors();
	IndexType n_spins  = info.GetNumSpins();

#pragma omp parallel for shared(n_colors, n_spins) collapse(2)
	for(IndexType spin=0; spin < n_spins; ++spin) {
		for(IndexType color=0; color < n_colors; ++color) {
			float *cb0_start = &(my_spinor.Index(0,0,spin,color,RE));  // First site of cb = 0
			float *cb1_start = &(my_spinor.Index(1,0,spin,color,RE));  // First site of cb = 1;
			EXPECT_EQ( reinterpret_cast<unsigned long>(cb0_start) & (MG_DEFAULT_ALIGNMENT-1), static_cast<IndexType>(0));
			EXPECT_EQ( reinterpret_cast<unsigned long>(cb1_start) & (MG_DEFAULT_ALIGNMENT-1), static_cast<IndexType>(0));
		}
	}
}

TEST(TestSpinor, TestCBSOALayoutCBAlignedDouble)
{
	IndexArray latdims={{4,3,5,2}};
	LatticeInfo info(latdims);
	CBSOASpinorLayout<double> float_layout(info);
	GenericLayoutContainer<double, CBSOASpinorLayout<double>> my_spinor(float_layout);

	IndexType n_colors = info.GetNumColors();
	IndexType n_spins  = info.GetNumSpins();

#pragma omp parallel for shared(n_colors, n_spins) collapse(2)
	for(IndexType spin=0; spin < n_spins; ++spin) {
		for(IndexType color=0; color < n_colors; ++color) {
			double *cb0_start = &(my_spinor.Index(0,0,spin,color,RE));  // First site of cb = 0
			double *cb1_start = &(my_spinor.Index(1,0,spin,color,RE));  // First site of cb = 1;
			EXPECT_EQ( reinterpret_cast<unsigned long>(cb0_start) & (MG_DEFAULT_ALIGNMENT-1), static_cast<IndexType>(0));
			EXPECT_EQ( reinterpret_cast<unsigned long>(cb1_start) & (MG_DEFAULT_ALIGNMENT-1), static_cast<IndexType>(0));
		}
	}
}

TEST(TestSpinor, TestSpinorNormSq)
{
	IndexArray latdims={{4,4,4,4}}; // Each cb: 2*4*4*4 = 128 sites => 128 * n_complex * sizeof(float) = 1024 bytes = 16 blocks of length 64 bytes - no padding.
	LatticeInfo info(latdims);
	CBSOASpinorLayout<double> float_layout(info);
	GenericLayoutContainer<double, CBSOASpinorLayout<double>> my_spinor(float_layout);

	std::complex<double> one_plus_i = std::complex<double>{1,1};
	Fill(one_plus_i, my_spinor);
	double expected_norm_sq = 2*info.GetNumColors()*info.GetNumSpins()*info.GetNumSites();
	EXPECT_DOUBLE_EQ( NormSq(my_spinor), expected_norm_sq);

}

TEST(TestSpinor, TestSpinorNormSqPadded)
{
	IndexArray latdims={{2,2,3,3}}; // Each cb: 1*2*3*3 = 18 sites => 18 * n_complex * sizeof(float) = 144 bytes = 2 blocks of length 64 bytes + 32 bytes
	// - padding of 32 bytes.
	LatticeInfo info(latdims);
	CBSOASpinorLayout<double> float_layout(info);
	GenericLayoutContainer<double, CBSOASpinorLayout<double>> my_spinor(float_layout);

	std::complex<double> one_plus_i= std::complex<double>{1,1};
	Fill(one_plus_i, my_spinor);
	double expected_norm_sq = 2*info.GetNumColors()*info.GetNumSpins()*info.GetNumSites();
	EXPECT_DOUBLE_EQ( NormSq(my_spinor), expected_norm_sq);

}


TEST(TestSpinor, TestInnerProd)
{
	IndexArray latdims={{4,4,4,4}}; // Each cb: 2*4*4*4 = 128 sites => 128 * n_complex * sizeof(float) = 1024 bytes = 16 blocks of length 64 bytes - no padding.
	LatticeInfo info(latdims);
	CBSOASpinorLayout<double> float_layout(info);
	GenericLayoutContainer<double, CBSOASpinorLayout<double>> left(float_layout);
	GenericLayoutContainer<double, CBSOASpinorLayout<double>> right(float_layout);
	std::complex<double> one_plus_i = std::complex<double>{1,1};
	std::complex<double> one_minus_i = std::complex<double>{1,-1};

	Fill(one_plus_i, right);
	Fill(one_minus_i, left );


	// Check the fill worked;
	double expected_norm_sq = 2*info.GetNumColors()*info.GetNumSpins()*info.GetNumSites();
	EXPECT_DOUBLE_EQ( NormSq( right ), expected_norm_sq );
	EXPECT_DOUBLE_EQ( NormSq( left  ), expected_norm_sq );

	/* For each element of the iprod we have:  conj(1 - i)*(1 + i) = (1+i)*(1+i) = 1 + i + i + i^2 = 2i
	 * So summed over each site, spin_color the real part ought to be 0, the imag part ought to be 2*n_dof = expected norm_sq
	 */
	std::complex<double> iprod = InnerProduct(left,right);
	EXPECT_DOUBLE_EQ( 0 , iprod.real());
	EXPECT_DOUBLE_EQ( expected_norm_sq, iprod.imag());


}

TEST(TestSpinor, TestSpinorInnerProdPadded)
{
	IndexArray latdims={{2,2,3,3}};  // Each cb: 1*2*3*3 = 18 sites => 18 * n_complex * sizeof(float) = 144 bytes = 2 blocks of length 64 bytes + 32 bytes
	LatticeInfo info(latdims);
	CBSOASpinorLayout<double> float_layout(info);
	GenericLayoutContainer<double, CBSOASpinorLayout<double>> left(float_layout);
	GenericLayoutContainer<double, CBSOASpinorLayout<double>> right(float_layout);
	std::complex<double> one_plus_i = std::complex<double>{1,1};
	std::complex<double> one_minus_i = std::complex<double>{1,-1};

	Fill(one_plus_i, right);
	Fill(one_minus_i, left );


	double expected_norm_sq = 2*info.GetNumColors()*info.GetNumSpins()*info.GetNumSites();
	// Check the fill worked;
	EXPECT_DOUBLE_EQ( NormSq( right ), expected_norm_sq );
	EXPECT_DOUBLE_EQ( NormSq( left  ), expected_norm_sq );

	/* For each element of the iprod we have:  conj(1 - i)*(1 + i) = (1+i)*(1+i) = 1 + i + i + i^2 = 2i
	 * So summed over each site, spin_color the real part ought to be 0, the imag part ought to be 2*n_dof = expected norm_sq
	 */

	std::complex<double> iprod = InnerProduct(left,right);
	EXPECT_DOUBLE_EQ( expected_norm_sq, iprod.imag());
	EXPECT_DOUBLE_EQ(  0, iprod.real());
}

TEST(TestSpinor, TestSpinorVScale)
{
	IndexArray latdims={{4,4,4,4}}; // Each cb: 2*4*4*4 = 128 sites => 128 * n_complex * sizeof(float) = 1024 bytes = 16 blocks of length 64 bytes - no padding.
	LatticeInfo info(latdims);
	CBSOASpinorLayout<double> float_layout(info);
	GenericLayoutContainer<double, CBSOASpinorLayout<double>> my_spinor(float_layout);

	std::complex<double> one_plus_i = std::complex<double>{1,1};
	Fill(one_plus_i, my_spinor);


	VScale(std::sqrt(0.5), my_spinor);

	double expected_norm_sq = info.GetNumColors()*info.GetNumSpins()*info.GetNumSites();
	EXPECT_DOUBLE_EQ( NormSq(my_spinor), expected_norm_sq);

}

TEST(TestSpinor, TestSpinorVScalePadded)
{
	IndexArray latdims={{2,2,3,3}}; // Each cb: 1*2*3*3 = 18 sites => 18 * n_complex * sizeof(float) = 144 bytes = 2 blocks of length 64 bytes + 32 bytes
	// - padding of 32 bytes.
	LatticeInfo info(latdims);
	CBSOASpinorLayout<double> float_layout(info);
	GenericLayoutContainer<double, CBSOASpinorLayout<double>> my_spinor(float_layout);

	std::complex<double> one_plus_i= std::complex<double>{1,1};
	Fill(one_plus_i, my_spinor);


	VScale(std::sqrt(0.5), my_spinor);

	double expected_norm_sq = info.GetNumColors()*info.GetNumSpins()*info.GetNumSites();
	EXPECT_DOUBLE_EQ( NormSq(my_spinor), expected_norm_sq);

}


TEST(TestSpinor, TestMCaxpy)
{
	IndexArray latdims={{4,4,4,4}}; // Each cb: 2*4*4*4 = 128 sites => 128 * n_complex * sizeof(float) = 1024 bytes = 16 blocks of length 64 bytes - no padding.
	LatticeInfo info(latdims);
	CBSOASpinorLayout<double> float_layout(info);
	GenericLayoutContainer<double, CBSOASpinorLayout<double>> x(float_layout);
	GenericLayoutContainer<double, CBSOASpinorLayout<double>> y(float_layout);

	std::complex<double> one_plus_i = std::complex<double>{1,1};
	Fill(one_plus_i, y);
	Fill(one_plus_i, x);

	std::complex<double> half=std::complex<double>{0.5,0};
	MCaxpy(y, half, x);


	double expected_norm_y_sq = 0.5*info.GetNumColors()*info.GetNumSpins()*info.GetNumSites();
	double expected_norm_x_sq = 2*info.GetNumColors()*info.GetNumSpins()*info.GetNumSites();
	EXPECT_DOUBLE_EQ( NormSq(y), expected_norm_y_sq);
	EXPECT_DOUBLE_EQ( NormSq(x), expected_norm_x_sq);
}

TEST(TestSpinor, TestMCaxpyPadded)
{
	IndexArray latdims={{2,2,3,3}}; // Each cb: 1*2*3*3 = 18 sites => 18 * n_complex * sizeof(float) = 144 bytes = 2 blocks of length 64 bytes + 32 bytes
	LatticeInfo info(latdims);
	CBSOASpinorLayout<double> float_layout(info);
	GenericLayoutContainer<double, CBSOASpinorLayout<double>> x(float_layout);
	GenericLayoutContainer<double, CBSOASpinorLayout<double>> y(float_layout);

	std::complex<double> one_plus_i = std::complex<double>{1,1};
	Fill(one_plus_i, y);
	Fill(one_plus_i, x);

	std::complex<double> half=std::complex<double>{0.5,0};
	MCaxpy(y, half, x);


	double expected_norm_y_sq = 0.5*info.GetNumColors()*info.GetNumSpins()*info.GetNumSites();
	double expected_norm_x_sq = 2*info.GetNumColors()*info.GetNumSpins()*info.GetNumSites();
	EXPECT_DOUBLE_EQ( NormSq(y), expected_norm_y_sq);
	EXPECT_DOUBLE_EQ( NormSq(x), expected_norm_x_sq);
}

#if 1
template<typename T, typename BlockedLayout, typename Aggregation>
void BlockOrthonormalize(std::vector<AggregateLayoutContainer<T,BlockedLayout,Aggregation>>& vectors)
{
	auto num_vectors = vectors.size();

	// Dumb?
	if( num_vectors == 0 ) return;

	// Vectors zero now is guaranteed to exist.
	auto& aggr = vectors[0].GetAggregation();

	IndexType num_blocks = aggr.GetNumBlocks();
	IndexType num_outerspins = aggr.GetNumAggregates();

	// There is some amount of nested parallelism needed. I am not going to bother with it
	// I will loop this level without threading, and I'll thread over the actual spinors.

	for(IndexType block =0; block < num_blocks; ++block) {
		for(IndexType outer_spin=0; outer_spin < num_outerspins; ++outer_spin) {

			// This is the sub-spinor type
			using BlockSpinorType = typename ContainerTraits<T,BlockedLayout,GenericLayoutContainer<T,BlockedLayout>>::subview_container_type;

			// A vector to hold the sub-spinors
			std::vector<BlockSpinorType> block_spinors(num_vectors);
			for(auto v=0; v < num_vectors; ++v) {

				block_spinors[v] = vectors[v].GetSubview(block, outer_spin);

			}

			GramSchmidt(block_spinors);

		}
	}
}
#endif

TEST(TestSpinor, BlockOrthogonalize)
{
	using Spinor = LatticeLayoutContainer<float,CBSOASpinorLayout<float>>;
	using BlockSpinor = AggregateLayoutContainer<float, BlockAggregateVectorLayout<float>>;

	// Now I have to be able to create a vector of block spinors. But How?
	// Since I need to be able to
	const int num_vec = 8;
	IndexArray latdims={{4,4,4,4}}; // Each cb: 2*4*4*4 = 128 sites => 128 * n_complex * sizeof(float) = 1024 bytes = 16 blocks of length 64 bytes - no padding.
	LatticeInfo fine_info(latdims);

	/* Initialize num_vec fine lattice spinors */
	std::vector<Spinor> fine_spinors(num_vec, Spinor(fine_info));

	IndexArray blockdims={{2,2,2,2}};
	StandardAggregation aggr(latdims,blockdims); // Create a standard aggregation
	std::vector<BlockSpinor> block_aggregate_spinors(num_vec, BlockSpinor(fine_info,aggr));

	/* Fill spinor with random numbers */
	/* Create a C++ random number generator. Let's use Merseenne Twister */
	std::random_device rd;
	std::mt19937 gen(rd());

	/* Fill all the spinors with noise */
	for(IndexType vec = 0; vec < num_vec; ++vec) {
		FillGaussian(fine_spinors[vec], gen);
	}

	/* Zip to blocked layout */
	for(IndexType vec=0; vec < num_vec; ++vec) {
		zip(block_aggregate_spinors[vec], fine_spinors[vec]);
	}

	BlockOrthonormalize(block_aggregate_spinors);


}

int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

