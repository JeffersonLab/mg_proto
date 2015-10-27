#include "gtest/gtest.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/lattice_spinor.h"
#include "utils/print_utils.h"
#include "utils/memory.h"
#include <vector>

#include "test_env.h"
#include "mock_nodeinfo.h"

using namespace MGGeometry;
using namespace MGUtils;

/* Test the layout functions */
TEST(TestLayout, TestCBLayoutCreate)
{
	IndexArray latdims={{4,3,3,4}};
	LatticeInfo info(latdims);

	std::size_t before_alloc = GetCurrentRegularMemoryUsage();
	CompactCBAOSSpinorLayout<float> float_layout(info);

	MasterLog(DEBUG, "Float layout needs %u bytes", float_layout.DataInBytes());
	{
		MasterLog(DEBUG, "Memory USage before alloc = %u bytes", before_alloc);
		GeneralLatticeSpinor<float, CompactCBAOSSpinorLayout<float> > f_spinor(float_layout);
		std::size_t after_alloc = GetCurrentRegularMemoryUsage();
		MasterLog(DEBUG, "Memory Usage after alloc = %u bytes", after_alloc);

		std::size_t alloced = after_alloc - before_alloc;
		MasterLog(DEBUG, "Allocating spinor alloced %u bytes", alloced);

		// i) Check that the correct number of bytes was allocated
		// ii) Check that the correct number of bytes is what we think it ought to be
		ASSERT_EQ( float_layout.DataInBytes(), alloced);
		ASSERT_EQ( float_layout.DataInBytes(),
				float_layout.DataNumElem()*sizeof(float));
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

#pragma omp parallel for collapse(4)
	for(IndexType t=0; t < latdims[T_DIR]; ++t) {
		for(IndexType z=0; z < latdims[Z_DIR]; ++z) {
			for(IndexType y=0; y < latdims[Y_DIR]; ++y) {
				for(IndexType x=0; x < latdims[X_DIR]; ++x) {

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

	#pragma omp parallel for collapse(4)
	for(IndexType t=0; t < latdims[T_DIR]; ++t) {
		for(IndexType z=0; z < latdims[Z_DIR]; ++z) {
			for(IndexType y=0; y < latdims[Y_DIR]; ++y) {
				for(IndexType x=0; x < latdims[X_DIR]; ++x) {


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

	ASSERT_EQ(float_layout.DataNumElem(), 3072);
}

TEST(TestLayout, TestCBSOALayoutSpinorCreate)
{
	IndexArray latdims={{4,3,5,2}};
	LatticeInfo info(latdims);
	CBSOASpinorLayout<float> float_layout(info);

	std::size_t before_alloc = GetCurrentRegularMemoryUsage();
	MasterLog(DEBUG, "Float layout needs %u bytes", float_layout.DataInBytes());
	{
		MasterLog(DEBUG, "Memory USage before alloc = %u bytes", before_alloc);
		GeneralLatticeSpinor<float, CBSOASpinorLayout<float> > f_spinor(float_layout);
		std::size_t after_alloc = GetCurrentRegularMemoryUsage();
		MasterLog(DEBUG, "Memory Usage after alloc = %u bytes", after_alloc);

		std::size_t alloced = after_alloc - before_alloc;
		MasterLog(DEBUG, "Allocating spinor alloced %u bytes", alloced);

		// i) Check that the correct number of bytes was allocated
		// ii) Check that the correct number of bytes is what we think it ought to be
		ASSERT_EQ( float_layout.DataInBytes(), alloced);
		ASSERT_EQ( float_layout.DataInBytes(),
				float_layout.DataNumElem()*sizeof(float));
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

#pragma omp parallel for shared(n_colors, n_spins) collapse(2)
	for(IndexType spin=0; spin < n_spins; ++spin) {
		for(IndexType color=0; color < n_colors; ++color) {

			IndexType start_offset = (color+n_colors*spin)*n_checkerboard*float_layout.GetCBSitesStride()*n_complex;

			for(IndexType site=0; site < info.GetNumSites(); ++site) {



        		IndexType cb = site/info.GetNumCBSites();
        		IndexType cbsite = site % info.GetNumCBSites();
        		IndexType site_index = n_complex*(cbsite + float_layout.GetCBSitesStride()*cb) + start_offset;
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

#pragma omp parallel for shared(n_colors, n_spins) collapse(2)
	for(IndexType spin=0; spin < n_spins; ++spin) {
		for(IndexType color=0; color < n_colors; ++color) {

			IndexType start_offset = (color+n_colors*spin)*n_checkerboard*float_layout.GetCBSitesStride()*n_complex;

			for(IndexType cb =0; cb < n_checkerboard; ++cb) {
				for(IndexType cbsite=0; cbsite < info.GetNumCBSites(); ++cbsite) {

					IndexType site_index = n_complex*(cbsite + float_layout.GetCBSitesStride()*cb) + start_offset;
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

#pragma omp parallel for shared(n_colors, n_spins) collapse(6)
	for(IndexType spin=0; spin < n_spins; ++spin) {
		for(IndexType color=0; color < n_colors; ++color) {
			for(IndexType t=0; t < latdims[T_DIR]; ++t) {
				for(IndexType z=0; z < latdims[Z_DIR]; ++z) {
					for(IndexType y=0; y < latdims[Y_DIR]; ++y) {
						for(IndexType x=0; x < latdims[X_DIR]; ++x) {

							// Site index within a checkerboard
							IndexType cbsite = CoordsToIndex({{x/2,y,z,t}}, cb_latdims);
							IndexType cb = (x + y + z + t) &1;

							IndexType start_offset = (color+n_colors*spin)*n_checkerboard*float_layout.GetCBSitesStride()*n_complex;
							IndexType site_index = n_complex*(cbsite + float_layout.GetCBSitesStride()*cb) + start_offset;

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

TEST(TestSpinor, TestCBSOALayout_CoordsCB)
{
	IndexArray latdims={{2,3,3,2}};
	MockNodeInfo nodeinfo( {{2,2,1,1}}, {{0,1,0,0}});
	LatticeInfo info(latdims,4,3,nodeinfo);

	CBSOASpinorLayout<float> float_layout(info);

	IndexType n_colors = info.GetNumColors();
	IndexType n_spins  = info.GetNumSpins();
	IndexArray cb_latdims = info.GetCBLatticeDimensions();

#pragma omp parallel for shared(n_colors, n_spins) collapse(6)
	for(IndexType spin=0; spin < n_spins; ++spin) {
		for(IndexType color=0; color < n_colors; ++color) {
			for(IndexType t=0; t < latdims[T_DIR]; ++t) {
				for(IndexType z=0; z < latdims[Z_DIR]; ++z) {
					for(IndexType y=0; y < latdims[Y_DIR]; ++y) {
						for(IndexType x=0; x < latdims[X_DIR]; ++x) {

							// Site index within a checkerboard
							IndexType cbsite = CoordsToIndex({{x/2,y,z,t}}, cb_latdims);
							IndexType cb = (x + y + z + t + info.GetCBOrigin() ) &1;

							IndexType start_offset = (color+n_colors*spin)*n_checkerboard*float_layout.GetCBSitesStride()*n_complex;
							IndexType site_index = n_complex*(cbsite + float_layout.GetCBSitesStride()*cb) + start_offset;

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
#if 0
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

#pragma omp parallel for collapse(4)
	for(IndexType t=0; t < latdims[T_DIR]; ++t) {
		for(IndexType z=0; z < latdims[Z_DIR]; ++z) {
			for(IndexType y=0; y < latdims[Y_DIR]; ++y) {
				for(IndexType x=0; x < latdims[X_DIR]; ++x) {

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

	#pragma omp parallel for collapse(4)
	for(IndexType t=0; t < latdims[T_DIR]; ++t) {
		for(IndexType z=0; z < latdims[Z_DIR]; ++z) {
			for(IndexType y=0; y < latdims[Y_DIR]; ++y) {
				for(IndexType x=0; x < latdims[X_DIR]; ++x) {


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
#endif

int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

