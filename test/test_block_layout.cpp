#include "gtest/gtest.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/lattice_spinor.h"
#include "utils/print_utils.h"
#include "utils/memory.h"
#include <vector>

#include "test_env.h"
#include "mock_nodeinfo.h"
#include "lattice/layouts/block_cb_soa_spinor_layout.h"
#include "lattice/block_operations.h"

using namespace MG;
using namespace MG;

TEST(TestBlockLayout, TestBlockLayoutCreate)
{
	IndexArray latdims={{8,8,8,8}};
	IndexArray blockdims={{2,2,2,2}};
	LatticeInfo linfo(latdims);
	StandardAggregation aggr(latdims, blockdims);

	// Make a blocked Layout
	BlockAggregateVectorLayout<float> block_layout(linfo,aggr);

	IndexType num_blocks = block_layout.GetNumBlocks();
	ASSERT_EQ( num_blocks, aggr.GetNumBlocks());
}

TEST(TestBlockLayout, TestBlockLayoutAlignedSpinorBlocks)
{
	IndexArray latdims={{8,8,8,8}};
	IndexArray blockdims={{2,2,2,2}};
	LatticeInfo linfo(latdims);
	StandardAggregation aggr(latdims, blockdims);

	// Make a blocked Layout
	BlockAggregateVectorLayout<float> block_layout(linfo,aggr);

	auto num_blocks = block_layout.GetNumBlocks();
 	auto num_aggr = aggr.GetNumAggregates();
#pragma omp parallel for shared(block_layout,aggr) 
	for(IndexType block=0; block < num_blocks; ++block) {
		for(IndexType cblock=0; cblock < num_aggr; ++cblock) {

			const IndexType blocksite=0;
			const IndexType spin=0;
			const IndexType color=0;
			const IndexType reim=0;

			IndexType block_begin = block_layout.ContainerIndex(block,cblock, blocksite, spin, color, reim);
			IndexType byte_offset = block_begin*sizeof(float);
			EXPECT_EQ(byte_offset % MG_DEFAULT_ALIGNMENT, static_cast<IndexType>(0));
		}
	}
}

TEST(TestBlockLayout, TestBlockSiteStride)
{
	IndexArray latdims={{4,6,6,4}};
	IndexArray blockdims={{2,2,2,2}};
	LatticeInfo linfo(latdims);
	StandardAggregation aggr(latdims, blockdims);

	// Make a blocked Layout
	BlockAggregateVectorLayout<float> block_layout(linfo,aggr);

	// This is the stride for sites. The number of elements
	IndexType site_stride = block_layout.GetNumData()/(aggr.GetNumAggregates()*aggr.GetNumBlocks()
			*aggr.GetSourceSpins(0).size()*aggr.GetSourceColors(0).size());

	IndexType	num_blocks = block_layout.GetNumBlocks();
	IndexType 	num_aggr = aggr.GetNumAggregates();	

#pragma omp parallel for shared(num_blocks, num_aggr) collapse(2)
	for(IndexType block=0; block < num_blocks; ++block) {
		for(IndexType cblock=0; cblock < num_aggr; ++cblock) {


			for(IndexType spin=0; spin  < aggr.GetSourceSpins(0).size(); spin++) {
				for(IndexType color=0; color < aggr.GetSourceColors(0).size()-1; color++) {


					IndexType true_stride=block_layout.ContainerIndex(block,cblock,0,spin,color+1,0)
							-block_layout.ContainerIndex(block,cblock,0,spin,color,0);
					EXPECT_EQ(true_stride,site_stride);
				}
			}
		}
	}
}

TEST(TestBlockLayout, TestCBlockBlockStride)
{
	IndexArray latdims={{4,6,6,4}};
	IndexArray blockdims={{2,2,2,2}};
	LatticeInfo linfo(latdims);
	StandardAggregation aggr(latdims, blockdims);

	// Make a blocked Layout
	BlockAggregateVectorLayout<float> block_layout(linfo,aggr);

	// This is the stride for sites. The number of elements
	IndexType cblock_stride = block_layout.GetNumData()/(aggr.GetNumBlocks()*aggr.GetNumAggregates());

#pragma omp parallel for
	for(IndexType block=0; block < block_layout.GetNumBlocks(); ++block) {
		for(IndexType cblock=0; cblock < aggr.GetNumAggregates()-1; ++cblock) {
					IndexType true_stride=block_layout.ContainerIndex(block,cblock+1,0,0,0,0)
							-block_layout.ContainerIndex(block,cblock,0,0,0,0);
					EXPECT_EQ(true_stride,cblock_stride);


		}
	}
}


TEST(TestBlockLayout, TestCBlockAggregation)
{
	IndexArray latdims={{4,6,6,4}};
	IndexArray blockdims={{2,2,2,2}};
	LatticeInfo linfo(latdims);
	StandardAggregation aggr(latdims, blockdims);

	// Make a blocked Layout
	BlockAggregateVectorLayout<float> block_layout(linfo,aggr);

	ASSERT_EQ(aggr.GetNumBlocks()*aggr.GetNumAggregates()*aggr.GetSourceSpins(0).size()
			*aggr.GetSourceColors(0).size()*aggr.GetBlockVolume(),
			linfo.GetNumSites()*linfo.GetNumColors()*linfo.GetNumSpins());
}

TEST(TestBlockLayout, TestBlockSubview)
{
	IndexArray latdims={{4,6,6,4}};
	IndexArray blockdims={{2,2,2,2}};
	LatticeInfo linfo(latdims);
	StandardAggregation aggr(latdims, blockdims);

//	BlockAggregateVectorLayout<IndexType> block_layout(linfo,aggr);
	LatticeBlockSpinorIndex v_block(linfo,aggr);

	IndexType num_blocks = aggr.GetNumBlocks();
	IndexType num_outer_spins = aggr.GetNumAggregates();

#pragma omp parallel for collapse(2)
	for(IndexType block = 0; block < num_blocks; ++block) {
		for(IndexType cblock=0; cblock < num_outer_spins; ++cblock) {
			auto block_spinor = v_block.GetSubview(block,cblock);
			auto block_info = block_spinor.GetLatticeInfo();

			for(IndexType spin=0; spin < block_info.GetNumSpins(); ++spin) {
				for(IndexType color=0; color < block_info.GetNumColors(); ++color) {
					for(IndexType blocksite =0; blocksite < block_info.GetNumSites(); ++blocksite) {
						for(IndexType reim=0; reim < n_complex; ++reim) {
							block_spinor.Index(blocksite, spin, color, reim) = cblock + num_outer_spins*block;
						}
					}
				}
			}
		}
	}

	auto num_inner_spins = aggr.GetSourceSpins(0).size();
	auto num_inner_colors = aggr.GetSourceColors(0).size();
	auto num_block_vol = aggr.GetBlockVolume();


#pragma omp parallel for collapse(6)
	for(IndexType block = 0; block < num_blocks; ++block) {
		for(IndexType cblock=0; cblock < num_outer_spins; ++cblock) {
			for(IndexType spin=0; spin < num_inner_spins; ++spin) {
				for(IndexType color=0; color < num_inner_colors; ++color) {
					for(IndexType blocksite =0; blocksite < num_block_vol; ++blocksite) {
						for(IndexType reim=0; reim < n_complex; ++reim) {
							EXPECT_EQ( v_block.Index(block,cblock,blocksite,spin,color,reim), cblock + num_outer_spins*block);
						}
					}
				}
			}
		}
	}


}

TEST(TestBlockLayout, TestZipUnzip)
{
	IndexArray latdims={{4,6,6,4}};
	IndexArray blockdims={{2,2,2,2}};
	LatticeInfo linfo(latdims);
	StandardAggregation aggr(latdims, blockdims);

	// Make a non-blocked layout
	CBSOASpinorLayout<IndexType> flat_layout(linfo);
	// Make a blocked Layout
	BlockAggregateVectorLayout<IndexType> block_layout(linfo,aggr);

//	LatticeSpinorIndex v_in(flat_layout);
//	LatticeSpinorIndex v_out(flat_layout);
	//LatticeBlockSpinorIndex v_block(block_layout);

	// New interface... The layout is magic-ed up behind the scenes
	LatticeSpinorIndex v_in(linfo);
	LatticeSpinorIndex v_out(linfo);
	LatticeBlockSpinorIndex v_block(linfo,aggr);

	IndexType num_spins = linfo.GetNumSpins();
	IndexType num_colors = linfo.GetNumColors();
	IndexType num_sites = linfo.GetNumSites();



	// This will need to be converted to some generic fill routine:
#pragma omp parallel for shared(num_spins, num_colors, num_sites) collapse(4)
	for(IndexType spin=0; spin < num_spins;++spin){
		for(IndexType color=0; color < num_colors; ++color) {
			for(IndexType site=0; site < num_sites; ++site)   {
				for(IndexType reim=0; reim < n_complex; ++reim) {
					// Hash the params
					IndexType value = flat_layout.ContainerIndex(site,spin,color,reim);
					v_in.Index(site,spin,color,reim) = value;
				}
			}
		}
	}

	zip<LatticeBlockSpinorIndex,LatticeSpinorIndex>(v_block, v_in);
	unzip<LatticeSpinorIndex,LatticeBlockSpinorIndex>(v_out, v_block);

#pragma omp parallel for shared(num_spins, num_colors, num_sites) collapse(4)
	for(IndexType spin=0; spin < num_spins; ++spin){
		for(IndexType color=0; color < num_colors; ++color) {
			for(IndexType site=0; site < num_sites; ++site)   {
				for(IndexType reim=0; reim < n_complex; ++reim) {


					EXPECT_EQ(v_out.Index(site,spin,color,reim),
							  v_in.Index(site,spin,color,reim));
				}
			}
		}
	}



}

int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

