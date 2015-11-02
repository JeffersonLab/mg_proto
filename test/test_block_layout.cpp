#include "gtest/gtest.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/lattice_spinor.h"
#include "utils/print_utils.h"
#include "utils/memory.h"
#include <vector>

#include "test_env.h"
#include "mock_nodeinfo.h"
#include "lattice/block_cb_soa_spinor_layout.h"
#include "lattice/block_operations.h"

using namespace MGGeometry;
using namespace MGUtils;

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

#pragma omp parallel for collapse(2)
	for(IndexType block=0; block < block_layout.GetNumBlocks(); ++block) {
		for(IndexType cblock=0; cblock < aggr.GetNumAggregates(); ++cblock) {

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
	IndexType site_stride = block_layout.DataNumElem()/(aggr.GetNumAggregates()*aggr.GetNumBlocks()
			*aggr.GetSourceSpins(0).size()*aggr.GetSourceColors(0).size());

#pragma omp parallel for collapse(2)
	for(IndexType block=0; block < block_layout.GetNumBlocks(); ++block) {
		for(IndexType cblock=0; cblock < aggr.GetNumAggregates(); ++cblock) {


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
	IndexType cblock_stride = block_layout.DataNumElem()/(aggr.GetNumBlocks()*aggr.GetNumAggregates());

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

	LatticeSpinorIndex v_in(flat_layout);
	LatticeSpinorIndex v_out(flat_layout);
	LatticeBlockSpinorIndex v_block(block_layout);

	// This will need to be converted to some generic fill routine:
#pragma omp parallel for collapse(4)
	for(IndexType spin=0; spin < linfo.GetNumSpins();++spin){
		for(IndexType color=0; color < linfo.GetNumColors(); ++color) {
			for(IndexType site=0; site < linfo.GetNumSites(); ++site)   {
				for(IndexType reim=0; reim < n_complex; ++reim) {
					// Hash the params
					IndexType value = reim+n_complex*(site+linfo.GetNumSites()*(color+linfo.GetNumColors()*spin));


					v_in.Index(site,spin,color,reim) = value;
				}
			}
		}
	}

	zip<LatticeBlockSpinorIndex,LatticeSpinorIndex>(v_block, v_in);
	unzip<LatticeSpinorIndex,LatticeBlockSpinorIndex>(v_out, v_block);

#pragma omp parallel for collapse(4)
	for(IndexType spin=0; spin < linfo.GetNumSpins();++spin){
		for(IndexType color=0; color < linfo.GetNumColors(); ++color) {
			for(IndexType site=0; site < linfo.GetNumSites(); ++site)   {
				for(IndexType reim=0; reim < n_complex; ++reim) {

					EXPECT_EQ(v_out.Index(site,spin,color,reim),
							  v_in.Index(site,spin,color,reim));
				}
			}
		}
	}



}

TEST(TestBlockLayout, TestZipUnzipArray)
{
	IndexArray latdims={{4,6,6,4}};
	IndexArray blockdims={{2,2,2,2}};
	IndexType n_vec = 24;

	LatticeInfo linfo(latdims);
	StandardAggregation aggr(latdims, blockdims);

	// Make a non-blocked layout
	CBSOASpinorLayout<IndexType> flat_layout(linfo);


	// Make a blocked Layout
	BlockAggregateVectorArrayLayout<IndexType> block_layout(linfo,aggr,n_vec);

	LatticeSpinorIndex v_in(flat_layout);
	LatticeSpinorIndex v_out(flat_layout);
	LatticeBlockSpinorArrayIndex v_block(block_layout);

	// This will need to be converted to some generic fill routine:
	for(IndexType vec=0; vec < n_vec; ++vec) {

#pragma omp parallel for collapse(4)
		for(IndexType spin=0; spin < linfo.GetNumSpins();++spin){
			for(IndexType color=0; color < linfo.GetNumColors(); ++color) {
				for(IndexType site=0; site < linfo.GetNumSites(); ++site)   {
					for(IndexType reim=0; reim < n_complex; ++reim) {
					// Hash the params
						IndexType value = reim+n_complex*(site+linfo.GetNumSites()*(color+linfo.GetNumColors()*(spin + linfo.GetNumSpins()*vec)));


						v_in.Index(site,spin,color,reim) = value;
					}
				}
			}
		}

		zip<LatticeBlockSpinorArrayIndex,LatticeSpinorIndex>(v_block, v_in, vec);

		unzip<LatticeSpinorIndex,LatticeBlockSpinorArrayIndex>(v_out, v_block,vec);

#pragma omp parallel for collapse(4)
		for(IndexType spin=0; spin < linfo.GetNumSpins();++spin){
			for(IndexType color=0; color < linfo.GetNumColors(); ++color) {
				for(IndexType site=0; site < linfo.GetNumSites(); ++site)   {
					for(IndexType reim=0; reim < n_complex; ++reim) {

						EXPECT_EQ(v_out.Index(site,spin,color,reim),
								   v_in.Index(site,spin,color,reim));
					}
				}
			}
		}

	} // Vec Loop


}

int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

