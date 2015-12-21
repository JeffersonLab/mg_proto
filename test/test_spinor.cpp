#include "gtest/gtest.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/basic_spinor.h"
#include "lattice/block_operations.h"
#include "utils/print_utils.h"
#include "utils/memory.h"
#include <vector>

#include "test_env.h"
#include "mock_nodeinfo.h"
#include <cmath>

#include <random>
#include "lattice/virtual_node.h"
#include "lattice/ilattice.h"
#include "lattice/rcomplex.h"

using namespace MGGeometry;
using namespace MGUtils;






TEST(BasicSpinor, Instantiate)
{
	IndexArray latdims={{4,4,4,4}}; // Each cb: 2*4*4*4 = 128 sites => 128 * n_complex * sizeof(float) = 1024 bytes = 16 blocks of length 64 bytes - no padding.
	LatticeInfo fine_info(latdims);
	LatticeGeneralSpinor<float> lattice(fine_info);

}

TEST(BasicSpinor, CopyAssign)
{
	IndexArray latdims={{4,4,4,4}}; // Each cb: 2*4*4*4 = 128 sites => 128 * n_complex * sizeof(float) = 1024 bytes = 16 blocks of length 64 bytes - no padding.
	LatticeInfo fine_info(latdims);
	LatticeGeneralSpinor<float> lattice(fine_info);
	LatticeGeneralSpinor<float> l2(fine_info);

	l2 = lattice;
}


TEST(VNTest, InstantiateVN)
{
	ASSERT_EQ( VN_SSE<float>::n_dim, static_cast<const unsigned int>(2));
	ASSERT_EQ( VN_SSE<float>::n_sites, static_cast<const unsigned int>(4));
	ASSERT_EQ( VN_SSE<float>::mask, static_cast<const unsigned int>(0x3));

	ASSERT_EQ( VN_SSE<double>::n_dim, static_cast<const unsigned int>(1));
	ASSERT_EQ( VN_SSE<double>::n_sites, static_cast<const unsigned int>(2));
	ASSERT_EQ( VN_SSE<double>::mask, static_cast<const unsigned int>(0x1));



}

TEST(VNTest, TestScalarVNode)
{
	ASSERT_EQ( VN_Scalar<float>::n_dim, static_cast<const unsigned int>(0));
	ASSERT_EQ( VN_Scalar<float>::n_sites, static_cast<const unsigned int>(1));
	ASSERT_EQ( VN_Scalar<float>::mask, static_cast<const unsigned int>(0x0));

	ASSERT_EQ( VN_Scalar<double>::n_dim, static_cast<const unsigned int>(0));
	ASSERT_EQ( VN_Scalar<double>::n_sites, static_cast<const unsigned int>(1));
	ASSERT_EQ( VN_Scalar<double>::mask, static_cast<const unsigned int>(0x0));



}

TEST(ILatticeTest, ILatticeFloatTraits)
{
	// Check that ILattice behaves as expected
	IndexType i=SizeTraits< ILattice<float,VN_SSE<float>> >::IndexSpaceSize();
	ASSERT_EQ( i, static_cast<IndexType>(VN_SSE<float>::n_sites));

	ASSERT_EQ( i*sizeof(float),
			    static_cast<size_t>(VN_SSE<float>::n_sites)*sizeof(float));

	ILattice<float, VN_SSE<float>> il_sse_f;

	ASSERT_EQ( il_sse_f.size(), static_cast<IndexType>(VN_SSE<float>::n_sites));

	for(IndexType s=0; s < VN_SSE<float>::n_sites; ++s) {
		ASSERT_EQ( il_sse_f.OffsetIndex(s), static_cast<IndexType>(s));
	}

}

TEST(ILatticeTest, RComplexFloatTraits)
{
	// Check that ILattice behaves as expected
	ASSERT_EQ( SizeTraits< RComplex<float> >::IndexSpaceSize(), n_complex);
	ASSERT_EQ( SizeTraits< RComplex<float> >::MemorySize(), n_complex*sizeof(float));
	RComplex<float> c;
	ASSERT_EQ( c.RealIndex(), static_cast<IndexType>(0));
	ASSERT_EQ( c.ImagIndex(), static_cast<IndexType>(1));


}

TEST(ILatticeTest, RComplexILatticeTraits)
{
	// Check that ILattice behaves as expected
	using IL = ILattice<float, VN_SSE<float>>;
	ASSERT_EQ( SizeTraits< RComplex< IL > >::IndexSpaceSize(), n_complex*VN_SSE<float>::n_sites);
	ASSERT_EQ( SizeTraits< RComplex< IL > >::MemorySize(),
			static_cast<size_t>(n_complex*VN_SSE<float>::n_sites*sizeof(float)));
	RComplex<IL> c;
	ASSERT_EQ( c.RealIndex(), static_cast<IndexType>(0));
	ASSERT_EQ( c.ImagIndex(), static_cast<IndexType>(VN_SSE<float>::n_sites));
}


#if 0
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
			using BlockSubSpinorType = typename ContainerTraits<T,BlockedLayout,GenericLayoutContainer<T,BlockedLayout>>::subview_container_type;

			// A vector to hold the sub-spinors
			std::vector<BlockSubSpinorType> block_spinors(num_vectors);
			for(IndexType v=0; v < num_vectors; ++v) {

				block_spinors[v] = vectors[v].GetSubview(block, outer_spin);

			}

			GramSchmidt(block_spinors);

		}
	}
}

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

#endif


int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

