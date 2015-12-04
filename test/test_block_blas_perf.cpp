#include "gtest/gtest.h"
#include "MG_config.h"
#include "lattice/constants.h"
#include "utils/print_utils.h"
#include "utils/memory.h"
#include <vector>

#include "test_env.h"
#include "mock_nodeinfo.h"

#include "lattice/block_operations.h"
#include <omp.h>
#include <utility>

#include "lattice/contiguous_blas.h"
using namespace MGGeometry;
using namespace MGUtils;

namespace MGGeometry {




};

class TestBLAS : public ::testing::Test, public ::testing::WithParamInterface<std::pair<IndexType,IndexType>> {

};

//using IndexType = MGGeometry::IndexType;

TEST_P(TestBLAS, SingleThreadNormSqSingle)
{
	const std::pair<IndexType, IndexType> param=GetParam();
	IndexType num_complex = param.first;
	IndexType num_iters = param.second;

	IndexType num_float = num_complex*n_complex;
	float* data=reinterpret_cast<float *>(MemoryAllocate(num_float*sizeof(float)));

#pragma omp parallel for
	for(IndexType site=0; site < num_float;++site) {
		data[site] = 0.25;
	}
	double data_norm=0;

	double time=-omp_get_wtime();
	for(IndexType experiment =0; experiment < num_iters; ++experiment) {
		data_norm = NormSq(data, num_float);
	}
	time += omp_get_wtime();
	MasterLog(INFO, "%u items %u iterations: %lf sec  %lf MB/sec",
				num_float, num_iters, time, (double)(num_iters*num_float*sizeof(float))/(1.0e6*time));
	double expected = static_cast<double>(num_float)*0.25*0.25;
	EXPECT_DOUBLE_EQ(data_norm,expected);
	MemoryFree(data);
}

TEST_P(TestBLAS, SingleThreadNormSqDouble)
{
	const std::pair<IndexType, IndexType> param=GetParam();
	IndexType num_complex = param.first;
	IndexType num_iters = param.second;

	IndexType num_float = num_complex*n_complex;
	double* data=reinterpret_cast<double *>(MemoryAllocate(num_float*sizeof(double)));

#pragma omp parallel for
	for(IndexType site=0; site < num_float;++site) {
		data[site] = 0.25;
	}
	double data_norm=0;

	double time=-omp_get_wtime();
	for(IndexType experiment =0; experiment < num_iters; ++experiment) {
		data_norm = NormSq(data, num_float);
	}
	time += omp_get_wtime();
	MasterLog(INFO, "%u items %u iterations: %lf sec  %lf MB/sec",
				num_float, num_iters, time, (double)(num_iters*num_float*sizeof(double))/(1.0e6*time));
	double expected = static_cast<double>(num_float)*0.25*0.25;
	EXPECT_DOUBLE_EQ(data_norm,expected);
	MemoryFree(data);
}

TEST_P(TestBLAS, SingleThreadInnerProdFloat)
{
	const std::pair<IndexType, IndexType> param=GetParam();
	IndexType num_complex = param.first;
	IndexType num_iters = param.second;

	float* data_left=reinterpret_cast<float*>(
			MemoryAllocate(num_complex*n_complex*sizeof(float)));

	float* data_right=reinterpret_cast<float*>(
			MemoryAllocate(num_complex*n_complex*sizeof(float)));

#pragma omp parallel for
	for(IndexType site=0; site < num_complex;++site) {
		data_left[2*site] = 0.25;
		data_left[2*site+1] = 0.25;

		data_right[2*site] = 0.25;
		data_right[2*site+1] = 0.25;
	}
	std::complex<double> data_norm;

	double time=-omp_get_wtime();
	for(IndexType experiment =0; experiment < num_iters; ++experiment) {
		data_norm = InnerProduct( data_left, data_right, num_complex);
	}
	time += omp_get_wtime();
	MasterLog(INFO, "%u items %u iterations: %lf sec  %lf MB/sec",
				num_complex, num_iters, time, (double)(num_iters*2*2*num_complex*sizeof(float))/(1.0e6*time));
	double expected = static_cast<double>(2*num_complex)*0.25*0.25;
	EXPECT_DOUBLE_EQ(data_norm.real(),expected);
	EXPECT_DOUBLE_EQ(data_norm.imag(),0);
	MemoryFree(data_left);
	MemoryFree(data_right);
}

TEST_P(TestBLAS, SingleThreadInnerProdDouble)
{
	const std::pair<IndexType, IndexType> param=GetParam();
	IndexType num_complex = param.first;
	IndexType num_iters = param.second;


	double* data_left=reinterpret_cast<double*>(
			MemoryAllocate(num_complex*n_complex*sizeof(double)));

	double* data_right=reinterpret_cast<double*>(
			MemoryAllocate(num_complex*n_complex*sizeof(double)));

#pragma omp parallel for
	for(IndexType site=0; site < num_complex;++site) {
		data_left[2*site] = 0.25;
		data_left[2*site+1] = 0.25;

		data_right[2*site] = 0.25;
		data_right[2*site+1] = 0.25;
	}
	std::complex<double> data_norm;

	double time=-omp_get_wtime();
	for(IndexType experiment =0; experiment < num_iters; ++experiment) {
		data_norm = InnerProduct( data_left, data_right, num_complex);
	}
	time += omp_get_wtime();
	MasterLog(INFO, "%u items %u iterations: %lf sec  %lf MB/sec",
				num_complex, num_iters, time, (double)(num_iters*2*2*num_complex*sizeof(double))/(1.0e6*time));
	double expected = static_cast<double>(2*num_complex)*0.25*0.25;
	EXPECT_DOUBLE_EQ(data_norm.real(),expected);
	EXPECT_DOUBLE_EQ(data_norm.imag(),0);
	MemoryFree(data_left);
	MemoryFree(data_right);
}
INSTANTIATE_TEST_CASE_P(NormSqTests,
                        TestBLAS,
                        ::testing::Values(
                        			std::make_pair<IndexType,IndexType>(3,10000000),
									std::make_pair<IndexType,IndexType>(6,100000000),
									std::make_pair<IndexType,IndexType>(9,100000000),
									std::make_pair<IndexType,IndexType>(17,80000000),
									std::make_pair<IndexType,IndexType>(129,10000000),
									std::make_pair<IndexType,IndexType>(1025,1000000),
									std::make_pair<IndexType,IndexType>(4097,1000000)));

#if 0
									std::make_pair<IndexType,IndexType>(16385,100000),
									std::make_pair<IndexType,IndexType>(1048577,1000),
									std::make_pair<IndexType,IndexType>(16777217,100),
									std::make_pair<IndexType,IndexType>(67108865,10)
                        )
						);
#endif

int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

