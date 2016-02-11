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

#include "../include/lattice/contiguous_blas.h"
using namespace MG;
using namespace MG;

namespace MG {




};

class TestBLAS : public ::testing::Test, public ::testing::WithParamInterface<IndexType> {

};

//using IndexType = MGGeometry::IndexType;

TEST_P(TestBLAS, SingleThreadNormSqSingle)
{
	IndexType num_complex = GetParam();

	IndexType num_float = num_complex*n_complex;
	float* data=reinterpret_cast<float *>(MemoryAllocate(num_float*sizeof(float)));

#pragma omp parallel for
	for(IndexType site=0; site < num_float;++site) {
		data[site] = 0.25;
	}
	double data_norm = NormSq(data, num_float);

	double expected = static_cast<double>(num_float)*0.25*0.25;
	EXPECT_DOUBLE_EQ(data_norm,expected);
	MemoryFree(data);
}

TEST_P(TestBLAS, SingleThreadNormSqDouble)
{
	IndexType num_complex = GetParam();
	IndexType num_float = num_complex*n_complex;

	double* data=reinterpret_cast<double *>(MemoryAllocate(num_float*sizeof(double)));

#pragma omp parallel for
	for(IndexType site=0; site < num_float;++site) {
		data[site] = 0.25;
	}
	double data_norm= NormSq(data, num_float);
	double expected = static_cast<double>(num_float)*0.25*0.25;
	EXPECT_DOUBLE_EQ(data_norm,expected);
	MemoryFree(data);
}

TEST_P(TestBLAS, SingleThreadInnerProdFloat)
{
	IndexType num_complex = GetParam();

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
	std::complex<double> data_iprod = InnerProduct(data_left, data_right, num_complex);
	double expected = static_cast<double>(2*num_complex)*0.25*0.25;
	EXPECT_DOUBLE_EQ(data_iprod.real(),expected);
	EXPECT_DOUBLE_EQ(data_iprod.imag(),0);
	MemoryFree(data_left);
	MemoryFree(data_right);
}

TEST_P(TestBLAS, SingleThreadInnerProdDouble)
{
	IndexType num_complex = GetParam();


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
	std::complex<double> data_iprod = InnerProduct(data_left, data_right, num_complex);
	double expected = static_cast<double>(2*num_complex)*0.25*0.25;
	EXPECT_DOUBLE_EQ(data_iprod.real(),expected);
	EXPECT_DOUBLE_EQ(data_iprod.imag(),0);
	MemoryFree(data_left);
	MemoryFree(data_right);
}

TEST_P(TestBLAS, SingleThreadVScalSingle)
{
	IndexType num_complex = GetParam();
	float* data = reinterpret_cast<float*>(
			MemoryAllocate(num_complex*n_complex*sizeof(float)));

#pragma omp parallel for
	for(IndexType site=0; site < num_complex;++site) {
		data[2*site] = 0.25;
		data[2*site+1] = -0.3;
	}

	double normsq_data=NormSq(data, num_complex*n_complex);

	float norm=1/sqrt(normsq_data);
	VScale(norm, data, num_complex*n_complex);
	normsq_data = NormSq(data, num_complex*n_complex);

	EXPECT_FLOAT_EQ(static_cast<float>(normsq_data), 1);
	MemoryFree(data);
}

TEST_P(TestBLAS, SingleThreadVScalDouble)
{
	IndexType num_complex = GetParam();
	double* data = reinterpret_cast<double*>(
			MemoryAllocate(num_complex*n_complex*sizeof(double)));

#pragma omp parallel for
	for(IndexType site=0; site < num_complex;++site) {
		data[2*site] = 0.25;
		data[2*site+1] = -0.3;
	}

	double normsq_data = NormSq( data, num_complex*n_complex);

	double norm=1/sqrt(normsq_data);
	VScale(norm, data, num_complex*n_complex);
	normsq_data = NormSq(data, num_complex*n_complex);

	EXPECT_NEAR(normsq_data, 1, 1.0e-10);
	MemoryFree(data);
}

TEST_P(TestBLAS, SingleThreadMCaxpySingle)
{
	IndexType num_complex = GetParam();
	float* x = reinterpret_cast<float*>(
			MemoryAllocate(num_complex*n_complex*sizeof(float)));
	float* y = reinterpret_cast<float*>(
			MemoryAllocate(num_complex*n_complex*sizeof(float)));

	std::complex<float> a = std::complex<float>(-0.3, 2.1);

#pragma omp parallel for
	for(IndexType site=0; site < num_complex;++site) {
		x[2*site] = 0.25;
		x[2*site+1] = -0.3;
		y[2*site] = 0.4;
		y[2*site+1] = -0.43;
	}

	double normsq_y_before=NormSq(y, num_complex*n_complex);

	MCaxpy(y,a,x,num_complex);
	a	 *= -1;
	MCaxpy(y,a,x,num_complex);

	double normsq_y_after=	NormSq(y, num_complex*n_complex);



	EXPECT_FLOAT_EQ((float)normsq_y_before,(float)normsq_y_after);

	MemoryFree(x);
	MemoryFree(y);
}

TEST_P(TestBLAS, SingleThreadMCaxpyDouble)
{
	IndexType num_complex = GetParam();
	double* x = reinterpret_cast<double*>(
			MemoryAllocate(num_complex*n_complex*sizeof(double)));
	double* y = reinterpret_cast<double*>(
			MemoryAllocate(num_complex*n_complex*sizeof(double)));

	std::complex<double> a = std::complex<double>(-0.3, 2.1);

#pragma omp parallel for
	for(IndexType site=0; site < num_complex;++site) {
		x[2*site] = 0.25;
		x[2*site+1] = -0.3;
		y[2*site] = 0.4;
		y[2*site+1] = -0.43;
	}

	double normsq_y_before= NormSq(y, num_complex*n_complex);

	MCaxpy(y,a,x,num_complex);
	a	 *= -1;
	MCaxpy(y,a,x,num_complex);

	double normsq_y_after= NormSq(y, num_complex*n_complex);



	EXPECT_DOUBLE_EQ(normsq_y_before,normsq_y_after);

	MemoryFree(x);
	MemoryFree(y);
}





INSTANTIATE_TEST_CASE_P(NormSqTests,
                        TestBLAS,
                        ::testing::Values(
                        			16,
                        			256,
									4097,
									16385,
									1048577,
									16777217));

int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

