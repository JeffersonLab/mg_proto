#include "gtest/gtest.h"
#include "lattice/constants.h"
#include "utils/print_utils.h"
#include "utils/memory.h"
#include <vector>

#include "test_env.h"
#include "mock_nodeinfo.h"

#include "lattice/block_operations.h"
#include "lattice/block_blas.h"

#include <omp.h>
using namespace MGGeometry;
using namespace MGUtils;

TEST(TestBlockBLAS, SingleThreadNorm2)
{
	IndexType num_complex=128*1024*4*3;
	IndexType num_float = num_complex*n_complex;
	float* data=reinterpret_cast<float *>(MemoryAllocate(num_float*sizeof(float)));

#pragma omp parallel for
	for(IndexType site=0; site < num_float;++site) {
		data[site] = 0.25;
	}

	// SIMD Loop
	double data_norm=0;
	IndexType num_iters = 500;
	IndexType num_experiment = 10;
	for (IndexType experiment = 0; experiment < num_experiment; experiment++) {
		double start_time = omp_get_wtime();
		for (IndexType iters = 0; iters < num_iters; iters++) {
			Norm2<float>(data_norm, data, num_complex);
		}
		double expected = static_cast<double>(num_float)*0.25*0.25;
		EXPECT_DOUBLE_EQ(data_norm,expected);

		double end_time = omp_get_wtime();
		double time = end_time - start_time;
		MasterLog(INFO, "Time for reduction = %lf", time);
		double bw = num_iters*num_float*sizeof(float)/(1024.0*1024.0*time);
		MasterLog(INFO, "Bandwidth=%lf MB/s", bw);
	}




}

int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

