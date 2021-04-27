/*
 * test_nodeinfo.cpp
 *
 *  Created on: Sep 25, 2015
 *      Author: bjoo
 */


#include "gtest/gtest.h"
#include "lattice/cmat_mult.h"
#include "utils/memory.h"
#include "utils/print_utils.h"
#include <random>
#include "MG_config.h"
#include "test_env.h"

#include <omp.h>
#include <cstdio>

#include "lattice/coarse/coarse_types.h"

using namespace MG;
using namespace MG;

#define N_SMT 1

#define COMPLEX_ADD_FLOPS 2
#define COMPLEX_MULT_FLOPS 6

// Eigen Dense header -- for testing

#include <Eigen/Dense>
using namespace Eigen;

#include <cstdlib>
// This is our test fixture. No tear down or setup
class CMatMultTime : public ::testing::TestWithParam<int> {
protected:
	void SetUp() override {
		const int N = GetParam();
		MG::MasterLog(MG::INFO, "Setting up with N=%d", GetParam());

		x = static_cast<float*>(MG::MemoryAllocate(2*N*sizeof(float)));
		y = static_cast<float*>(MG::MemoryAllocate(2*N*sizeof(float)));
		y2 = static_cast<float*>(MG::MemoryAllocate(2*N*sizeof(float)));
		tmp = static_cast<float*>(MG::MemoryAllocate(2*N*sizeof(float)));
		A = static_cast<float*>(MG::MemoryAllocate(2*N*N*sizeof(float)));

		for(int i=0; i < 2*N; ++i) {
			MG::MasterLog(MG::DEBUG2, "i=%d",i);
			x[i] = (float)drand48();
			x[i] = (float)drand48();
			y[i] = (float)drand48();
			y2[i] =y[i];
		}

		MG::MasterLog(MG::DEBUG2, "Filling Matrices");
		for(IndexType row=0; row < N; ++row) {
			for(IndexType col=0; col < N; ++col) {
				for(int z =0; z < 2; ++z) {
					A[z + 2*(row + N*col)] =  (float)drand48();
				}
			}
		}

		MG::MasterLog(MG::DEBUG2, "Done");
	}

	void TearDown() override {
		MG::MemoryFree(x);
		MG::MemoryFree(y);
		MG::MemoryFree(y2);
		MG::MemoryFree(tmp);
		MG::MemoryFree(A);
	}

	float *x;
	float *y;
	float*y2;
	float *tmp;
	float *A;

	// Define Eigen Complex Matrix Class
	using ComplexMatrixDef = Matrix<std::complex<float>, Dynamic, Dynamic, ColMajor>;
	using ComplexVectorDef = Matrix<std::complex<float>, Dynamic, 1>;
	using ComplexMatrix = Map< ComplexMatrixDef >;
	using ComplexVector = Map< ComplexVectorDef >;
};

/* ------- TESTS START HERE --------- */


TEST_P(CMatMultTime, TimeCMatMult)
{
	const int N = GetParam();
	MG::MasterLog(INFO, "Timing N=%d", N);

	MG::MasterLog(INFO, "Calibrating");
	double time=0;
	int iters = 1;
	do {
		iters *= 2;
		time -= omp_get_wtime();
		for(int j=0; j < iters; ++j) {
			CMatMultNaive(y,A,x,N );
		}
		time += omp_get_wtime();
	}
	while( time < 1.0);
	MG::MasterLog(INFO, "Warming up");
	for(int j=0; j < iters; ++j) {
		CMatMultNaive(y,A,x,N );
	}
	MG::MasterLog(INFO, "Timing: %d iters", iters);
	time -= omp_get_wtime();
	for(int j=0; j < iters; ++j) {
			CMatMultNaive(y,A,x,N );
	}
	time += omp_get_wtime();

	// Iters * rows * ( colunm * cmult + (column-1)*cmadd
	double iterflops = static_cast<double>(N*(  N*COMPLEX_MULT_FLOPS + (N-1)*COMPLEX_ADD_FLOPS ) );
	double flops =static_cast<double>( iters )*iterflops;

	double total_gflops = (flops/time)*1.0e-9;

	double iter_read_bytes = static_cast<double>( ( N*N  + N )*2*sizeof(float));
	double iter_write_bytes = static_cast<double>( N*2*sizeof(float));

	double total_read_bytes = static_cast<double>(iters*iter_read_bytes);
	double total_write_bytes = static_cast<double>(iters*iter_write_bytes);
	double total_bytes = total_read_bytes + total_write_bytes;

	double gigi = 1024.0*1024.0*1024.0;

	double read_bw = (total_read_bytes / time)/gigi;
	double write_bw = (total_write_bytes / time)/gigi;
	double total_bw = (total_bytes)/gigi;

	MasterLog(INFO, "Naive CMatMult Time=%16.8e (sec)  GFLOPS=%16.8e   Read BW=%16.8e  GiB/sec  Write BW=%16.8e GiB/sec  Total BW=%16.8e GiB/sec",
			time, total_gflops, read_bw, write_bw, total_bw);

}

TEST_P(CMatMultTime, TimeGcCMatMultGc)
{
	const int N = GetParam();
	MG::MasterLog(INFO, "Timing N=%d", N);

	MG::MasterLog(INFO, "Calibrating");
	double time=0;
	int iters = 1;
	do {
		iters *= 2;
		time -= omp_get_wtime();
		for(int j=0; j < iters; ++j) {
			GcCMatMultGcNaive(y,A,x,N );
		}
		time += omp_get_wtime();
	}
	while( time < 1.0);
	MG::MasterLog(INFO, "Warming up");
	for(int j=0; j < iters; ++j) {
		GcCMatMultGcNaive(y,A,x,N );
	}
	MG::MasterLog(INFO, "Timing: %d iters", iters);
	time -= omp_get_wtime();
	for(int j=0; j < iters; ++j) {
			GcCMatMultGcNaive(y,A,x,N );
	}
	time += omp_get_wtime();

	// Iters * rows * ( colunm * cmult + (column-1)*cmadd -- sign flips don't count as flops
	double iterflops = static_cast<double>(N*(  N*COMPLEX_MULT_FLOPS + (N-1)*COMPLEX_ADD_FLOPS ) );
	double flops =static_cast<double>( iters )*iterflops;

	double total_gflops = (flops/time)*1.0e-9;

	double iter_read_bytes = static_cast<double>( ( N*N  + N )*2*sizeof(float));
	double iter_write_bytes = static_cast<double>( N*2*sizeof(float));

	double total_read_bytes = static_cast<double>(iters*iter_read_bytes);
	double total_write_bytes = static_cast<double>(iters*iter_write_bytes);
	double total_bytes = total_read_bytes + total_write_bytes;

	double gigi = 1024.0*1024.0*1024.0;

	double read_bw = (total_read_bytes / time)/gigi;
	double write_bw = (total_write_bytes / time)/gigi;
	double total_bw = (total_bytes)/gigi;

	MasterLog(INFO, "Naive GcCMatMultGc Time=%16.8e (sec)  GFLOPS=%16.8e   Read BW=%16.8e  GiB/sec  Write BW=%16.8e GiB/sec  Total BW=%16.8e GiB/sec",
			time, total_gflops, read_bw, write_bw, total_bw);

}
TEST_P(CMatMultTime, TimeCMatMultAdd)
{
	const int N = GetParam();
	const float alpha =-0.2456;

	MG::MasterLog(INFO, "Timing N=%d", N);

	MG::MasterLog(INFO, "Calibrating");
	double time=0;
	int iters = 1;
	do {
		iters *= 2;
		time -= omp_get_wtime();
		for(int j=0; j < iters; ++j) {
			CMatMultAddNaive(y,A,x,N );
		}
		time += omp_get_wtime();
	}
	while( time < 1.0);
	MG::MasterLog(INFO, "Warming up");
	for(int j=0; j < iters; ++j) {
		CMatMultAddNaive(y,A,x,N );
	}
	MG::MasterLog(INFO, "Timing: %d iters", iters);
	time -= omp_get_wtime();
	for(int j=0; j < iters; ++j) {
			CMatMultAddNaive(y,A,x,N );
	}
	time += omp_get_wtime();

	// Iters * rows * ( colunm * cmult + (column-1)*cmadd +  2 adds )
	double iterflops = static_cast<double>(N*(  N*COMPLEX_MULT_FLOPS + (N-1)*COMPLEX_ADD_FLOPS + 2  ) );
	double flops =static_cast<double>( iters )*iterflops;

	double total_gflops = (flops/time)*1.0e-9;

	// N * N is the matrix 2*N is input and output (read for output) and next 2* is for complex
	double iter_read_bytes = static_cast<double>( ( N*N  + 2*N )*2*sizeof(float));
	double iter_write_bytes = static_cast<double>( N*2*sizeof(float));

	double total_read_bytes = static_cast<double>(iters*iter_read_bytes);
	double total_write_bytes = static_cast<double>(iters*iter_write_bytes);
	double total_bytes = total_read_bytes + total_write_bytes;

	double gigi = 1024.0*1024.0*1024.0;

	double read_bw = (total_read_bytes / time)/gigi;
	double write_bw = (total_write_bytes / time)/gigi;
	double total_bw = (total_bytes/time)/gigi;

	MasterLog(INFO, "Naive CMatMultAdd Time=%16.8e (sec)  GFLOPS=%16.8e   Read BW=%16.8e  GiB/sec  Write BW=%16.8e GiB/sec  Total BW=%16.8e GiB/sec",
			time, total_gflops, read_bw, write_bw, total_bw);

}

TEST_P(CMatMultTime, TimeCMatMultCoeffAdd)
{
	const int N = GetParam();
	const float alpha =-0.2456;

	MG::MasterLog(INFO, "Timing N=%d", N);

	MG::MasterLog(INFO, "Calibrating");
	double time=0;
	int iters = 1;
	do {
		iters *= 2;
		time -= omp_get_wtime();
		for(int j=0; j < iters; ++j) {
			CMatMultCoeffAddNaive(1.0,y,alpha,A,x,N );
		}
		time += omp_get_wtime();
	}
	while( time < 1.0);
	MG::MasterLog(INFO, "Warming up");
	for(int j=0; j < iters; ++j) {
		CMatMultCoeffAddNaive(1.0,y,alpha,A,x,N );
	}
	MG::MasterLog(INFO, "Timing: %d iters", iters);
	time -= omp_get_wtime();
	for(int j=0; j < iters; ++j) {
			CMatMultCoeffAddNaive(1.0,y,alpha,A,x,N );
	}
	time += omp_get_wtime();

	// Iters * rows * ( colunm * cmult + (column-1)*cmadd + 2 multiplies by alpha + 2 adds )
	double iterflops = static_cast<double>(N*(  N*COMPLEX_MULT_FLOPS + (N-1)*COMPLEX_ADD_FLOPS + 2 + 2 ) );
	double flops =static_cast<double>( iters )*iterflops;

	double total_gflops = (flops/time)*1.0e-9;

	// N * N is the matrix 2*N is input and output (read for output) and next 2* is for complex
	double iter_read_bytes = static_cast<double>( ( N*N  + 2*N )*2*sizeof(float));
	double iter_write_bytes = static_cast<double>( N*2*sizeof(float));

	double total_read_bytes = static_cast<double>(iters*iter_read_bytes);
	double total_write_bytes = static_cast<double>(iters*iter_write_bytes);
	double total_bytes = total_read_bytes + total_write_bytes;

	double gigi = 1024.0*1024.0*1024.0;

	double read_bw = (total_read_bytes / time)/gigi;
	double write_bw = (total_write_bytes / time)/gigi;
	double total_bw = (total_bytes/time)/gigi;

	MasterLog(INFO, "Naive CMatMultCoeffAdd Time=%16.8e (sec)  GFLOPS=%16.8e   Read BW=%16.8e  GiB/sec  Write BW=%16.8e GiB/sec  Total BW=%16.8e GiB/sec",
			time, total_gflops, read_bw, write_bw, total_bw);

}

TEST_P(CMatMultTime, TimeGcCMatMultGcCoeffAdd)
{
	const int N = GetParam();
	const float alpha =-0.2456;

	MG::MasterLog(INFO, "Timing N=%d", N);

	MG::MasterLog(INFO, "Calibrating");
	double time=0;
	int iters = 1;
	do {
		iters *= 2;
		time -= omp_get_wtime();
		for(int j=0; j < iters; ++j) {
			GcCMatMultGcCoeffAddNaive(1.0,y,alpha,A,x,N );
		}
		time += omp_get_wtime();
	}
	while( time < 1.0);
	MG::MasterLog(INFO, "Warming up");
	for(int j=0; j < iters; ++j) {
		GcCMatMultGcCoeffAddNaive(1.0,y,alpha,A,x,N );
	}
	MG::MasterLog(INFO, "Timing: %d iters", iters);
	time -= omp_get_wtime();
	for(int j=0; j < iters; ++j) {
			GcCMatMultGcCoeffAddNaive(1.0,y,alpha,A,x,N );
	}
	time += omp_get_wtime();

	// Iters * rows * ( colunm * cmult + (column-1)*cmadd + 2 multiplies by alpha + 2 adds )
	double iterflops = static_cast<double>(N*(  N*COMPLEX_MULT_FLOPS + (N-1)*COMPLEX_ADD_FLOPS + 2 + 2 ) );
	double flops =static_cast<double>( iters )*iterflops;

	double total_gflops = (flops/time)*1.0e-9;

	// N * N is the matrix 2*N is input and output (read for output) and next 2* is for complex
	double iter_read_bytes = static_cast<double>( ( N*N  + 2*N )*2*sizeof(float));
	double iter_write_bytes = static_cast<double>( N*2*sizeof(float));

	double total_read_bytes = static_cast<double>(iters*iter_read_bytes);
	double total_write_bytes = static_cast<double>(iters*iter_write_bytes);
	double total_bytes = total_read_bytes + total_write_bytes;

	double gigi = 1024.0*1024.0*1024.0;

	double read_bw = (total_read_bytes / time)/gigi;
	double write_bw = (total_write_bytes / time)/gigi;
	double total_bw = (total_bytes/time)/gigi;

	MasterLog(INFO, "Naive  GcCMatMultGcCoeffAdd Time=%16.8e (sec)  GFLOPS=%16.8e   Read BW=%16.8e  GiB/sec  Write BW=%16.8e GiB/sec  Total BW=%16.8e GiB/sec",
			time, total_gflops, read_bw, write_bw, total_bw);

}
TEST_P(CMatMultTime, TimeCMatMultEigen)
{
	const int N = GetParam();
	MG::MasterLog(INFO, "Timing Eigen CMat Mult N=%d", N);

	ComplexMatrix  in_mat(reinterpret_cast<std::complex<float>*>(A),N,N);
	ComplexVector  eigen_x(reinterpret_cast<std::complex<float>*>(x),N);
	ComplexVector  eigen_out(reinterpret_cast<std::complex<float>*>(y2),N);

	MG::MasterLog(INFO, "Calibrating");
	double time=0;
	int iters = 1;
	do {
		iters *= 2;
		time -= omp_get_wtime();
		for(int j=0; j < iters; ++j) {
			eigen_out = in_mat*eigen_x;
		}
		time += omp_get_wtime();
	}
	while( time < 1.0);
	MG::MasterLog(INFO, "Warming up");
	for(int j=0; j < iters; ++j) {
		eigen_out = in_mat*eigen_x;
	}
	MG::MasterLog(INFO, "Timing: %d iters", iters);
	time -= omp_get_wtime();
	for(int j=0; j < iters; ++j) {
		eigen_out = in_mat*eigen_x;
	}
	time += omp_get_wtime();

	// Iters * rows * ( colunm * cmult + (column-1)*cmadd
	double iterflops = static_cast<double>(N*(  N*COMPLEX_MULT_FLOPS + (N-1)*COMPLEX_ADD_FLOPS ) );
	double flops =static_cast<double>( iters )*iterflops;

	double total_gflops = (flops/time)*1.0e-9;

	double iter_read_bytes = static_cast<double>( ( N*N  + N )*2*sizeof(float));
	double iter_write_bytes = static_cast<double>( N*2*sizeof(float));

	double total_read_bytes = static_cast<double>(iters*iter_read_bytes);
	double total_write_bytes = static_cast<double>(iters*iter_write_bytes);
	double total_bytes = total_read_bytes + total_write_bytes;

	double gigi = 1024.0*1024.0*1024.0;

	double read_bw = (total_read_bytes / time)/gigi;
	double write_bw = (total_write_bytes / time)/gigi;
	double total_bw = (total_bytes)/gigi;

	MasterLog(INFO, "Eigen CMatMult Time=%16.8e (sec)  GFLOPS=%16.8e   Read BW=%16.8e  GiB/sec  Write BW=%16.8e GiB/sec  Total BW=%16.8e GiB/sec",
			time, total_gflops, read_bw, write_bw, total_bw);

}


#ifdef MG_USE_AVX512
class CMatMultTimeAVX512 : public CMatMultTime {};

TEST_P(CMatMultTimeAVX512, TimeCMatMult)
{
	const int N = GetParam();
	MG::MasterLog(INFO, "Timing N=%d", N);

	MG::MasterLog(INFO, "Calibrating");
	double time=0;
	int iters = 1;
	do {
		iters *= 2;
		time -= omp_get_wtime();
		for(int j=0; j < iters; ++j) {
			CMatMultAVX512(y,A,x,N );
		}
		time += omp_get_wtime();
	}
	while( time < 1.0);
	MG::MasterLog(INFO, "Warming up");
	for(int j=0; j < iters; ++j) {
		CMatMultAVX512(y,A,x,N );
	}
	MG::MasterLog(INFO, "Timing: %d iters", iters);
	time -= omp_get_wtime();
	for(int j=0; j < iters; ++j) {
			CMatMultAVX512(y,A,x,N );
	}
	time += omp_get_wtime();

	// Iters * rows * ( colunm * cmult + (column-1)*cmadd
	double iterflops = static_cast<double>(N*(  N*COMPLEX_MULT_FLOPS + (N-1)*COMPLEX_ADD_FLOPS ) );
	double flops =static_cast<double>( iters )*iterflops;

	double total_gflops = (flops/time)*1.0e-9;

	double iter_read_bytes = static_cast<double>( ( N*N  + N )*2*sizeof(float));
	double iter_write_bytes = static_cast<double>( N*2*sizeof(float));

	double total_read_bytes = static_cast<double>(iters*iter_read_bytes);
	double total_write_bytes = static_cast<double>(iters*iter_write_bytes);
	double total_bytes = total_read_bytes + total_write_bytes;

	double gigi = 1024.0*1024.0*1024.0;

	double read_bw = (total_read_bytes / time)/gigi;
	double write_bw = (total_write_bytes / time)/gigi;
	double total_bw = (total_bytes/time)/gigi;

	MasterLog(INFO, "AVX12 CMatMult Time=%16.8e (sec)  GFLOPS=%16.8e   Read BW=%16.8e  GiB/sec  Write BW=%16.8e GiB/sec  Total BW=%16.8e GiB/sec",
			time, total_gflops, read_bw, write_bw, total_bw);

}


TEST_P(CMatMultTimeAVX512, TimeCMatMultAdd)
{
	const int N = GetParam();
	MG::MasterLog(INFO, "Timing N=%d", N);

	MG::MasterLog(INFO, "Calibrating");
	double time=0;
	int iters = 1;
	do {
		iters *= 2;
		time -= omp_get_wtime();
		for(int j=0; j < iters; ++j) {
			CMatMultAddAVX512(y,A,x,N );
		}
		time += omp_get_wtime();
	}
	while( time < 1.0);
	MG::MasterLog(INFO, "Warming up");
	for(int j=0; j < iters; ++j) {
		CMatMultAddAVX512(y,A,x,N );
	}
	MG::MasterLog(INFO, "Timing: %d iters", iters);
	time -= omp_get_wtime();
	for(int j=0; j < iters; ++j) {
			CMatMultAddAVX512(y,A,x,N );
	}
	time += omp_get_wtime();

	// Iters * rows * ( colunm * cmult + (column-1)*cmadd + 2 adds to result )
	double iterflops = static_cast<double>(N*(  N*COMPLEX_MULT_FLOPS + (N-1)*COMPLEX_ADD_FLOPS + 2 ) );
	double flops =static_cast<double>( iters )*iterflops;

	double total_gflops = (flops/time)*1.0e-9;

	double iter_read_bytes = static_cast<double>( ( N*N  + 2*N )*2*sizeof(float));
	double iter_write_bytes = static_cast<double>( N*2*sizeof(float));

	double total_read_bytes = static_cast<double>(iters*iter_read_bytes);
	double total_write_bytes = static_cast<double>(iters*iter_write_bytes);
	double total_bytes = total_read_bytes + total_write_bytes;

	double gigi = 1024.0*1024.0*1024.0;

	double read_bw = (total_read_bytes / time)/gigi;
	double write_bw = (total_write_bytes / time)/gigi;
	double total_bw = (total_bytes/time)/gigi;

	MasterLog(INFO, "AVX12 CMatMultAdd Time=%16.8e (sec)  GFLOPS=%16.8e   Read BW=%16.8e  GiB/sec  Write BW=%16.8e GiB/sec  Total BW=%16.8e GiB/sec",
			time, total_gflops, read_bw, write_bw, total_bw);

}

TEST_P(CMatMultTimeAVX512, TimeCMatMultCoeffAdd)
{
	const int N = GetParam();
	const float alpha =-0.2456;

	MG::MasterLog(INFO, "Timing N=%d", N);

	MG::MasterLog(INFO, "Calibrating");
	double time=0;
	int iters = 1;
	do {
		iters *= 2;
		time -= omp_get_wtime();
		for(int j=0; j < iters; ++j) {
			CMatMultCoeffAddAVX512(y,alpha,A,x,N );
		}
		time += omp_get_wtime();
	}
	while( time < 1.0);
	MG::MasterLog(INFO, "Warming up");
	for(int j=0; j < iters; ++j) {
		CMatMultCoeffAddAVX512(y,alpha,A,x,N );
	}
	MG::MasterLog(INFO, "Timing: %d iters", iters);
	time -= omp_get_wtime();
	for(int j=0; j < iters; ++j) {
			CMatMultCoeffAddAVX512(y,alpha,A,x,N );
	}
	time += omp_get_wtime();

	// Iters * rows * ( colunm * cmult + (column-1)*cmadd + 2 multiplies by alpha + 2 adds )
	double iterflops = static_cast<double>(N*(  N*COMPLEX_MULT_FLOPS + (N-1)*COMPLEX_ADD_FLOPS + 2 + 2 ) );
	double flops =static_cast<double>( iters )*iterflops;

	double total_gflops = (flops/time)*1.0e-9;

	// N * N is the matrix 2*N is input and output (read for output) and next 2* is for complex
	double iter_read_bytes = static_cast<double>( ( N*N  + 2*N )*2*sizeof(float));
	double iter_write_bytes = static_cast<double>( N*2*sizeof(float));

	double total_read_bytes = static_cast<double>(iters*iter_read_bytes);
	double total_write_bytes = static_cast<double>(iters*iter_write_bytes);
	double total_bytes = total_read_bytes + total_write_bytes;

	double gigi = 1024.0*1024.0*1024.0;

	double read_bw = (total_read_bytes / time)/gigi;
	double write_bw = (total_write_bytes / time)/gigi;
	double total_bw = (total_bytes/time)/gigi;

	MasterLog(INFO, "AVX12 CMatMultCoeffAdd Time=%16.8e (sec)  GFLOPS=%16.8e   Read BW=%16.8e  GiB/sec  Write BW=%16.8e GiB/sec  Total BW=%16.8e GiB/sec",
			time, total_gflops, read_bw, write_bw, total_bw);

}

TEST_P(CMatMultTimeAVX512, TimeCMatAdjMult)
{
	const int N = GetParam();
	const float alpha =-0.2456;

	MG::MasterLog(INFO, "Timing N=%d", N);

	MG::MasterLog(INFO, "Calibrating");
	double time=0;
	int iters = 1;
	do {
		iters *= 2;
		time -= omp_get_wtime();
		for(int j=0; j < iters; ++j) {
			CMatAdjMultAVX512(y,A,x,N );
		}
		time += omp_get_wtime();
	}
	while( time < 1.0);
	MG::MasterLog(INFO, "Warming up");
	for(int j=0; j < iters; ++j) {
		CMatAdjMultAVX512(y,A,x,N );
	}
	MG::MasterLog(INFO, "Timing: %d iters", iters);
	time -= omp_get_wtime();
	for(int j=0; j < iters; ++j) {
			CMatAdjMultAVX512(y,A,x,N );
	}
	time += omp_get_wtime();

	// Iters * rows * ( colunm * cmult + (column-1)*cmadd + 2 multiplies by alpha + 2 adds )
	double iterflops = static_cast<double>(N*(  N*COMPLEX_MULT_FLOPS + (N-1)*COMPLEX_ADD_FLOPS ) );
	double flops =static_cast<double>( iters )*iterflops;

	double total_gflops = (flops/time)*1.0e-9;

	// N * N is the matrix 2*N is input and output (read for output) and next 2* is for complex
	double iter_read_bytes = static_cast<double>( ( N*N  + 2*N )*2*sizeof(float));
	double iter_write_bytes = static_cast<double>( N*2*sizeof(float));

	double total_read_bytes = static_cast<double>(iters*iter_read_bytes);
	double total_write_bytes = static_cast<double>(iters*iter_write_bytes);
	double total_bytes = total_read_bytes + total_write_bytes;

	double gigi = 1024.0*1024.0*1024.0;

	double read_bw = (total_read_bytes / time)/gigi;
	double write_bw = (total_write_bytes / time)/gigi;
	double total_bw = (total_bytes/time)/gigi;

	MasterLog(INFO, "AVX12 CMatAdjMult Time=%16.8e (sec)  GFLOPS=%16.8e   Read BW=%16.8e  GiB/sec  Write BW=%16.8e GiB/sec  Total BW=%16.8e GiB/sec",
			time, total_gflops, read_bw, write_bw, total_bw);

}

TEST_P(CMatMultTimeAVX512, TimeGcCMatMultGc)
{
	const int N = GetParam();
	const float alpha =-0.2456;

	MG::MasterLog(INFO, "Timing N=%d", N);

	MG::MasterLog(INFO, "Calibrating");
	double time=0;
	int iters = 1;
	do {
		iters *= 2;
		time -= omp_get_wtime();
		for(int j=0; j < iters; ++j) {
			GcCMatMultGcAVX512(y,A,x,N );
		}
		time += omp_get_wtime();
	}
	while( time < 1.0);
	MG::MasterLog(INFO, "Warming up");
	for(int j=0; j < iters; ++j) {
		GcCMatMultGcAVX512(y,A,x,N );
	}
	MG::MasterLog(INFO, "Timing: %d iters", iters);
	time -= omp_get_wtime();
	for(int j=0; j < iters; ++j) {
			GcCMatMultGcAVX512(y,A,x,N );
	}
	time += omp_get_wtime();

	// Iters * rows * ( colunm * cmult + (column-1)*cmadd )
	double iterflops = static_cast<double>(N*(  N*COMPLEX_MULT_FLOPS + (N-1)*COMPLEX_ADD_FLOPS) );
	double flops =static_cast<double>( iters )*iterflops;

	double total_gflops = (flops/time)*1.0e-9;

	// N * N is the matrix 2*N is input and output (read for output) and next 2* is for complex
	double iter_read_bytes = static_cast<double>( ( N*N  + 2*N )*2*sizeof(float));
	double iter_write_bytes = static_cast<double>( N*2*sizeof(float));

	double total_read_bytes = static_cast<double>(iters*iter_read_bytes);
	double total_write_bytes = static_cast<double>(iters*iter_write_bytes);
	double total_bytes = total_read_bytes + total_write_bytes;

	double gigi = 1024.0*1024.0*1024.0;

	double read_bw = (total_read_bytes / time)/gigi;
	double write_bw = (total_write_bytes / time)/gigi;
	double total_bw = (total_bytes/time)/gigi;

	MasterLog(INFO, "AVX512  GcCMatMultGc Time=%16.8e (sec)  GFLOPS=%16.8e   Read BW=%16.8e  GiB/sec  Write BW=%16.8e GiB/sec  Total BW=%16.8e GiB/sec",
			time, total_gflops, read_bw, write_bw, total_bw);

}
TEST_P(CMatMultTimeAVX512, TimeGcCMatMultGcCoeffAdd)
{
	const int N = GetParam();
	const float alpha =-0.2456;

	MG::MasterLog(INFO, "Timing N=%d", N);

	MG::MasterLog(INFO, "Calibrating");
	double time=0;
	int iters = 1;
	do {
		iters *= 2;
		time -= omp_get_wtime();
		for(int j=0; j < iters; ++j) {
			GcCMatMultGcCoeffAddAVX512(y,alpha,A,x,N );
		}
		time += omp_get_wtime();
	}
	while( time < 1.0);
	MG::MasterLog(INFO, "Warming up");
	for(int j=0; j < iters; ++j) {
		GcCMatMultGcCoeffAddAVX512(y,alpha,A,x,N );
	}
	MG::MasterLog(INFO, "Timing: %d iters", iters);
	time -= omp_get_wtime();
	for(int j=0; j < iters; ++j) {
			GcCMatMultGcCoeffAddAVX512(y,alpha,A,x,N );
	}
	time += omp_get_wtime();

	// Iters * rows * ( colunm * cmult + (column-1)*cmadd + 2 multiplies by alpha + 2 adds )
	double iterflops = static_cast<double>(N*(  N*COMPLEX_MULT_FLOPS + (N-1)*COMPLEX_ADD_FLOPS + 2 + 2 ) );
	double flops =static_cast<double>( iters )*iterflops;

	double total_gflops = (flops/time)*1.0e-9;

	// N * N is the matrix 2*N is input and output (read for output) and next 2* is for complex
	double iter_read_bytes = static_cast<double>( ( N*N  + 2*N )*2*sizeof(float));
	double iter_write_bytes = static_cast<double>( N*2*sizeof(float));

	double total_read_bytes = static_cast<double>(iters*iter_read_bytes);
	double total_write_bytes = static_cast<double>(iters*iter_write_bytes);
	double total_bytes = total_read_bytes + total_write_bytes;

	double gigi = 1024.0*1024.0*1024.0;

	double read_bw = (total_read_bytes / time)/gigi;
	double write_bw = (total_write_bytes / time)/gigi;
	double total_bw = (total_bytes/time)/gigi;

	MasterLog(INFO, "AVX512  GcCMatMultGcCoeffAdd Time=%16.8e (sec)  GFLOPS=%16.8e   Read BW=%16.8e  GiB/sec  Write BW=%16.8e GiB/sec  Total BW=%16.8e GiB/sec",
			time, total_gflops, read_bw, write_bw, total_bw);

}
#endif

INSTANTIATE_TEST_CASE_P(TestAllSizes,
                       CMatMultTime,
                        ::testing::Values(16,32,48, 64 ));

#ifdef MG_USE_AVX512
INSTANTIATE_TEST_CASE_P(TestAllSizes,
                       CMatMultTimeAVX512,
                        ::testing::Values(16,32,48,64 ));
#endif
int main(int argc, char *argv[])
{
	return MGTesting::TestMain(&argc, argv);
}
