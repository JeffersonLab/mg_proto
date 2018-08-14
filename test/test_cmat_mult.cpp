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

// Eigen Dense header -- for testing

#include <Eigen/Dense>
using namespace Eigen;

#include <cstdlib>
// This is our test fixture. No tear down or setup
class CMatMultTest : public ::testing::TestWithParam<int> {
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

TEST_P(CMatMultTest, TestCMatMultWithEigen)
{
	const int N = GetParam();
	MG::MasterLog(INFO, "Testing N=%d", N);

	MG::MasterLog(MG::DEBUG2, "Computing Reference");
	CMatMultNaive(y,A,x,N );

	ComplexMatrix  in_mat(reinterpret_cast<std::complex<float>*>(A),N,N);
	ComplexVector  eigen_x(reinterpret_cast<std::complex<float>*>(x),N);
	ComplexVector  eigen_out(reinterpret_cast<std::complex<float>*>(y2),N);

	eigen_out = in_mat*eigen_x;
	for(int i=0; i < 2*N; ++i) {
		float absdiff = fabs(y[i]-y2[i]);
		ASSERT_LT(absdiff, 5.0e-5);

	}
}

TEST_P(CMatMultTest, TestCMatMultAddWithEigen)
{
	const int N = GetParam();
	MG::MasterLog(INFO, "Testing N=%d", N);

	MG::MasterLog(MG::DEBUG2, "Computing Reference");
	CMatMultNaiveAdd(y,A,x,N );

	ComplexMatrix  in_mat(reinterpret_cast<std::complex<float>*>(A),N,N);
	ComplexVector  eigen_x(reinterpret_cast<std::complex<float>*>(x),N);
	ComplexVector  eigen_out(reinterpret_cast<std::complex<float>*>(y2),N);

	eigen_out += (in_mat*eigen_x);
	for(int i=0; i < 2*N; ++i) {
		float absdiff = fabs(y[i]-y2[i]);
		ASSERT_LT(absdiff, 5.0e-5);

	}
}

TEST_P(CMatMultTest, TestCMatMultCoeffAddWithEigen)
{
	const int N = GetParam();
	MG::MasterLog(INFO, "Testing N=%d", N);

	float alpha=-0.754;

	MG::MasterLog(MG::DEBUG2, "Computing Reference");
	CMatMultNaiveCoeffAdd(y,alpha,A,x,N );

	ComplexMatrix  in_mat(reinterpret_cast<std::complex<float>*>(A),N,N);
	ComplexVector  eigen_x(reinterpret_cast<std::complex<float>*>(x),N);
	ComplexVector  eigen_out(reinterpret_cast<std::complex<float>*>(y2),N);

	eigen_out += alpha*(in_mat*eigen_x);
	for(int i=0; i < 2*N; ++i) {
		float absdiff = fabs(y[i]-y2[i]);
		ASSERT_LT(absdiff, 5.0e-5);

	}
}


TEST_P(CMatMultTest, TestCMatAdjMultWithEigen)
{
	const int N = GetParam();
	MG::MasterLog(INFO, "Testing N=%d", N);

	MG::MasterLog(MG::DEBUG2, "Computing Reference");
	CMatAdjMultNaive(y,A,x,N );

	ComplexMatrix  in_mat(reinterpret_cast<std::complex<float>*>(A),N,N);
	ComplexVector  eigen_x(reinterpret_cast<std::complex<float>*>(x),N);
	ComplexVector  eigen_out(reinterpret_cast<std::complex<float>*>(y2),N);

	eigen_out = in_mat.adjoint()*eigen_x;
	for(int i=0; i < 2*N; ++i) {
		float absdiff = fabs(y[i]-y2[i]);
		ASSERT_LT(absdiff, 5.0e-5);

	}
}


TEST_P(CMatMultTest, TestCMatAdjMultAddWithEigen)
{
	const int N = GetParam();
	MG::MasterLog(INFO, "Testing N=%d", N);

	MG::MasterLog(MG::DEBUG2, "Computing Reference");
	CMatAdjMultNaiveAdd(y,A,x,N );

	ComplexMatrix  in_mat(reinterpret_cast<std::complex<float>*>(A),N,N);
	ComplexVector  eigen_x(reinterpret_cast<std::complex<float>*>(x),N);
	ComplexVector  eigen_out(reinterpret_cast<std::complex<float>*>(y2),N);

	eigen_out += (in_mat.adjoint()*eigen_x);
	for(int i=0; i < 2*N; ++i) {
		float absdiff = fabs(y[i]-y2[i]);
		ASSERT_LT(absdiff, 5.0e-5);

	}
}

TEST_P(CMatMultTest, TestCMatAdjMultCoeffAddWithEigen)
{
	const int N = GetParam();
	MG::MasterLog(INFO, "Testing N=%d", N);

	float alpha=-0.754;

	MG::MasterLog(MG::DEBUG2, "Computing Reference");
	CMatAdjMultNaiveCoeffAdd(y,alpha,A,x,N );

	ComplexMatrix  in_mat(reinterpret_cast<std::complex<float>*>(A),N,N);
	ComplexVector  eigen_x(reinterpret_cast<std::complex<float>*>(x),N);
	ComplexVector  eigen_out(reinterpret_cast<std::complex<float>*>(y2),N);

	eigen_out += alpha*(in_mat.adjoint()*eigen_x);
	for(int i=0; i < 2*N; ++i) {
		float absdiff = fabs(y[i]-y2[i]);
		ASSERT_LT(absdiff, 5.0e-5);

	}
}

INSTANTIATE_TEST_CASE_P(TestAllSizes,
                       CMatMultTest,
                        ::testing::Values(6, 8, 12, 16, 24, 32, 48, 64 ));

int main(int argc, char *argv[])
{
	return MGTesting::TestMain(&argc, argv);
}
