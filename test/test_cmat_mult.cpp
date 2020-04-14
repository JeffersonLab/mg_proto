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
		tmp2 = static_cast<float*>(MG::MemoryAllocate(2*N*sizeof(float)));
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
		MG::MemoryFree(tmp2);
		MG::MemoryFree(A);
	}

	float *x;
	float *y;
	float*y2;
	float *tmp;
	float *tmp2;

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
	CMatMultAddNaive(y,A,x,N );

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
	CMatMultCoeffAddNaive( 1.0,y,alpha,A,x,N );

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


TEST_P(CMatMultTest, TestGcCMatMultGcNaive)
{
	const int N = GetParam();
	MG::MasterLog(INFO, "Testing N=%d", N);

	MG::MasterLog(MG::DEBUG2, "Computing Reference");
	GcCMatMultGcNaive(y,A,x,N );

	// tmp = Gamma_c x
	for(int i=0; i < N; ++i) {
		tmp[i] = x[i];
	}
	for(int i=N; i < 2*N; ++i) {
		tmp[i] = -x[i];
	}

	CMatMultNaive(y2,A,tmp,N );

	// Gamma_c y2 (in place, so flip signs of lower)
	for(int i=N; i < 2*N; ++i) {
		y2[i] = -y2[i];
	}


	for(int i=0; i < 2*N; ++i) {
		float absdiff = fabs(y[i]-y2[i]);
		ASSERT_LT(absdiff, 5.0e-5);

	}
}

TEST_P(CMatMultTest, TestGcCMatMultGcNaiveCoeffAdd)
{
	const int N = GetParam();
	MG::MasterLog(INFO, "Testing N=%d", N);

	MG::MasterLog(MG::DEBUG2, "Computing Reference");
	float alpha = 0.263;
	GcCMatMultGcCoeffAddNaive( 1.0,y,alpha,A,x,N );

	// tmp = Gamma_c x
	for(int i=0; i < N; ++i) {
		tmp[i] = x[i];
	}
	for(int i=N; i < 2*N; ++i) {
		tmp[i] = -x[i];
	}

	CMatMultNaive(tmp2,A,tmp,N );

	// Gamma_c y2 (in place, so flip signs of lower)
	for(int i=0; i < N; ++i) {
			y2[i] += alpha*tmp2[i];
		}
	for(int i=N; i < 2*N; ++i) {
		y2[i] -= alpha*tmp2[i];
	}


	for(int i=0; i < 2*N; ++i) {
		float absdiff = fabs(y[i]-y2[i]);
		ASSERT_LT(absdiff, 5.0e-5);

	}
}


INSTANTIATE_TEST_CASE_P(TestAllSizes,
                       CMatMultTest,
                        ::testing::Values(6, 8, 12, 16, 24, 32, 48, 64 ));

#ifdef MG_USE_AVX512

class CMatMultTestAVX512 : public CMatMultTest {};

/* ------- TESTS START HERE --------- */

TEST_P(CMatMultTestAVX512, TestCMatMultWithEigen)
{
	const int N = GetParam();
	MG::MasterLog(INFO, "Testing N=%d", N);

	MG::MasterLog(MG::DEBUG2, "Computing Reference");
	CMatMultAVX512(y,A,x,N );

	ComplexMatrix  in_mat(reinterpret_cast<std::complex<float>*>(A),N,N);
	ComplexVector  eigen_x(reinterpret_cast<std::complex<float>*>(x),N);
	ComplexVector  eigen_out(reinterpret_cast<std::complex<float>*>(y2),N);

	eigen_out = in_mat*eigen_x;
	for(int i=0; i < 2*N; ++i) {
		float absdiff = fabs(y[i]-y2[i]);
		ASSERT_LT(absdiff, 5.0e-5);

	}
}

TEST_P(CMatMultTestAVX512, TestCMatMultAddWithEigen)
{
	const int N = GetParam();
	MG::MasterLog(INFO, "Testing N=%d", N);

	MG::MasterLog(MG::DEBUG2, "Computing Reference");
	CMatMultAddAVX512(y,A,x,N );

	ComplexMatrix  in_mat(reinterpret_cast<std::complex<float>*>(A),N,N);
	ComplexVector  eigen_x(reinterpret_cast<std::complex<float>*>(x),N);
	ComplexVector  eigen_out(reinterpret_cast<std::complex<float>*>(y2),N);

	eigen_out += in_mat*eigen_x;
	for(int i=0; i < 2*N; ++i) {
		float absdiff = fabs(y[i]-y2[i]);
		ASSERT_LT(absdiff, 5.0e-5);

	}
}

TEST_P(CMatMultTestAVX512, TestCMatMultCoeffAddWithEigen)
{
	const int N = GetParam();
	MG::MasterLog(INFO, "Testing N=%d", N);

	float alpha=-0.2456;
	MG::MasterLog(MG::DEBUG2, "Computing Reference");
	CMatMultCoeffAddAVX512(y,alpha,A,x,N );

	ComplexMatrix  in_mat(reinterpret_cast<std::complex<float>*>(A),N,N);
	ComplexVector  eigen_x(reinterpret_cast<std::complex<float>*>(x),N);
	ComplexVector  eigen_out(reinterpret_cast<std::complex<float>*>(y2),N);

	eigen_out += alpha*in_mat*eigen_x;
	for(int i=0; i < 2*N; ++i) {
		float absdiff = fabs(y[i]-y2[i]);
		ASSERT_LT(absdiff, 5.0e-5);

	}
}

TEST_P(CMatMultTestAVX512, TestCMatAdjMultEigen)
{
	const int N = GetParam();
	MG::MasterLog(INFO, "Testing N=%d", N);


	MG::MasterLog(MG::DEBUG2, "Computing Reference");
	CMatAdjMultAVX512(y,A,x,N );

	ComplexMatrix  in_mat(reinterpret_cast<std::complex<float>*>(A),N,N);
	ComplexVector  eigen_x(reinterpret_cast<std::complex<float>*>(x),N);
	ComplexVector  eigen_out(reinterpret_cast<std::complex<float>*>(y2),N);

	eigen_out = in_mat.adjoint()*eigen_x;
	for(int i=0; i < 2*N;i++) {
		float absdiff= fabs(y[i]-y2[i]);
		// MasterLog(INFO, "i=%d  diff=%16.8e", i, absdiff);
		ASSERT_LT(absdiff, 5.0e-5);

	}
}

TEST_P(CMatMultTestAVX512, TestGcCMatMultGcCoeffAddAVX512)
{
	const int N = GetParam();
	MG::MasterLog(INFO, "Testing N=%d", N);

	MG::MasterLog(MG::DEBUG2, "Computing Reference");
	float alpha = 0.263;
	GcCMatMultGcCoeffAddAVX512(y,alpha,A,x,N );

	// tmp = Gamma_c x
	for(int i=0; i < N; ++i) {
		tmp[i] = x[i];
	}
	for(int i=N; i < 2*N; ++i) {
		tmp[i] = -x[i];
	}

	CMatMultNaive(tmp2,A,tmp,N );

	// Gamma_c y2 (in place, so flip signs of lower)
	for(int i=0; i < N; ++i) {
			y2[i] += alpha*tmp2[i];
		}
	for(int i=N; i < 2*N; ++i) {
		y2[i] -= alpha*tmp2[i];
	}


	for(int i=0; i < 2*N; ++i) {
		float absdiff = fabs(y[i]-y2[i]);
		ASSERT_LT(absdiff, 5.0e-5);

	}
}

TEST_P(CMatMultTestAVX512, TestGcCMatMultGcAVX512)
{
	const int N = GetParam();
	MG::MasterLog(INFO, "Testing N=%d", N);

	MG::MasterLog(MG::DEBUG2, "Computing Reference");
	GcCMatMultGcAVX512(y,A,x,N );

	// tmp = Gamma_c x
	for(int i=0; i < N; ++i) {
		tmp[i] = x[i];
	}
	for(int i=N; i < 2*N; ++i) {
		tmp[i] = -x[i];
	}

	CMatMultNaive(y2,A,tmp,N );

	// Gamma_c y2 (in place, so flip signs of lower)
	for(int i=N; i < 2*N; ++i) {
		y2[i] = -y2[i];
	}


	for(int i=0; i < 2*N; ++i) {
		float absdiff = fabs(y[i]-y2[i]);
		ASSERT_LT(absdiff, 5.0e-5);

	}
}

INSTANTIATE_TEST_CASE_P(TestAVX512Sizes,
                       CMatMultTestAVX512,
                        ::testing::Values(8,16, 24, 32, 48, 56, 64 ));

#endif


int main(int argc, char *argv[])
{
	return MGTesting::TestMain(&argc, argv);
}
