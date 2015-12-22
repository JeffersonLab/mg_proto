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

using namespace MGGeometry;


TEST(CMatMult, TestCorrectness)
{
	const int N = 40;
#if 1
	float *x = static_cast<float*>(MGUtils::MemoryAllocate(2*N*sizeof(float)));
	float *y = static_cast<float*>(MGUtils::MemoryAllocate(2*N*sizeof(float)));
	float *A = static_cast<float*>(MGUtils::MemoryAllocate(2*N*N*sizeof(float)));
	float *A_T = static_cast<float*>(MGUtils::MemoryAllocate(2*N*N*sizeof(float)));
	float *y2 = static_cast<float*>(MGUtils::MemoryAllocate(2*N*sizeof(float)));
#else
	__declspec(align(64)) float x[2*N];
	__declspec(align(64)) float y[2*N];
	__declspec(align(64)) float y2[2*N];
	__declspec(align(64)) float A[2*N*N];
	__declspec(align(64)) float A_T[2*N*N];
#endif
	/* Fill A and X with Gaussian Noise */
#if 0
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<> d(0,2);
#endif
	for(int i=0; i < 2*N; ++i) {
		MGUtils::MasterLog(MGUtils::DEBUG2, "i=%d",i);
		x[i] = (float)(i);
	}
	MGUtils::MasterLog(MGUtils::DEBUG2, "Filling Matrices");
	for(IndexType row=0; row < N; ++row) {
			for(IndexType col=0; col < N; ++col) {
				for(IndexType z=0; z < 2; ++z) {
					MGUtils::MasterLog(MGUtils::DEBUG2, "(row,col,cmplx)=(%d,%d,%d)",
								row,col,z);

					A[(2*N)*row + 2*col + z] = (float)((2*N)*row + 2*col + z);
					A_T[ (2*N)*col + 2*row + z ] = A[(2*N)*row + 2*col + z];
				}
			}
		}
	MGUtils::MasterLog(MGUtils::DEBUG2, "Done");

	std::complex<float>* xc = reinterpret_cast<std::complex<float>*>(&x[0]);
	std::complex<float>* Ac = reinterpret_cast<std::complex<float>*>(&A[0]);
	std::complex<float>* yc = reinterpret_cast<std::complex<float>*>(&y[0]);

	MGUtils::MasterLog(MGUtils::DEBUG2, "Computing Reference");
	CMatMultNaive(yc,Ac,xc,N );
	MGUtils::MasterLog(MGUtils::DEBUG2, "Computing Optimized");
	CMatMult(y2,A_T,x,N);
	MGUtils::MasterLog(MGUtils::DEBUG2, "Comparing");
	for(int i=0; i < 2*N; ++i) {
		MGUtils::MasterLog(MGUtils::DEBUG3, "x[%d]=%g y[%d]=%g y2[%d]=%g",i,x[i],i,y[i],i,y2[i]);
		ASSERT_NEAR(y[i],y2[i], 5.0e-6*abs(y2[i]));
	}
	MGUtils::MasterLog(MGUtils::DEBUG2, "Done");
#if 1
	MGUtils::MemoryFree(y);
	MGUtils::MemoryFree(y2);
	MGUtils::MemoryFree(A);
	MGUtils::MemoryFree(A_T);
	MGUtils::MemoryFree(x);
#endif
}

TEST(CMatMult, TestSpeed)
{
	const int N = 40;
	const int N_iter = 10000000;
	const int N_warm = 2000;
#if 0
	__declspec(align(64)) float x[2*N];
	__declspec(align(64)) float y[2*N];
	__declspec(align(64)) float y2[2*N];
	__declspec(align(64)) float A[2*N*N];
#else
	float *x = static_cast<float*>(MGUtils::MemoryAllocate(2*N*sizeof(float)));
	float *y = static_cast<float*>(MGUtils::MemoryAllocate(2*N*sizeof(float)));
	float *A = static_cast<float*>(MGUtils::MemoryAllocate(2*N*N*sizeof(float)));
	float *y2 = static_cast<float*>(MGUtils::MemoryAllocate(2*N*sizeof(float)));
#endif

	std::complex<float>* xc = reinterpret_cast<std::complex<float>*>(&x[0]);
	std::complex<float>* Ac = reinterpret_cast<std::complex<float>*>(&A[0]);
	std::complex<float>* yc = reinterpret_cast<std::complex<float>*>(&y[0]);

	/* Fill A and X with Gaussian Noise */
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<> d(0,2);
	for(int i=0; i < 2*N; ++i) {
		//x[i] = d(gen);
		x[i] = (float)(i);
	}

	for(IndexType row=0; row < N; ++row) {
		for(IndexType col=0; col < N; ++col) {
			for(IndexType cmplx=0; cmplx < 2; ++cmplx) {
//				A[(2*N)*row + 2*col + cmplx] = d(gen);
				A[(2*N)*row + 2*col + cmplx] = (float)((2*N)*row + 2*col + cmplx);
			}
		}
	}


	double start_time=0;
	double end_time=0;
	double time=0;
	double N_dble = static_cast<double>(N);
	double N_iter_dble = static_cast<double>(N_iter);
	double gflops=N_iter_dble*(N_dble*(8*N_dble-2))/1.0e9;

	// Warm cache
	for(int iter=0; iter < N_warm; ++iter ) {
			CMatMult(y2,A,x,N);
	}

	// Time Optimized
	start_time = omp_get_wtime();
	for(int iter=0; iter < N_iter; ++iter ) {
		CMatMult(y2,A,x,N);
	}
	end_time = omp_get_wtime();
	time=end_time - start_time;
	double gflops_opt = gflops/time;

	MGUtils::MasterLog(MGUtils::INFO, "Iters=%d CMatMult: Time=%16.8e (sec) Flops = %16.8e (GF)",
			N_iter, time, gflops_opt);

	// Time Naive

	// Warm cache
	for(int iter=0; iter < N_warm; ++iter ) {
		CMatMultNaive(yc,Ac,xc,N );
	}

	start_time = omp_get_wtime();
	for(int iter=0; iter < N_iter; ++iter ) {

		CMatMultNaive(yc,Ac,xc,N );

	}
	end_time = omp_get_wtime();
	time = end_time - start_time;
	double gflops_naive = gflops/time;
	MGUtils::MasterLog(MGUtils::INFO, "Iters=%d CMatMultNaive: Time=%16.8e (sec) Flops = %16.8e (GF)",
			N_iter, time, gflops_naive);


	MGUtils::MasterLog(MGUtils::INFO, "Speedup = gflops_opt/gflops_naive=%16.8e", gflops_opt/gflops_naive);



#if 1
	MGUtils::MemoryFree(y);
	MGUtils::MemoryFree(y2);
	MGUtils::MemoryFree(A);
	MGUtils::MemoryFree(x);
#endif
}
int main(int argc, char *argv[])
{
	return MGTesting::TestMain(&argc, argv);
}
