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

#pragma omp parallel shared(y2,A_T,x)
	{
		int tid=omp_get_thread_num();
		int nthreads=omp_get_num_threads();
		CMatMult(y2,A_T,x,N, tid, nthreads);
	}
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
	const int N_iter = 3000000;
	const int N_warm = 200;
#if 1
	float *x = static_cast<float*>(MGUtils::MemoryAllocate(2*N*sizeof(float)));
	float *y = static_cast<float*>(MGUtils::MemoryAllocate(2*N*sizeof(float)));
	float *A = static_cast<float*>(MGUtils::MemoryAllocate(2*N*N*sizeof(float)));
	float *y2 = static_cast<float*>(MGUtils::MemoryAllocate(2*N*sizeof(float)));

#else
	__declspec(align(64)) float x[2*N];
	__declspec(align(64)) float y[2*N];
	__declspec(align(64)) float y2[2*N];
	__declspec(align(64)) float A[2*N*N];

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

#pragma omp parallel shared(y2)
	{
		int tid=omp_get_thread_num();
		int nthreads=omp_get_num_threads();

		// Warm cache
		for(int iter=0; iter < N_warm; ++iter ) {
			CMatMult(y2,A,x,N,tid,nthreads);
		}

		// Time Optimized
#pragma omp single
		{
			start_time = omp_get_wtime();
		}
		for(int iter=0; iter < N_iter; ++iter ) {
			CMatMult(y2,A,x,N,tid,nthreads);
		}

#pragma omp single
		{
			end_time = omp_get_wtime();
		}
	} // end parallel
	time=end_time - start_time;
	double gflops_opt = gflops/time;

	MGUtils::MasterLog(MGUtils::INFO, "Iters=%d CMatMult: Time=%16.8e (sec) Flops = %16.8e (GF)",
			N_iter, time, gflops_opt);

#if 0
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
#endif


#if 1
	MGUtils::MemoryFree(y);
	MGUtils::MemoryFree(y2);
	MGUtils::MemoryFree(A);
	MGUtils::MemoryFree(x);
#endif
}

TEST(CMatMult, TestSpeed2)
{
	const int N = 40;
	const int N_iter = 1000000;
	const int N_warm = 200;
	const int N_dir = 8;

#if 1
	float *x = static_cast<float*>(MGUtils::MemoryAllocate(N_dir*2*N*sizeof(float)));
	float *y_dir = static_cast<float*>(MGUtils::MemoryAllocate(N_dir*2*N*sizeof(float)));
	float *A = static_cast<float*>(MGUtils::MemoryAllocate(N_dir*2*N*N*sizeof(float)));
	float *y = static_cast<float*>(MGUtils::MemoryAllocate(2*N*sizeof(float)));
#else
	__declspec(align(64)) float x[2*N*N_dir];
	__declspec(align(64)) float y_dir[2*N*N_dir];
	__declspec(align(64)) float 2[2*N];
	__declspec(align(64)) float A[2*N*N*N_dir];

#endif


	/* Fill A and X with Gaussian Noise */
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<> d(0,2);

	MGUtils::MasterLog(MGUtils::INFO, "Filling x");
	for(int dir=0; dir < N_dir; ++dir) {
		for(int i=0; i < 2*N; ++i) {
			//x[i] = d(gen);
			x[i + (2*N)*dir] = (float)(i+(2*N)*dir);
		}
	}

	MGUtils::MasterLog(MGUtils::INFO, "Filling A");

	for(int dir=0; dir < N_dir; ++dir) {
		for(IndexType row=0; row < N; ++row) {
			for(IndexType col=0; col < N; ++col) {
				for(IndexType cmplx=0; cmplx < 2; ++cmplx) {
//					A[(2*N)*row + 2*col + cmplx] = d(gen);
					A[(2*N*N*dir) + (2*N)*row + 2*col + cmplx] = (float)((2*N*N*dir)+(2*N)*row + 2*col + cmplx);
				}
			}
		}
	}

	MGUtils::MasterLog(MGUtils::INFO, "Done");
	double start_time=0;
	double end_time=0;
	double time=0;
	double N_dble = static_cast<double>(N);
	double N_iter_dble = static_cast<double>(N_iter);
	double gflops=N_iter_dble*N_dir*(N_dble*(8*N_dble-2))/1.0e9;

#pragma omp parallel shared(y_dir)
	{
		const int tid=omp_get_thread_num();
		const int nthreads=omp_get_num_threads();
		const int threads_per_mv = 2;
		const int n_groups = nthreads/threads_per_mv; /* Number of thread teams. Each team works on an MV */
		const int mv_tid = tid % threads_per_mv; /* thread  within an MV */
		const int mv_group = tid / threads_per_mv; /* Which MV this thread works on */


		// Warm cache
		for(int iter=0; iter < 1; ++iter ) {
			for(int dir=mv_group; dir < N_dir; dir+=n_groups) {
#if 1
#pragma omp critical
				{
					printf("tid=%d mv_tid=%d mv_group=%d dir=%d threads_per_mv=%d\n",
							tid, mv_tid, mv_group, dir, threads_per_mv);
				}
#endif
				CMatMult(&y_dir[2*N*dir],&A[2*N*N*dir],&x[2*N*dir],N,mv_tid,threads_per_mv);
			}
		}

		// Time Optimized
#pragma omp single
		{
			start_time = omp_get_wtime();
		}
		for(int iter=0; iter < N_iter; ++iter ) {
			for(int dir=mv_group; dir < N_dir; dir+=n_groups) {
				CMatMult(&y_dir[2*N*dir],&A[2*N*N*dir],&x[2*N*dir],N,mv_tid,threads_per_mv);
			}
		}

#pragma omp single
		{
			end_time = omp_get_wtime();
		}
	} // end parallel
	time=end_time - start_time;
	double gflops_opt = gflops/time;

	MGUtils::MasterLog(MGUtils::INFO, "Iters=%d CMatMult: Time=%16.8e (sec) Flops = %16.8e (GF)",
			N_iter, time, gflops_opt);



#if 1
	MGUtils::MemoryFree(y);
	MGUtils::MemoryFree(y_dir);
	MGUtils::MemoryFree(A);
	MGUtils::MemoryFree(x);
#endif
}
int main(int argc, char *argv[])
{
	return MGTesting::TestMain(&argc, argv);
}
