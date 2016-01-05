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

using namespace MGGeometry;
using namespace MGUtils;

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

TEST(CMatMultVrow, TestCorrectness)
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
		int n_threads=omp_get_num_threads();
		int N_vrows = N / (VECLEN/2);

		int n_vrow_per_thread = N_vrows/n_threads;
		if ( N_vrows % n_threads != 0 ) n_vrow_per_thread++;
		int min_vrow = tid*n_vrow_per_thread;
		int max_vrow = MinInt( (tid+1)*n_vrow_per_thread, N_vrows);
#pragma omp critical
{
		std::printf("Thread=%d of %d: N_vrows=%d min_vrow=%d max_vrow=%d\n",
					tid,n_threads,N_vrows, min_vrow, max_vrow);

		if ( min_vrow >= N_vrows) {
			std::printf("Thread %d is idle\n",tid);
		}
}
		CMatMultVrow(y2,A_T,x,N,min_vrow, max_vrow);
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


TEST(CMatMultVrow, TestSpeed)
{
	const int N = 40;
	const int N_iter = 10000000;
	const int N_warm = 2000;
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

#pragma omp parallel shared(y2,A,x,N,start_time,end_time)
	{
		int tid=omp_get_thread_num();
		int n_threads=omp_get_num_threads();
		int N_vrows = N / (VECLEN/2);
		int N_floats_per_vrow = VECLEN;
		int cache_line_size_in_floats = 16;
		int N_vrows_per_cacheline = cache_line_size_in_floats/N_floats_per_vrow;
		int N_cachelines = N_vrows/N_vrows_per_cacheline;

		int N_cachelines_per_thread = N_cachelines/n_threads;
		if ( N_cachelines % n_threads != 0 ) N_cachelines_per_thread++;
		int n_vrow_per_thread = N_cachelines_per_thread*N_vrows_per_cacheline;
		int min_vrow = tid*n_vrow_per_thread;
		int max_vrow = MinInt( (tid+1)*n_vrow_per_thread, N_vrows);
#pragma omp critical
{
		std::printf("Thread=%d of %d: N_vrows=%d min_vrow=%d max_vrow=%d\n",
					tid,n_threads,N_vrows, min_vrow, max_vrow);

		if ( min_vrow >= N_vrows) {
			std::printf("Thread %d is idle\n",tid);
		}
}

		// Warm cache
		MasterLog(INFO, "Warming up");
		for(int iter=0; iter < N_warm; ++iter ) {
			CMatMultVrow(y2,A,x,N,min_vrow, max_vrow);
		}

		MasterLog(INFO, "Timing %d iterations", N_iter);
		// Time Optimized
#pragma omp master
		{
			start_time = omp_get_wtime();
		}

#pragma omp barrier
		for(int iter=0; iter < N_iter; ++iter ) {
			CMatMultVrow(y2, A, x,N, min_vrow, max_vrow);
		}
#pragma omp barrier
#pragma omp master
		{
			end_time = omp_get_wtime();
		}
	} // end parallel
	time=end_time - start_time;
	double gflops_opt = gflops/time;

	MGUtils::MasterLog(MGUtils::INFO, "Iters=%d CMatMult: Time=%16.8e (sec) Flops = %16.8e (GF)",
			N_iter, time, gflops_opt);

#if 1
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



TEST(CMatMultVrow, TestSpeedNoSum)
{
	const int N = 40;
	const int N_iter = 5000000;
	const int N_warm = 2000;
	const int N_dir = 8;

	const int N_mv_parallel = 1;


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
	__declspec(aligned(64)) double start_time[128][8];
	__declspec(aligned(64)) double end_time[128][8];
	double time=0;
	double N_dble = static_cast<double>(N);
	double N_iter_dble = static_cast<double>(N_iter);
	double gflops=N_iter_dble*(N_dir*(N_dble*(8*N_dble-2)))/1.0e9;


#pragma omp parallel shared(y,y_dir,start_time, end_time, N_dir,N_warm,N_iter,N_mv_parallel)
	{
		const int tid=omp_get_thread_num();
		const int n_threads=omp_get_num_threads();
		const int N_dir_parallel = n_threads/N_mv_parallel;

		// Now split tid into mv_par_id and dir_par_id
		int dir_par_id = tid/N_mv_parallel;
		int mv_par_id = tid - N_mv_parallel*dir_par_id;

		// Now assign min_vrow and max_vrow based on the cache lines of earlier
		int N_vrows = N / (VECLEN/2);
		int N_floats_per_vrow = VECLEN;
		int cache_line_size_in_floats = 16;
		int N_vrows_per_cacheline = cache_line_size_in_floats/N_floats_per_vrow;
		int N_cachelines = N_vrows/N_vrows_per_cacheline;

		int N_cachelines_per_thread = N_cachelines/N_mv_parallel;
		if ( N_cachelines % N_mv_parallel != 0 ) N_cachelines_per_thread++;
		int n_vrow_per_thread = N_cachelines_per_thread*N_vrows_per_cacheline;
		int min_vrow = mv_par_id*n_vrow_per_thread;
		int max_vrow = MinInt( (mv_par_id+1)*n_vrow_per_thread, N_vrows);

		//  Next we should determine the minimum and maximum dir we will
		//  Work with.
		int N_dir_per_dirparallel = N_dir/N_dir_parallel;
		if( N_dir % N_dir_parallel !=0 ) N_dir_per_dirparallel++;
		int min_dir = dir_par_id * N_dir_per_dirparallel;
		int max_dir = MinInt( (dir_par_id + 1)*N_dir_per_dirparallel, N_dir);

#if 1

#pragma omp critical
{
		std::printf("Thread=%d of %d: mv_par_id=%d of %d, dir_par_id=%d of %d:  N_vrows=%d min_dir=%d max_dir=%d min_vrow=%d max_vrow=%d\n",
					tid, n_threads, mv_par_id, N_mv_parallel, dir_par_id, N_dir_parallel,  N_vrows, min_dir, max_dir, min_vrow, max_vrow);

		if ( min_vrow >= N_vrows) {
			std::printf("Thread %d is idle\n",tid);
		}
}

				MGUtils::MasterLog(MGUtils::INFO, "Warming Up");
		for(int iter=0; iter < N_warm; ++iter ) {

			for(int dir=min_dir; dir < max_dir; ++dir) {
				CMatMultVrow(&y_dir[2*N*dir],&A[2*N*N*dir],&x[2*N*dir],N,min_vrow,max_vrow);
			}
		}
#pragma omp barrier

		MGUtils::MasterLog(MGUtils::INFO, "Done");
		MGUtils::MasterLog(MGUtils::INFO, "Timing");
#endif

#pragma omp barrier
		start_time[tid][0] = omp_get_wtime();

		for(int iter=0; iter < N_iter; ++iter ) {
			for(int dir=min_dir; dir < max_dir; ++dir) {
				CMatMultVrow(&y_dir[2*N*dir],&A[2*N*N*dir],&x[2*N*dir],N,min_vrow,max_vrow);
			}
//#pragma omp barrier
		}
		end_time[tid][0] = omp_get_wtime();
	} // end parallel

	time=0;
	double max_time=0;
	double min_time=9999999999;
	int n_threads=omp_get_max_threads();
	for(int i=0;i < n_threads;++i ) {
		double thread_time = end_time[i][0]-start_time[i][0];
		printf("Thread %d took %16.8e secs\n", i, thread_time);
		time += thread_time;
		if( thread_time > max_time ) max_time = thread_time;
		if( thread_time < min_time ) min_time = thread_time;

	}
	time/=n_threads;
	double gflops_opt = gflops/time;

	MGUtils::MasterLog(MGUtils::INFO, "Iters=%d CMatMultVrow: Time=%16.8e (sec) Flops = %16.8e / %16.8e / %16.8e (GF)  Avg/Min Thread/Max Thread",
			N_iter, time, gflops_opt, gflops/max_time, gflops/min_time);



#if 1
	MGUtils::MemoryFree(y);
	MGUtils::MemoryFree(y_dir);
	MGUtils::MemoryFree(A);
	MGUtils::MemoryFree(x);
#endif
}

TEST(CMatMultVrow, TestSpeedSumDirs)
{
	const int N = 40;
	const int N_iter = 5000000;
	const int N_warm = 2000;
	const int N_dir = 8;

	const int N_mv_parallel = 1;


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
	double gflops=N_iter_dble*(N_dir*(N_dble*(8*N_dble-2))+(N_dir-1)*2*N)/1.0e9;


#pragma omp parallel shared(y,y_dir,start_time, end_time, N_dir,N_warm,N_iter,N_mv_parallel)
	{
		const int tid=omp_get_thread_num();
		const int n_threads=omp_get_num_threads();
		const int N_dir_parallel = n_threads/N_mv_parallel;

		// Now split tid into mv_par_id and dir_par_id
		int dir_par_id = tid/N_mv_parallel;
		int mv_par_id = tid - N_mv_parallel*dir_par_id;

		// Now assign min_vrow and max_vrow based on the cache lines of earlier
		int N_vrows = N / (VECLEN/2);
		int N_floats_per_vrow = VECLEN;
		int cache_line_size_in_floats = 16;
		int N_vrows_per_cacheline = cache_line_size_in_floats/N_floats_per_vrow;
		int N_cachelines = N_vrows/N_vrows_per_cacheline;

		int N_cachelines_per_thread = N_cachelines/N_mv_parallel;
		if ( N_cachelines % N_mv_parallel != 0 ) N_cachelines_per_thread++;
		int n_vrow_per_thread = N_cachelines_per_thread*N_vrows_per_cacheline;
		int min_vrow = mv_par_id*n_vrow_per_thread;
		int max_vrow = MinInt( (mv_par_id+1)*n_vrow_per_thread, N_vrows);

		//  Next we should determine the minimum and maximum dir we will
		//  Work with.
		int N_dir_per_dirparallel = N_dir/N_dir_parallel;
		if( N_dir % N_dir_parallel !=0 ) N_dir_per_dirparallel++;
		int min_dir = dir_par_id * N_dir_per_dirparallel;
		int max_dir = MinInt( (dir_par_id + 1)*N_dir_per_dirparallel, N_dir);

#if 1

#pragma omp critical
{
		std::printf("Thread=%d of %d: mv_par_id=%d of %d, dir_par_id=%d of %d:  N_vrows=%d min_dir=%d max_dir=%d min_vrow=%d max_vrow=%d\n",
					tid, n_threads, mv_par_id, N_mv_parallel, dir_par_id, N_dir_parallel,  N_vrows, min_dir, max_dir, min_vrow, max_vrow);

		if ( min_vrow >= N_vrows) {
			std::printf("Thread %d is idle\n",tid);
		}
}

				MGUtils::MasterLog(MGUtils::INFO, "Warming Up");
		for(int iter=0; iter < N_warm; ++iter ) {

			for(int dir=min_dir; dir < max_dir; ++dir) {
				CMatMultVrow(&y_dir[2*N*dir],&A[2*N*N*dir],&x[2*N*dir],N,min_vrow,max_vrow);
			}
		}

		MGUtils::MasterLog(MGUtils::INFO, "Done");
		MGUtils::MasterLog(MGUtils::INFO, "Timing");
#endif

#pragma omp master
		{
		start_time = omp_get_wtime();
		}
#pragma omp barrier
		for(int iter=0; iter < N_iter; ++iter ) {

			for(int dir=min_dir; dir < max_dir; ++dir) {
				CMatMultVrow(&y_dir[2*N*dir],&A[2*N*N*dir],&x[2*N*dir],N,min_vrow,max_vrow);
			}
		} // iters
#pragma omp barrier
#pragma omp master
		{
		end_time = omp_get_wtime();
		}

	} // end parallel
	time=end_time - start_time;
	double gflops_opt = gflops/time;

	MGUtils::MasterLog(MGUtils::INFO, "Iters=%d CMatMultVrow: Time=%16.8e (sec) Flops = %16.8e (GF)",
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
