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
#if 1
TEST(CMatMult, TestCorrectness)
{
	const int N = 40;
#if 1
	float *x = static_cast<float*>(MG::MemoryAllocate(2*N*sizeof(float)));
	float *y = static_cast<float*>(MG::MemoryAllocate(2*N*sizeof(float)));
	float *A = static_cast<float*>(MG::MemoryAllocate(2*N*N*sizeof(float)));
	float *A_T = static_cast<float*>(MG::MemoryAllocate(2*N*N*sizeof(float)));
	float *y2 = static_cast<float*>(MG::MemoryAllocate(2*N*sizeof(float)));
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
		MG::MasterLog(MG::DEBUG2, "i=%d",i);
		x[i] = (float)(i);
	}
	MG::MasterLog(MG::DEBUG2, "Filling Matrices");
	for(IndexType row=0; row < N; ++row) {
			for(IndexType col=0; col < N; ++col) {
				for(IndexType z=0; z < 2; ++z) {
					MG::MasterLog(MG::DEBUG2, "(row,col,cmplx)=(%d,%d,%d)",
								row,col,z);

					A[(2*N)*row + 2*col + z] = (float)((2*N)*row + 2*col + z);
					A_T[ (2*N)*col + 2*row + z ] = A[(2*N)*row + 2*col + z];
				}
			}
		}
	MG::MasterLog(MG::DEBUG2, "Done");

	std::complex<float>* xc = reinterpret_cast<std::complex<float>*>(&x[0]);
	std::complex<float>* Ac = reinterpret_cast<std::complex<float>*>(&A[0]);
	std::complex<float>* yc = reinterpret_cast<std::complex<float>*>(&y[0]);

	MG::MasterLog(MG::DEBUG2, "Computing Reference");
	CMatMultNaive(yc,Ac,xc,N );
	MG::MasterLog(MG::DEBUG2, "Computing Optimized");

#pragma omp parallel shared(y2,A_T,x)
	{
		int tid=omp_get_thread_num();
		int nthreads=omp_get_num_threads();
		CMatMult(y2,A_T,x,N, tid, nthreads);
	}
	MG::MasterLog(MG::DEBUG2, "Comparing");
	for(int i=0; i < 2*N; ++i) {
		MG::MasterLog(MG::DEBUG3, "x[%d]=%g y[%d]=%g y2[%d]=%g",i,x[i],i,y[i],i,y2[i]);
		ASSERT_NEAR(y[i],y2[i], 5.0e-6*abs(y2[i]));
	}
	MG::MasterLog(MG::DEBUG2, "Done");
#if 1
	MG::MemoryFree(y);
	MG::MemoryFree(y2);
	MG::MemoryFree(A);
	MG::MemoryFree(A_T);
	MG::MemoryFree(x);
#endif
}

TEST(CMatMultVrow, TestCorrectness)
{
	const int N = 40;
#if 1
	float *x = static_cast<float*>(MG::MemoryAllocate(2*N*sizeof(float)));
	float *y = static_cast<float*>(MG::MemoryAllocate(2*N*sizeof(float)));
	float *A = static_cast<float*>(MG::MemoryAllocate(2*N*N*sizeof(float)));
	float *A_T = static_cast<float*>(MG::MemoryAllocate(2*N*N*sizeof(float)));
	float *y2 = static_cast<float*>(MG::MemoryAllocate(2*N*sizeof(float)));
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
		MG::MasterLog(MG::DEBUG2, "i=%d",i);
		x[i] = (float)(i);
	}
	MG::MasterLog(MG::DEBUG2, "Filling Matrices");
	for(IndexType row=0; row < N; ++row) {
			for(IndexType col=0; col < N; ++col) {
				for(IndexType z=0; z < 2; ++z) {
					MG::MasterLog(MG::DEBUG2, "(row,col,cmplx)=(%d,%d,%d)",
								row,col,z);

					A[(2*N)*row + 2*col + z] = (float)((2*N)*row + 2*col + z);
					A_T[ (2*N)*col + 2*row + z ] = A[(2*N)*row + 2*col + z];
				}
			}
		}
	MG::MasterLog(MG::DEBUG2, "Done");

	std::complex<float>* xc = reinterpret_cast<std::complex<float>*>(&x[0]);
	std::complex<float>* Ac = reinterpret_cast<std::complex<float>*>(&A[0]);
	std::complex<float>* yc = reinterpret_cast<std::complex<float>*>(&y[0]);

	MG::MasterLog(MG::DEBUG2, "Computing Reference");
	CMatMultNaive(yc,Ac,xc,N );
	MG::MasterLog(MG::DEBUG2, "Computing Optimized");

#pragma omp parallel shared(y2,A_T,x)
	{
		int tid=omp_get_thread_num();
		const int n_smt = N_SMT;
		// tid=smd_it + n_smt*core_id;
		int n_threads=omp_get_num_threads();

		int core_id = tid/n_smt;
		int smt_id = tid - n_smt*core_id;

		int N_vrows = N / (VECLEN/2);

#pragma omp critical
{
		std::printf("Thread=%d of %d: cid=%d, smtid=%d\n",
					tid,n_threads,core_id,smt_id);


}
		CMatMultVrowSMT(y2,A_T,x,N,smt_id, n_smt,N_vrows);
	}
	MG::MasterLog(MG::DEBUG2, "Comparing");
	for(int i=0; i < 2*N; ++i) {
		MG::MasterLog(MG::DEBUG3, "x[%d]=%g y[%d]=%g y2[%d]=%g",i,x[i],i,y[i],i,y2[i]);
		ASSERT_NEAR(y[i],y2[i], 5.0e-6*abs(y2[i]));
	}
	MG::MasterLog(MG::DEBUG2, "Done");
#if 1
	MG::MemoryFree(y);
	MG::MemoryFree(y2);
	MG::MemoryFree(A);
	MG::MemoryFree(A_T);
	MG::MemoryFree(x);
#endif
}


TEST(CMatMultVrow, TestSpeed)
{
	const int N = 40;
	const int N_iter = 10000000;
	const int N_warm = 2000;
	const int n_smt = N_SMT;

#if 1
	float *x = static_cast<float*>(MG::MemoryAllocate(2*N*sizeof(float)));
	float *y = static_cast<float*>(MG::MemoryAllocate(2*N*sizeof(float)));
	float *A = static_cast<float*>(MG::MemoryAllocate(2*N*N*sizeof(float)));
	float *y2 = static_cast<float*>(MG::MemoryAllocate(2*N*sizeof(float)));

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
		int core_id = tid/n_smt;
		int smt_id = tid - n_smt*core_id;

#pragma omp critical
{
		std::printf("Thread=%d of %d: N_vrows=%d core_id=%d smt_id=%d\n",
					tid,n_threads,N_vrows,core_id, smt_id);
}

		// Warm cache
		MasterLog(INFO, "Warming up");
		for(int iter=0; iter < N_warm; ++iter ) {
			CMatMultVrowSMT(y2,A,x,N,smt_id,n_smt,N_vrows);
		}

		MasterLog(INFO, "Timing %d iterations", N_iter);
		// Time Optimized
#pragma omp master
		{
			start_time = omp_get_wtime();
		}

#pragma omp barrier
		for(int iter=0; iter < N_iter; ++iter ) {
			CMatMultVrowSMT(y2, A, x,N,smt_id,n_smt,N_vrows);
		}
#pragma omp barrier
#pragma omp master
		{
			end_time = omp_get_wtime();
		}
	} // end parallel
	time=end_time - start_time;
	double gflops_opt = gflops/time;

	MG::MasterLog(MG::INFO, "Iters=%d CMatMultSMT: Time=%16.8e (sec) Flops = %16.8e (GF)",
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
	MG::MasterLog(MG::INFO, "Iters=%d CMatMultNaive: Time=%16.8e (sec) Flops = %16.8e (GF)",
			N_iter, time, gflops_naive);


	MG::MasterLog(MG::INFO, "Speedup = gflops_opt/gflops_naive=%16.8e", gflops_opt/gflops_naive);
#endif


#if 1
	MG::MemoryFree(y);
	MG::MemoryFree(y2);
	MG::MemoryFree(A);
	MG::MemoryFree(x);
#endif
}

#if 0
TEST(CMatMultVrow, TestSpeedNoSum)
{
	const int N = 40;
	const int N_iter = 3000000;
	const int N_warm = 0;
	const int N_dir = 8;
	const int N_mv_parallel = 1;


#if 1
	float *x = static_cast<float*>(MG::MemoryAllocate(N_dir*2*N*sizeof(float)));
	float *y_dir = static_cast<float*>(MG::MemoryAllocate(N_dir*2*N*sizeof(float)));
	float *A = static_cast<float*>(MG::MemoryAllocate(N_dir*2*N*N*sizeof(float)));
	float *y = static_cast<float*>(MG::MemoryAllocate(2*N*sizeof(float)));
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

	MG::MasterLog(MG::INFO, "Filling x");
	// Initialize y
#pragma omp simd aligned(y:16) safelen(VECLEN)
	for(int i=0; i < 2*N; ++i) y[i] = 0;

#pragma omp parallel
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


		for(int dir=min_dir; dir < max_dir; ++dir) {
			for(int vrow=min_vrow; vrow < max_vrow; ++vrow) {
				for(int i=0; i < VECLEN; ++i) {
//					x[i+vrow*VECLEN + (2*N)*dir] = (float)(i+vrow*VECLEN+(2*N)*dir);
					x[i+vrow*VECLEN + (2*N)*dir] = 0.1;
					y_dir[i+vrow*VECLEN + (2*N)*dir ] = 0;
				}
			}
		}


		for(int dir=min_dir; dir < max_dir; ++dir) {
			for(int vrow=min_vrow; vrow < max_vrow; ++vrow) {
				int row = vrow*VECLEN2;
				for(int col=0; col < N; ++col) {
					for(int cmplx=0; cmplx < 2; ++cmplx) {
//					A[(2*N)*row + 2*col + cmplx] = d(gen);
//						A[(2*N*N*dir) + (2*N)*row + 2*col + cmplx] = (float)((2*N*N*dir)+(2*N)*row + 2*col + cmplx);
						A[(2*N*N*dir) + (2*N)*row + 2*col + cmplx] = 1.4;
					}
				}
			}
		}
	}

	MG::MasterLog(MG::INFO, "Done");
	__declspec(aligned(64)) double start_time[128][8];
	__declspec(aligned(64)) double end_time[128][8];
	double time=0;
	double N_dble = static_cast<double>(N);
	double N_iter_dble = static_cast<double>(N_iter);
	double gflops=N_iter_dble*(N_dir*(N_dble*(8*N_dble-2))+(N_dir-1)*2*N)/1.0e9;
	double outer_start_time=omp_get_wtime();

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

MG::MasterLog(MG::INFO, "Warming Up");
// 	Start in sync
#pragma omp barrier
		for(int iter=0; iter < N_iter; ++iter ) {

				for(int dir=min_dir; dir < max_dir; ++dir) {

					CMatMultVrow(&y_dir[2*N*dir],&A[2*N*N*dir],&x[2*N*dir],N,min_vrow,max_vrow);

				}
#pragma omp barrier
#pragma omp master
				{
#pragma omp simd aligned(y:16) aligned(y_dir:16) safelen(VECLEN)
						for(int j=0; j < 2*N; ++j) {
							y[j]=y_dir[j];
						}
						for(int dir=1; dir < N_dir;++dir) {
#pragma omp simd aligned(y:16) aligned(y_dir:16) safelen(VECLEN)
							for(int j=0; j < 2*N; ++j) {
								y[j]+=y_dir[2*N*dir+j];
							}
						}

				}

		}
#pragma omp barrier

		MG::MasterLog(MG::INFO, "Done");
		MG::MasterLog(MG::INFO, "Timing");
#endif

		start_time[tid][0] = omp_get_wtime();
		for(int iter=0; iter < N_iter; ++iter ) {

			for(int dir=min_dir; dir < max_dir; ++dir) {

				CMatMultVrow(&y_dir[2*N*dir],&A[2*N*N*dir],&x[2*N*dir],N,min_vrow,max_vrow);

			}
#pragma omp barrier
#pragma omp master
				{
#pragma omp simd aligned(y:16) aligned(y_dir:16) safelen(VECLEN)
						for(int j=0; j < 2*N; ++j) {
							y[j]=y_dir[j];
						}
						for(int dir=1; dir < N_dir;++dir) {
#pragma omp simd aligned(y:16) aligned(y_dir:16) safelen(VECLEN)
							for(int j=0; j < 2*N; ++j) {
								y[j]=y_dir[2*N*dir+j];
							}
						}

				}
		}
		end_time[tid][0] = omp_get_wtime();
	} // end parallel
	double outer_end_time=omp_get_wtime();

	time=0;
	double max_time=0;
	double min_time=9999999999;
	int n_threads=omp_get_max_threads();
	for(int i=0;i < n_threads;++i ) {
		double thread_time = end_time[i][0]-start_time[i][0];
	//	double thread_time = outer_end_time - outer_start_time;
		printf("Thread %d took %16.8e secs\n", i, thread_time);
		time += thread_time;
		if( thread_time > max_time ) max_time = thread_time;
		if( thread_time < min_time ) min_time = thread_time;

	}
	time/=n_threads;
	double gflops_opt = gflops/time;

	MG::MasterLog(MG::INFO, "Iters=%d CMatMultVrow: Time=%16.8e (sec) Flops = %16.8e / %16.8e / %16.8e (GF)  Avg/Min Thread/Max Thread",
			N_iter, time, gflops_opt, gflops/max_time, gflops/min_time);



#if 1
	MG::MemoryFree(y);
	MG::MemoryFree(y_dir);
	MG::MemoryFree(A);
	MG::MemoryFree(x);
#endif
}


#endif
#endif


int main(int argc, char *argv[])
{
	return MGTesting::TestMain(&argc, argv);
}
