/*
 * memory.cpp
 *
 *  Created on: Oct 13, 2015
 *      Author: bjoo
 */

#include "MG_config.h"
#include "utils/print_utils.h"
#include "utils/memory.h"
#include <cstdlib>
#include <unordered_map>

using namespace std;

namespace MGUtils {

	// Track allocations

#ifndef MG_DEFAULT_ALIGNMENT
	static const size_t alignment = 64;      // 64 byte aligned
#else
	static const size_t alignment = MG_DEFAULT_ALIGNMENT;
#endif

	size_t GetMemoryAlignment(void)
	{
		return alignment;
	}

	// Counters to count maximum memory use.
	static size_t current_regular=0;
	static size_t max_regular=0;

	static size_t current_fast=0;
	static size_t max_fast=0;

	size_t
	GetCurrentRegularMemoryUsage(void)
	{
		return current_regular;
	}

	size_t
	GetMaxRegularMemoryUsage(void)
	{
		return max_regular;
	}

	size_t
	GetCurrentFastMemoryUsage(void)
	{
		return current_fast;
	}

	size_t
	GetMaxFastMemoryUsage(void)
	{
		return max_fast;
	}

	static unordered_map<void*, size_t> regular_mmap;
	static unordered_map<void*, size_t> fast_mmap;

	void regular_free(void *ptr, size_t size);
	void fast_free(void *ptr, size_t size);

	int regular_alloc(void **ptr, size_t size, size_t alignment);
	int fast_alloc(void **ptr, size_t size, size_t alignment);

	void* MemoryAllocate(std::size_t num_bytes, const MemorySpace space)
	{
		void *ret_val = nullptr;
		// Any thread can allocate but it is a critical
#pragma omp critical
		{
			// Allocate from the right Memory Space
			// The allocators have to abort on failure
			if ( space == FAST ) {
				if( fast_alloc( &ret_val, num_bytes, alignment ) != 0 ) {
					// We can elaborate policy later. Right now, just fail
					MasterLog(ERROR, "Fast Memory Allocation Failed" );

				}
				fast_mmap[ret_val] = num_bytes; // Add to memory map
				current_fast += num_bytes;
				if( current_fast > max_fast) max_fast = current_fast;
			}
			else {
				if( regular_alloc(&ret_val, num_bytes,alignment) != 0 ) {
					MasterLog(ERROR, "REgular Memory Allocation Failed" );
				}
				regular_mmap[ret_val] = num_bytes; // Add to memory map
				current_regular += num_bytes;
				if( current_regular > max_regular ) max_regular = current_regular;
			}

		} // End OMP critical
		return ret_val;
	}


	void MemoryFree(void *ptr)
	{
#pragma omp critical
		{
			// Locate the pointer firstin the fast memory
			auto it = fast_mmap.find(ptr);
			if ( it != fast_mmap.end() ) {
				// Found in fast map
				fast_free(it->first, it->second);
				current_fast -= it->second;
				fast_mmap.erase(it);
			}
			else {
				it = regular_mmap.find(ptr);
					if( it != regular_mmap.end() ) {
						// Found in regular map
						regular_free(it->first, it->second);
						current_regular -= it->second;
						regular_mmap.erase(it);
					}
					else {
						// If we are here, we didn't find the address in any map.
						// We need to fail.
						MasterLog(ERROR, "mem_free: Address to free %xu not in Fast or Regular memory maps",
								ptr);
					}
			}
		}

	}



	void InitMemory(void)
	{
#pragma omp master
		{
			regular_mmap.clear();
			fast_mmap.clear();
			current_regular = max_regular = 0;
			current_fast = max_fast =0;

		}
#pragma omp barrier
	}



	void FinalizeMemory(void)
	{
#pragma omp master
		{
			MasterLog(INFO, "Finalizing Memory Management");
			MasterLog(INFO, "Regular allocations: ");
			MasterLog(INFO, "\t current: %zu bytes %zu MBytes",
						current_regular, current_regular/(1024*1024));
			MasterLog(INFO, "\t max: %zu bytes %zu MBytes",
						max_regular, max_regular/(1024*1024));
			MasterLog(INFO, "Regular allocations: ");
			MasterLog(INFO, "\t current: %zu bytes %zu MBytes",
						current_fast, current_fast/(1024*1024));
			MasterLog(INFO, "\t max: %zu bytes %zu MBytes",
						max_fast, max_fast/(1024*1024));


			MasterLog(DEBUG2, "Dumping (and Freeing) Regular Table");
			for( auto it=regular_mmap.begin(); it != regular_mmap.end(); it++) {
				MasterLog(DEBUG2, "\t address=%zu, size=%zu", it->first, it->second);
				regular_free(it->first,it->second);
			}
			MasterLog(DEBUG2, "\n");
			MasterLog(DEBUG2, "Dumping (and Freeing) Fast Table");
			for( auto it=regular_mmap.begin(); it != fast_mmap.end(); it++) {
				MasterLog(DEBUG2, "\t address=%zu, size=%zu", it->first, it->second);
				fast_free(it->first,it->second);
			}

		}
	}

}

