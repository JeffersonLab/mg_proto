/*
 * memory_posix.cpp
 *
 *  Created on: Oct 13, 2015
 *      Author: bjoo
 */


#include <stdlib.h>

namespace MG
{
	int regular_alloc(void **ptr, size_t size, size_t alignment)
	{
		int ret_val = posix_memalign(ptr, alignment, size);
		return ret_val;
	}

	void regular_free(void *ptr, size_t size)
	{
		free(ptr);
	}

	int fast_alloc(void **ptr, size_t size, size_t alignment)
	{
		int ret_val = posix_memalign(ptr, alignment, size);
		return ret_val;
	}

	void fast_free(void *ptr, size_t size)
	{
		free(ptr);
	}

}



