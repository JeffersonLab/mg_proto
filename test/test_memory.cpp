/*
 * test_memory.cpp
 *
 *  Created on: Oct 13, 2015
 *      Author: bjoo
 */


#include "gtest/gtest.h"
#include "test_env.h"
#include <cstdlib>
#include "utils/memory.h"

using namespace MGUtils;

TEST(TestMemory, InitialSetup)
{
	ASSERT_EQ(GetCurrentRegularMemoryUsage(), static_cast<std::size_t>(0));
	ASSERT_EQ(GetCurrentFastMemoryUsage(), static_cast<std::size_t>(0));

	ASSERT_EQ(GetMaxRegularMemoryUsage(), static_cast<std::size_t>(0));
	ASSERT_EQ(GetMaxFastMemoryUsage(), static_cast<std::size_t>(0));
}

TEST(TestMemory, TestRegularAllocation)
{
	void *ptr = MemoryAllocate(1024, REGULAR);
	ASSERT_EQ(GetCurrentRegularMemoryUsage(), static_cast<std::size_t>(1024));
	MemoryFree(ptr);
	ASSERT_EQ(GetCurrentRegularMemoryUsage(), static_cast<std::size_t>(0));
	ASSERT_EQ(GetMaxRegularMemoryUsage(), static_cast<std::size_t>(1024));

}

TEST(TestMemory, TestFastAllocation)
{
	void *ptr = MemoryAllocate(1024, FAST);
	ASSERT_EQ(GetCurrentFastMemoryUsage(), static_cast<std::size_t>(1024));
	MemoryFree(ptr);
	ASSERT_EQ(GetCurrentFastMemoryUsage(), static_cast<std::size_t>(0));
	ASSERT_EQ(GetMaxFastMemoryUsage(), static_cast<std::size_t>(1024));
}

TEST(TestMemory, TestRegularAlignment)
{
	void *ptr = MemoryAllocate(1024, REGULAR);
	std::size_t sptr = reinterpret_cast<std::size_t>(ptr);
	ASSERT_EQ( sptr % GetMemoryAlignment(), static_cast<std::size_t>(0));
	MemoryFree(ptr);
}

TEST(TestMemory, TestFastAlignment)
{
	void *ptr = MemoryAllocate(1024, FAST);
	std::size_t sptr = reinterpret_cast<std::size_t>(ptr);
	ASSERT_EQ( sptr % GetMemoryAlignment(), static_cast<std::size_t>(0));
	MemoryFree(ptr);
}

TEST(TestMemory, TestFree)
{
	void *r_ptr = MemoryAllocate(512, REGULAR);
	void *f_ptr = MemoryAllocate(1024, FAST);
	void *r_ptr2 = MemoryAllocate(1024, REGULAR);
	void *f_ptr2 = MemoryAllocate(2048, FAST);

	ASSERT_EQ(GetCurrentRegularMemoryUsage(), static_cast<std::size_t>(1536));
	ASSERT_EQ(GetCurrentFastMemoryUsage(), static_cast<std::size_t>(3072));

	MemoryFree(r_ptr);

	ASSERT_EQ(GetCurrentRegularMemoryUsage(), static_cast<std::size_t>(1024));
	ASSERT_EQ(GetCurrentFastMemoryUsage(), static_cast<std::size_t>(3072));

	MemoryFree(f_ptr2);
	MemoryFree(f_ptr);

	ASSERT_EQ(GetCurrentRegularMemoryUsage(), static_cast<std::size_t>(1024));
	ASSERT_EQ(GetCurrentFastMemoryUsage(), static_cast<std::size_t>(0));

	MemoryFree(r_ptr2);
	ASSERT_EQ(GetCurrentRegularMemoryUsage(), static_cast<std::size_t>(0));
	ASSERT_EQ(GetCurrentFastMemoryUsage(), static_cast<std::size_t>(0));

	ASSERT_EQ(GetMaxRegularMemoryUsage(), static_cast<std::size_t>(1536));
	ASSERT_EQ(GetMaxFastMemoryUsage(), static_cast<std::size_t>(3072));
}


int main(int argc, char *argv[])
{
	return MGTesting::TestMain(&argc, argv);
}

