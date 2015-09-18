
# GTest needs Pthreads it seems -- this package finds it
find_package(Threads REQUIRED)

# this makes the executable out of my test_file and the 'gtest_all.cc' file
# and specifies linking to the 'libmg.a' library

add_executable(test_lattice test_lattice.cpp
"${CMAKE_SOURCE_DIR}/external/gtest/gtest-all.cc")
target_link_libraries(test_lattice mg)

# This should add the -lpthread onto the link line
if(CMAKE_THREAD_LIBS_INIT)
 target_link_libraries(test_lattice "${CMAKE_THREAD_LIBS_INIT}")
endif()

