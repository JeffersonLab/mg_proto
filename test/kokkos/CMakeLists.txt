
# this makes the executable out of my test_file and the 'gtest_all.cc' file
# and specifies linking to the 'libmg.a' library
add_compile_options(${EXT_CXXFLAGS})
			
add_executable(test_kokkos test_kokkos.cpp)
target_link_libraries(test_kokkos  qdpxx_utils gtest_all mg_test mg kokkos ${EXT_LIBS})

add_test( NAME TestKokkosWorking COMMAND ./test_kokkos -geom 1 1 1 1)
