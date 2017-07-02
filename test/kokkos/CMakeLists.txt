
# this makes the executable out of my test_file and the 'gtest_all.cc' file
# and specifies linking to the 'libmg.a' library

LIST(APPEND kokkos_source_list kokkos_constants.h
								kokkos_defaults.h
								kokkos_dslash.h
								kokkos_matvec.h
								kokkos_qdp_utils.h
								kokkos_spinproj.h
								kokkos_types.h)

add_compile_options(${EXT_CXXFLAGS})
			
add_executable(test_kokkos test_kokkos.cpp ${kokkos_source_list})
target_link_libraries(test_kokkos qdpxx_utils gtest_all mg_test mg kokkos ${EXT_LIBS})

add_executable(test_kokkos_perf test_kokkos_perf.cpp ${kokkos_source_list})
target_link_libraries(test_kokkos_perf  qdpxx_utils gtest_all mg_test mg kokkos ${EXT_LIBS})


add_test( NAME TestKokkosWorking COMMAND ./test_kokkos -geom 1 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
