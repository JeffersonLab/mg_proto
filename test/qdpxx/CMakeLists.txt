
# this makes the executable out of my test_file and the 'gtest_all.cc' file
# and specifies linking to the 'libmg.a' library
add_compile_options(${EXT_CXXFLAGS})
add_library(qdpxx_utils reunit.h reunit.cpp qdpxx_latticeinit.h qdpxx_latticeinit.cpp)

			
target_compile_options(mg_test PUBLIC ${EXT_CXXFLAGS})

add_executable(test_qdpxx test_qdpxx.cpp)
target_link_libraries(test_qdpxx  qdpxx_utils gtest_all mg_test mg  ${EXT_LIBS})

add_executable(test_blocking test_blocking.cpp)
target_link_libraries(test_blocking  qdpxx_utils gtest_all mg_test mg ${EXT_LIBS})

add_executable(test_coarse_qdpxx test_coarse_qdpxx.cpp)
target_link_libraries(test_coarse_qdpxx  qdpxx_utils gtest_all mg_test mg ${EXT_LIBS})

add_executable(test_coarse_qdpxx_site test_coarse_qdpxx_site.cpp)
target_link_libraries(test_coarse_qdpxx_site  qdpxx_utils gtest_all mg_test mg ${EXT_LIBS})

add_executable(test_qdpxx_clover_linop test_qdpxx_clover_linop.cpp)
target_link_libraries(test_qdpxx_clover_linop  qdpxx_utils gtest_all mg_test mg ${EXT_LIBS})

add_executable(test_qdpxx_coarse_clover_op test_qdpxx_coarse_clover_op.cpp)
target_link_libraries(test_qdpxx_coarse_clover_op qdpxx_utils gtest_all mg_test mg ${EXT_LIBS})

add_executable(test_vcycle test_vcycle.cpp)
target_link_libraries(test_vcycle qdpxx_utils gtest_all mg_test mg ${EXT_LIBS})

add_executable(test_coarse_coarse_blocking test_coarse_coarse_blocking.cpp)
target_link_libraries(test_coarse_coarse_blocking  qdpxx_utils gtest_all mg_test mg ${EXT_LIBS})

add_executable(test_recursive_vcycle test_recursive_vcycle.cpp)
target_link_libraries(test_recursive_vcycle qdpxx_utils gtest_all mg_test mg ${EXT_LIBS})

add_executable(test_eo_bits test_eo_bits.cpp)
target_link_libraries(test_eo_bits qdpxx_utils gtest_all mg_test mg ${EXT_LIBS})

if( MG_QDPXX_PARALLEL ) 
  add_subdirectory( qdpxx_parallel ) 
endif (MG_QDPXX_PARALLEL ) 


add_test( NAME CoarseQDPXXSite COMMAND ./test_coarse_qdpxx_site -geom 1 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME CoarseQDPXXBlock COMMAND ./test_coarse_qdpxx -geom 1 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME QDPXXCloverLinop COMMAND ./test_qdpxx_clover_linop -geom 1 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME QDPXXCoarseCloverLinop COMMAND ./test_qdpxx_coarse_clover_op -geom 1 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME QDPXXCoarseVCycle COMMAND ./test_vcycle -geom 1 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME CoarseCoarseBlocking COMMAND ./test_coarse_coarse_blocking -geom 1 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME RecursiveVCycle COMMAND ./test_recursive_vcycle -geom 1 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestEOBits COMMAND ./test_eo_bits -geom 1 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
