
# this makes the executable out of my test_file and the 'gtest_all.cc' file
# and specifies linking to the 'libmg.a' library
add_compile_options(${EXT_CXXFLAGS})
add_library(qdpxx_helpers qdpxx_helpers.h reunit.h reunit.cpp transf.h transf.cpp dslashm_w.h dslashm_w.cpp qdpxx_helpers.cpp clover_fermact_params_w.h clover_term_qdp_w.h mesfield.h mesfield.cpp)
add_library(test_aggregation aggregate_qdpxx.h aggregate_qdpxx.cpp aggregate_block_qdpxx.h aggregate_block_qdpxx.cpp)
target_compile_options(mg_test PUBLIC ${EXT_CXXFLAGS})

add_executable(test_qdpxx test_qdpxx.cpp)
target_link_libraries(test_qdpxx  qdpxx_helpers mg gtest_all mg_test ${EXT_LIBS})

add_executable(test_coarse_qdpxx test_coarse_qdpxx.cpp)
target_link_libraries(test_coarse_qdpxx  qdpxx_helpers test_aggregation mg gtest_all mg_test ${EXT_LIBS})

add_executable(test_coarse_qdpxx_site test_coarse_qdpxx_site.cpp)
target_link_libraries(test_coarse_qdpxx_site  qdpxx_helpers test_aggregation mg gtest_all mg_test ${EXT_LIBS})

add_executable(test_qdpxx_clover_linop test_qdpxx_clover_linop.cpp)
target_link_libraries(test_qdpxx_clover_linop  qdpxx_helpers test_aggregation mg gtest_all mg_test ${EXT_LIBS})

add_executable(test_qdpxx_coarse_clover_op test_qdpxx_coarse_clover_op.cpp)
target_link_libraries(test_qdpxx_coarse_clover_op  qdpxx_helpers test_aggregation mg gtest_all mg_test ${EXT_LIBS})

add_executable(test_vcycle test_vcycle.cpp)
target_link_libraries(test_vcycle  qdpxx_helpers test_aggregation mg gtest_all mg_test ${EXT_LIBS})