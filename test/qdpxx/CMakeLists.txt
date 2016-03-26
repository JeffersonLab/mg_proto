
# this makes the executable out of my test_file and the 'gtest_all.cc' file
# and specifies linking to the 'libmg.a' library
add_compile_options(${EXT_CXXFLAGS})
add_library(qdpxx_helpers qdpxx_helpers.h reunit.h reunit.cpp transf.h transf.cpp dslashm_w.h dslashm_w.cpp qdpxx_helpers.cpp )
target_compile_options(mg_test PUBLIC ${EXT_CXXFLAGS})

add_executable(test_qdpxx test_qdpxx.cpp)
target_link_libraries(test_qdpxx  qdpxx_helpers mg gtest_all mg_test ${EXT_LIBS})

add_executable(test_coarse_qdpxx test_coarse_qdpxx.cpp)
target_link_libraries(test_coarse_qdpxx  qdpxx_helpers mg gtest_all mg_test ${EXT_LIBS})
