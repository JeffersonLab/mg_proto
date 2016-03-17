
# this makes the executable out of my test_file and the 'gtest_all.cc' file
# and specifies linking to the 'libmg.a' library
add_compile_options(${EXT_CXXFLAGS})
add_library(qdpxx_helpers qdpxx_helpers.h qdpxx_helpers.cpp )
target_compile_options(mg_test PUBLIC ${EXT_CXXFLAGS})
add_executable(test_qdpxx test_qdpxx.cpp)
target_link_libraries(test_qdpxx  qdpxx_helpers mg gtest_all mg_test ${EXT_LIBS})

