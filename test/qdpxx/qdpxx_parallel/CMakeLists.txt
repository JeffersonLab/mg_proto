
# this makes the executable out of my test_file and the 'gtest_all.cc' file

add_compile_options(${EXT_CXXFLAGS})


add_executable(test_lattice_parallel test_lattice_parallel.cpp)
target_link_libraries(test_lattice_parallel qdpxx_helpers  mg gtest_all mg_test ${EXT_LIBS})