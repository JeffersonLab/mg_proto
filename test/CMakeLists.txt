
# this makes the executable out of my test_file and the 'gtest_all.cc' file
# and specifies linking to the 'libmg.a' library
add_compile_options(${EXT_CXXFLAGS})

add_executable(test_lattice test_lattice.cpp)
target_link_libraries(test_lattice mg gtest_all ${EXT_LIBS})

add_executable(test_logging test_logging.cpp)
target_link_libraries(test_logging mg gtest_all ${EXT_LIBS})
