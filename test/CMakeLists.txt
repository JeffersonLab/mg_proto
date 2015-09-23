
# this makes the executable out of my test_file and the 'gtest_all.cc' file
# and specifies linking to the 'libmg.a' library

add_executable(test_lattice test_lattice.cpp
"${CMAKE_SOURCE_DIR}/external/gtest/gtest-all.cc")

target_link_libraries(test_lattice mg)
add_compile_options(${EXT_CXXFLAGS})
target_link_libraries(test_lattice ${EXT_LIBS})


