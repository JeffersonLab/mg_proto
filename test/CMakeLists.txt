
# this makes the executable out of my test_file and the 'gtest_all.cc' file
# and specifies linking to the 'libmg.a' library
add_compile_options(${EXT_CXXFLAGS})

add_library(mg_test qmp_test_env.cpp)
target_compile_options(mg_test PUBLIC ${EXT_CXXFLAGS})


add_executable(test_lattice qmp_test_env.h  test_lattice.cpp)
target_link_libraries(test_lattice mg gtest_all mg_test ${EXT_LIBS})

add_executable(test_logging qmp_test_env.h test_logging.cpp)
target_link_libraries(test_logging mg gtest_all mg_test ${EXT_LIBS})

add_executable(test_nodeinfo qmp_test_env.h test_nodeinfo.cpp)
target_link_libraries(test_nodeinfo mg gtest_all mg_test ${EXT_LIBS})