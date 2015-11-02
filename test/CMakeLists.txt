
# this makes the executable out of my test_file and the 'gtest_all.cc' file
# and specifies linking to the 'libmg.a' library
add_compile_options(${EXT_CXXFLAGS})

set(TEST_ENV_FILES test_env.h)

add_library(mg_test test_env.h test_env.cpp )
target_compile_options(mg_test PUBLIC ${EXT_CXXFLAGS})


add_executable(test_lattice test_lattice.cpp)
target_link_libraries(test_lattice mg gtest_all mg_test ${EXT_LIBS})

add_executable(test_logging test_logging.cpp)
target_link_libraries(test_logging mg gtest_all mg_test ${EXT_LIBS})

add_executable(test_nodeinfo test_nodeinfo.cpp)
target_link_libraries(test_nodeinfo mg gtest_all mg_test ${EXT_LIBS})

add_executable(test_memory test_memory.cpp)
target_link_libraries(test_memory mg gtest_all mg_test ${EXT_LIBS})

add_executable(test_spinor test_spinor.cpp)
target_link_libraries(test_spinor mg gtest_all mg_test ${EXT_LIBS})

add_executable(test_block_layout test_block_layout.cpp)
target_link_libraries(test_block_layout mg gtest_all mg_test ${EXT_LIBS})
