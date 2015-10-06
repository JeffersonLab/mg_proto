
# this makes the executable out of my test_file and the 'gtest_all.cc' file
# and specifies linking to the 'libmg.a' library
add_compile_options(${EXT_CXXFLAGS})

set(TEST_ENV_FILES test_env.h)

if( QMP_DIR ) 
  list(APPEND TEST_ENV_FILES qmp_test_env.h qmp_test_env.cpp)
else( QMP_DIR )
  list(APPEND TEST_ENV_FILES scalar_test_env.cpp)
endif( QMP_DIR )

add_library(mg_test test_env.h ${TEST_ENV_FILES} )
target_compile_options(mg_test PUBLIC ${EXT_CXXFLAGS})


add_executable(test_lattice qmp_test_env.h  test_lattice.cpp)
target_link_libraries(test_lattice mg gtest_all mg_test ${EXT_LIBS})

add_executable(test_logging qmp_test_env.h test_logging.cpp)
target_link_libraries(test_logging mg gtest_all mg_test ${EXT_LIBS})

add_executable(test_nodeinfo qmp_test_env.h test_nodeinfo.cpp)
target_link_libraries(test_nodeinfo mg gtest_all mg_test ${EXT_LIBS})