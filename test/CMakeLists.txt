
# this makes the executable out of my test_file and the 'gtest_all.cc' file
# and specifies linking to the 'libmg.a' library
add_compile_options(${EXT_CXXFLAGS})

set(TEST_ENV_FILES test_env.h)

add_library(mg_test test_env.h test_env.cpp)
target_compile_options(mg_test PUBLIC ${EXT_CXXFLAGS})


add_executable(test_lattice test_lattice.cpp)
target_link_libraries(test_lattice mg gtest_all mg_test ${EXT_LIBS})

add_executable(test_logging test_logging.cpp)
target_link_libraries(test_logging mg gtest_all mg_test ${EXT_LIBS})

add_executable(test_nodeinfo test_nodeinfo.cpp)
target_link_libraries(test_nodeinfo mg gtest_all mg_test ${EXT_LIBS})

add_executable(test_memory test_memory.cpp)
target_link_libraries(test_memory mg gtest_all mg_test ${EXT_LIBS})

add_executable(test_cmat_mult test_cmat_mult.cpp)
target_link_libraries(test_cmat_mult mg gtest_all mg_test ${EXT_LIBS})

add_executable(test_coarse test_coarse.cpp)
target_link_libraries(test_coarse mg gtest_all mg_test ${EXT_LIBS})

if( MG_USE_QDPXX ) 
  add_subdirectory(qdpxx)
endif (MG_USE_QDPXX ) 

if( MG_USE_KOKKOS )
  add_subdirectory(kokkos)
endif()

if( QPhiX_FOUND )
  add_subdirectory(qphix)
endif()

add_test( NAME TestLattice COMMAND ./test_lattice -geom 1 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS} )
add_test( NAME TestLogging COMMAND ./test_logging -geom 1 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestNodeInfo COMMAND ./test_nodeinfo -geom 1 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestMemory COMMAND ./test_memory -geom 1 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME CMatMult COMMAND ./test_cmat_mult -geom 1 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME CoarseOp COMMAND ./test_coarse -geom 1 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})