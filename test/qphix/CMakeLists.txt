# this makes the executable out of my test_file and the 'gtest_all.cc' file
# and specifies linking to the 'libmg.a' library



add_library(qphix_testutils qphix_testutils.h qphix_testutils.cpp)
target_compile_options(qphix_testutils PUBLIC ${EXT_CXXFLAGS})
add_compile_options(${EXT_CXXFLAGS})
			
add_executable(test_qphix_cli test_qphix_cli.cpp)
target_link_libraries(test_qphix_cli qphix_testutils qdpxx_utils  mg_test mg ${QPhiX_LIBRARIES} ${EXT_LIBS} gtest_all)

add_executable(test_qphix_interface test_qphix_interface.cpp)
target_link_libraries(test_qphix_interface qphix_testutils qdpxx_utils  mg_test mg ${QPhiX_LIBRARIES} ${EXT_LIBS} gtest_all)

add_executable(test_qphix_blas test_qphix_blas.cpp)
target_link_libraries(test_qphix_blas qphix_testutils qdpxx_utils  mg_test mg ${QPhiX_LIBRARIES} ${EXT_LIBS} gtest_all)

add_executable(test_qphix_setup test_qphix_setup.cpp)
target_link_libraries(test_qphix_setup qphix_testutils qdpxx_utils  mg_test mg ${QPhiX_LIBRARIES} ${EXT_LIBS} gtest_all)

add_executable(test_qphix_vcycle test_qphix_vcycle.cpp)
target_link_libraries(test_qphix_vcycle qphix_testutils qdpxx_utils  mg_test mg ${QPhiX_LIBRARIES} ${EXT_LIBS} gtest_all)

add_executable(test_qphix_recursive_vcycle test_qphix_recursive_vcycle.cpp)
target_link_libraries(test_qphix_recursive_vcycle qphix_testutils qdpxx_utils  mg_test mg ${QPhiX_LIBRARIES} ${EXT_LIBS} gtest_all)

add_executable(recursive_vcycle_profile recursive_vcycle_profile.cpp)
target_link_libraries(recursive_vcycle_profile qphix_testutils qdpxx_utils  mg_test mg ${QPhiX_LIBRARIES} ${EXT_LIBS} gtest_all)

add_executable(coarse_op_profile coarse_op_profile.cpp)
target_link_libraries(coarse_op_profile qphix_testutils qdpxx_utils  mg_test mg ${QPhiX_LIBRARIES} ${EXT_LIBS} gtest_all)

add_executable(restrictor_profile restrictor_profile.cpp)
target_link_libraries(restrictor_profile qphix_testutils qdpxx_utils  mg_test mg ${QPhiX_LIBRARIES} ${EXT_LIBS} gtest_all)

add_executable(prolongator_profile prolongator_profile.cpp)
target_link_libraries(prolongator_profile qphix_testutils qdpxx_utils  mg_test mg ${QPhiX_LIBRARIES} ${EXT_LIBS} gtest_all)

add_executable(even_odd_pieces even_odd_pieces.cpp)
target_link_libraries(even_odd_pieces qphix_testutils qdpxx_utils mg_test mg ${QPhiX_LIBRARIES} ${EXT_LIBS} gtest_all)

add_executable(test_coarse_eo_solvers test_coarse_eo_solvers.cpp)
target_link_libraries(test_coarse_eo_solvers qphix_testutils qdpxx_utils mg_test mg ${QPhiX_LIBRARIES} ${EXT_LIBS} gtest_all)

target_link_libraries(even_odd_pieces qphix_testutils qdpxx_utils mg_test mg ${QPhiX_LIBRARIES} ${EXT_LIBS} gtest_all)

add_executable(test_qphix_vcycle_eo test_qphix_vcycle_eo.cpp)
target_link_libraries(test_qphix_vcycle_eo qphix_testutils qdpxx_utils  mg_test mg ${QPhiX_LIBRARIES} ${EXT_LIBS} gtest_all)


add_test( NAME QPhixCLI COMMAND env OMP_NUM_THREADS=1 ./test_qphix_cli -geom 1 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME QPhixInterfaceTests COMMAND env OMP_NUM_THREADS=1 ./test_qphix_interface -geom 1 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME QPhixBLASTests COMMAND  env OMP_NUM_THREADS=1 ./test_qphix_blas -geom 1 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME QPhiXBLASTestsParallel COMMAND env OMP_NUM_THREADS=1  mpirun -n 2 -host localhost ./test_qphix_blas -geom 1 1 1 2 ${DEFAULT_QPHIX_TEST_ARGS})

add_test( NAME QPhiXSetupTests COMMAND  env OMP_NUM_THREADS=1 ./test_qphix_setup -geom 1 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME QPhiXSetupTestsThreaded COMMAND env OMP_NUM_THREADS=8 ./test_qphix_setup -geom 1 1 1 1 ${THREADED_QPHIX_TEST_ARGS})

add_test( NAME QPhiXVCycleTests COMMAND env OMP_NUM_THREADS=1  ./test_qphix_vcycle -geom 1 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME QPhiXVCycleTestsThreaded COMMAND env OMP_NUM_THREADS=8 ./test_qphix_vcycle -geom 1 1 1 1 ${THREADED_QPHIX_TEST_ARGS})

add_test( NAME QPhiXRecursiveVCycleTests COMMAND  env OMP_NUM_THREADS=1 ./test_qphix_recursive_vcycle -geom 1 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME QPhiXRecursiveVCycleTestsThreaded COMMAND env OMP_NUM_THREADS=8 ./test_qphix_recursive_vcycle -geom 1 1 1 1 ${THREADED_QPHIX_TEST_ARGS})

add_test( NAME CoarseEOSolversQPhixSetup COMMAND env OMP_NUM_THREADS=1  mpirun -n 2 -host localhost ./test_coarse_eo_solvers -geom 1 1 1 2 ${DEFAULT_QPHIX_TEST_ARGS})
