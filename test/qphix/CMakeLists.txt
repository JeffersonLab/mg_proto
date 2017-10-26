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

add_test( NAME QPhixCLI COMMAND ./test_qphix_cli -geom 1 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME QPhixInterfaceTests COMMAND ./test_qphix_interface -geom 1 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME QPhixBLASTests COMMAND ./test_qphix_blas -geom 1 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME QPhiXBLASTestsParallel COMMAND mpirun -n 2 ./test_qphix_blas -geom 1 1 1 2 ${DEFAULT_QPHIX_TEST_ARGS})

add_test( NAME QPhiXSetupTests COMMAND ./test_qphix_setup -geom 1 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
