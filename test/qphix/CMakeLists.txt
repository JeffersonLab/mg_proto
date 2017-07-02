# this makes the executable out of my test_file and the 'gtest_all.cc' file
# and specifies linking to the 'libmg.a' library



add_compile_options(${EXT_CXXFLAGS})
			
add_executable(test_qphix_cli test_qphix_cli)
target_link_libraries(test_qphix_cli qdpxx_utils  mg_test mg ${QPhiX_LIBRARIES} ${EXT_LIBS} gtest_all)

add_test( NAME QPhixCLI COMMAND ./test_qphix_cli -geom 1 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})