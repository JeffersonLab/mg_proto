
# this makes the executable out of my test_file and the 'gtest_all.cc' file

add_compile_options(${EXT_CXXFLAGS})


add_executable(test_lattice_parallel test_lattice_parallel.cpp)
target_link_libraries(test_lattice_parallel qdpxx_helpers  mg gtest_all mg_test ${EXT_LIBS})


add_test( NAME TestLatticeParallelT COMMAND mpirun -n 2 ./test_lattice_parallel -geom 1 1 1 2 )
add_test( NAME TestLatticeParallelZ COMMAND mpirun -n 2 ./test_lattice_parallel -geom 1 1 2 1 )
add_test( NAME TestLatticeParallelZT COMMAND mpirun -n 4 ./test_lattice_parallel -geom 1 1 2 2 )
add_test( NAME TestLatticeParallelY COMMAND mpirun -n 2 ./test_lattice_parallel -geom 1 2 1 1 )
add_test( NAME TestLatticeParallelYT COMMAND mpirun -n 4 ./test_lattice_parallel -geom 1 2 1 2 )
add_test( NAME TestLatticeParallelYZT COMMAND mpirun -n 8 ./test_lattice_parallel -geom 1 2 2 2 )
add_test( NAME TestLatticeParallelX COMMAND mpirun -n 2 ./test_lattice_parallel -geom 2 1 1 1 )
add_test( NAME TestLatticeParallelXT COMMAND mpirun -n 4 ./test_lattice_parallel -geom 2 1 1 2 )
add_test( NAME TestLatticeParallelXZT COMMAND mpirun -n 8 ./test_lattice_parallel -geom 2 1 2 2 )
add_test( NAME TestLatticeParallelXYZT COMMAND mpirun -n 16 ./test_lattice_parallel -geom 2 2 2 2 )
add_test( NAME TestLatticeParallelT4 COMMAND mpirun -n 4 ./test_lattice_parallel -geom 1 1 1 4 )
