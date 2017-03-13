
# this makes the executable out of my test_file and the 'gtest_all.cc' file

add_compile_options(${EXT_CXXFLAGS})


add_executable(test_lattice_parallel test_lattice_parallel.cpp)
target_link_libraries(test_lattice_parallel qdpxx_helpers  mg gtest_all mg_test ${EXT_LIBS})

add_executable(test_halo test_halo.cpp)
target_link_libraries(test_halo qdpxx_helpers  mg gtest_all mg_test ${EXT_LIBS})

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
add_test( NAME TestHaloT COMMAND mpirun -n 2 ./test_halo -geom 1 1 1 2 )
add_test( NAME TestHaloZ COMMAND mpirun -n 2 ./test_halo -geom 1 1 2 1 )
add_test( NAME TestHaloY COMMAND mpirun -n 2 ./test_halo -geom 1 2 1 1 )
add_test( NAME TestHaloX COMMAND mpirun -n 2 ./test_halo -geom 2 1 1 1  )
add_test( NAME TestHaloZT COMMAND mpirun -n 4 ./test_halo -geom 1 1 2 2 )
add_test( NAME TestHaloYT COMMAND mpirun -n 4 ./test_halo -geom 1 2 1 2 )
add_test( NAME TestHaloXT COMMAND mpirun -n 4 ./test_halo -geom 2 1 1 2 )
add_test( NAME TestHaloXZT COMMAND mpirun -n 8 ./test_halo -geom 2 1 2 2 )
add_test( NAME TestHaloXYT COMMAND mpirun -n 8 ./test_halo -geom 2 2 1 2 )
add_test( NAME TestHaloXYZT COMMAND mpirun -n 16 ./test_halo -geom 2 2 2 2 )
