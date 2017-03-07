LIST(APPEND library_source_list lattice/aggregate_block_coarse.cpp
			   lattice/block.cpp
			   lattice/cmat_mult.cpp
			   lattice/coarse_l1_blas.cpp
			   lattice/coarse_op.cpp
			   lattice/lattice_info.cpp
			   lattice/mg_setup.cpp
			   lattice/nodeinfo.cpp
			   utils/initialize.cpp
			   utils/print_utils.cpp
			   utils/memory.cpp)

if ( MG_QMP_COMMS OR MG_QDPXX_PARALLEL ) 
	LIST(APPEND library_source_list lattice/nodeinfo_qmp.cpp)
	LIST(APPEND library_source_list lattice/spinor_halo_qmp.cpp)
	add_definitions(-DMG_USE_QMP_SPINOR_HALO)
else( MG_QMP_COMMS  OR MG_QDPXX_PARALLEL)
	LIST(APPEND library_source_list lattice/nodeinfo_single.cpp)
	add_definitions(-DMG_USE_SINGLE_PARALLEL_HALO)
endif( MG_QMP_COMMS OR MG_QDPXX_PARALLEL) 

LIST(APPEND library_source_list utils/memory_posix.cpp)

add_library(mg ${library_source_list})
			   
			   
			   			   
target_compile_options(mg PUBLIC ${EXT_CXXFLAGS})
install(TARGETS mg DESTINATION lib)