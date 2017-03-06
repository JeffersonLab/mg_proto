if ( MG_QMP_COMMS OR MG_QDPXX_PARALLEL ) 
	set(NODEINFO_IMPL  lattice/nodeinfo_qmp.cpp)
else( MG_QMP_COMMS OR MG_QDPXX_PARALLEL )
	set(NODEINFO_IMPL  lattice/nodeinfo_single.cpp)
endif( MG_QMP_COMMS OR MG_QDPXX_PARALLEL ) 

set( MEMORY_IMPL utils/memory_posix.cpp)

add_library(mg lattice/aggregate_block_coarse.cpp
			   lattice/block.cpp
			   lattice/cmat_mult.cpp
			   lattice/coarse_l1_blas.cpp
			   lattice/coarse_op.cpp
			   lattice/lattice_info.cpp
			   lattice/mg_setup.cpp
			   lattice/nodeinfo.cpp
			   ${NODEINFO_IMPL}
			   utils/initialize.cpp
			   utils/print_utils.cpp
			   utils/memory.cpp
			   ${MEMORY_IMPL})
			   			   
target_compile_options(mg PUBLIC ${EXT_CXXFLAGS})
install(TARGETS mg DESTINATION lib)