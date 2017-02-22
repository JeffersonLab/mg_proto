if ( MG_QMP_COMMS ) 
	set(NODEINFO_IMPL  lattice/nodeinfo_qmp.cpp)
else( MG_QMP_COMMS )
	set(NODEINFO_IMPL  lattice/nodeinfo_single.cpp)
endif( MG_QMP_COMMS ) 

set( MEMORY_IMPL utils/memory_posix.cpp)

add_library(mg lattice/lattice_info.cpp
			   lattice/nodeinfo.cpp
			   lattice/cmat_mult.cpp
			   lattice/block.cpp
			   lattice/coarse_op.cpp
			   lattice/coarse_l1_blas.cpp
			   ${NODEINFO_IMPL}
			   utils/print_utils.cpp
			   utils/memory.cpp
			   ${MEMORY_IMPL}
			   utils/initialize.cpp
			   lattice/mg_setup.cpp)
			   
target_compile_options(mg PUBLIC ${EXT_CXXFLAGS})
install(TARGETS mg DESTINATION lib)