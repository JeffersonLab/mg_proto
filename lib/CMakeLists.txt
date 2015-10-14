if ( QMP_COMMS ) 
	set(NODEINFO_IMPL  lattice/nodeinfo_qmp.cpp)
else( QMP_COMMS )
	set(NODEINFO_IMPL  lattice/nodeinfo_single.cpp)
endif( QMP_COMMS ) 

set( MEMORY_IMPL utils/memory_posix.cpp)

add_library(mg lattice/coarsen.cpp
			   lattice/lattice_info.cpp
			   lattice/nodeinfo.cpp
			   ${NODEINFO_IMPL}
			   utils/print_utils.cpp
			   utils/memory.cpp
			   ${MEMORY_IMPL}
			   utils/initialize.cpp)
			   
target_compile_options(mg PUBLIC ${EXT_CXXFLAGS})
install(TARGETS mg DESTINATION lib)