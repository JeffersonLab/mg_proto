if ( QMP_COMMS ) 
	set(NODEINFO_IMPL  lattice/nodeinfo_qmp.cpp)
else( QMP_COMMS )
	set(NODEINFO_IMPL  lattice/nodeinfo_single.cpp)
endif( QMP_COMMS ) 


add_library(mg lattice/lattice_info.cpp
			   lattice/nodeinfo.cpp
			   ${NODEINFO_IMPL}
			   utils/print_utils.cpp)
			   
target_compile_options(mg PUBLIC ${EXT_CXXFLAGS})
install(TARGETS mg DESTINATION lib)