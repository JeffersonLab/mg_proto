LIST(APPEND library_source_list lattice/aggregate_block_coarse.cpp
			   lattice/block.cpp
			   lattice/cmat_mult.cpp
			   lattice/coarse_l1_blas.cpp
			   lattice/coarse_op.cpp
			   lattice/givens.cpp
			   lattice/invbicgstab_coarse.cpp
			   lattice/invmr_coarse.cpp
			   lattice/lattice_info.cpp
			   lattice/mg_level_coarse.cpp
			   lattice/nodeinfo.cpp
			   utils/initialize.cpp
			   utils/print_utils.cpp
			   utils/memory.cpp)


if ( MG_QMP_COMMS OR MG_QDPXX_PARALLEL ) 
	LIST(APPEND library_source_list lattice/nodeinfo_qmp.cpp)
else( MG_QMP_COMMS  OR MG_QDPXX_PARALLEL)
	LIST(APPEND library_source_list lattice/nodeinfo_single.cpp)
endif( MG_QMP_COMMS OR MG_QDPXX_PARALLEL) 

LIST(APPEND library_source_list utils/memory_posix.cpp)

if( MG_USE_QDPXX )
	LIST(APPEND library_source_list lattice/fine_qdpxx/aggregate_block_qdpxx.cpp
		lattice/fine_qdpxx/aggregate_qdpxx.cpp
		lattice/fine_qdpxx/dslashm_w.cpp
		lattice/fine_qdpxx/invbicgstab_qdpxx.cpp
		lattice/fine_qdpxx/invfgmres_qdpxx.cpp
		lattice/fine_qdpxx/invmr_qdpxx.cpp
		lattice/fine_qdpxx/mesfield.cpp
		lattice/fine_qdpxx/mg_setup_qdpxx.cpp
		lattice/fine_qdpxx/qdpxx_helpers.cpp 
		lattice/fine_qdpxx/transf.cpp
		lattice/fine_qdpxx/vcycle_recursive_qdpxx.cpp)
endif( MG_USE_QDPXX )

if ( QPhiX_FOUND )
	LIST(APPEND library_source_list lattice/qphix/geom_utils.cpp
		lattice/qphix/mg_level_qphix.cpp
		lattice/qphix/qphix_aggregate.cpp
		lattice/qphix/qphix_blas_wrappers.cpp
		lattice/qphix/qphix_qdp_utils.cpp
		)
endif()

add_library(mg ${library_source_list})

if (MG_USE_KOKKOS) 
 target_link_libraries(mg kokkos) 
endif()

	
if ( QPhiX_FOUND )
  target_link_libraries(mg ${QPhiX_LIBRARIES})
endif()

target_compile_options(mg PUBLIC ${EXT_CXXFLAGS})
install(TARGETS mg DESTINATION lib)

