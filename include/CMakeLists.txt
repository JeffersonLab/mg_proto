install (FILES lattice/aggregation.h
			   lattice/coarsen.h
			   lattice/constants.h 
               lattice/lattice_info.h 
               lattice/nodeinfo.h
               lattice/geometry_utils.h
               lattice/layout_container.h
               lattice/cb_soa_layout.h
               lattice/compact_cb_aos_spinor_layout.h
               lattice/cmat_mult.h               
         DESTINATION include/lattice)
         
install (FILES utils/print_utils.h
			   utils/memory.h 
			   utils/initialize.h DESTINATION include/utils)
            
