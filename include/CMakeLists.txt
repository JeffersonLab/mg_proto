install (FILES lattice/aggregation.h
			   lattice/coarsen.h
			   lattice/constants.h 
               lattice/lattice_info.h 
               lattice/nodeinfo.h
               lattice/geometry_utils.h
               lattice/cmat_mult.h        
               lattice/coarse/coarse_l1_blas.h
               lattice/coarse/coarse_op.h
               lattice/coarse/thread_limits.h
               lattice/coarse/coarse_types.h 
         DESTINATION include/lattice)
         
install (FILES utils/print_utils.h
			   utils/memory.h 
			   utils/initialize.h DESTINATION include/utils)
            
