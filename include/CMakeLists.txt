install (FILES lattice/array2d.h
			   lattice/cmat_mult.h
               lattice/coarse/aggregate_block_coarse.h
               lattice/coarse/block.h
               lattice/coarse/coarse_l1_blas.h
               lattice/coarse/coarse_op.h
               lattice/coarse/coarse_types.h 
               lattice/coarse/coarse_wilson_clover_linear_operator.h
               lattice/coarse/thread_limits.h
               lattice/coarse/vcycle_coarse.h
			   lattice/constants.h
			   lattice/fgmres_common.h 
			   lattice/geometry_utils.h 
			   lattice/invbicgstab_coarse.h
			   lattice/invfgmres_coarse.h
			   lattice/invmr_coarse.h
               lattice/lattice_info.h 
			   lattice/linear_operator.h
               lattice/mr_params.h
               lattice/nodeinfo.h
			   lattice/solver.h               
         DESTINATION include/lattice)
         
install (FILES utils/initialize.h
			   utils/memory.h 
			   utils/print_utils.h 
		 DESTINATION include/utils)
		 
install (FILES fine_qdpxx/aggregate_block_qdpxx.h
			   fine_qdpxx/aggregate_qdpxx.h
			   fine_qdpxx/clover_fermact_params_w.h
			   fine_qdpxx/clover_term_qdp_w.h
			   fine_qdpxx/dslashm_w.h
			   fine_qdpxx/invbicgstab.h
			   fine_qdpxx/infgmres.h
			   fine_qdpxx/invmr.h
			   fine_qdpxx/mesfield.h
			   fine_qdpxx/qdpxx_helpers.h
			   fine_qdpxx/transf.h
			   fine_qdpxx/vcycle_qdpxx_coarse.h
			   fine_qdpxx/wilson_clover_linear_operator.h
		 DESTINATION include/fine_qdpxx)
            
