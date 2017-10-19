install (FILES lattice/array2d.h
			   lattice/cmat_mult.h
			   lattice/constants.h
			   lattice/givens.h
			   lattice/fgmres_common.h 
			   lattice/geometry_utils.h   
			   lattice/invfgmres_generic.h
         lattice/lattice_info.h 
			   lattice/linear_operator.h
			   lattice/mg_level_coarse.h			   
         lattice/mr_params.h
         lattice/nodeinfo.h
			   lattice/solver.h  
			   lattice/spinor_halo_qmp.h
			   lattice/spinor_halo_single.h
			   lattice/spinor_halo.h             
         DESTINATION include/lattice)
         
install (FILES lattice/coarse/aggregate_block_coarse.h
               lattice/coarse/block.h
               lattice/coarse/coarse_l1_blas.h
               lattice/coarse/coarse_op.h
               lattice/coarse/coarse_types.h 
               lattice/coarse/coarse_wilson_clover_linear_operator.h
               lattice/coarse/invbicgstab_coarse.h
               lattice/coarse/invfgmres_coarse.h
               lattice/coarse/invmr_coarse.h
               lattice/coarse/thread_limits.h
               lattice/coarse/vcycle_coarse.h
         DESTINATION include/lattice/coarse)
         
install (FILES utils/initialize.h
			   utils/memory.h 
			   utils/print_utils.h 
		 DESTINATION include/utils)
		 
install (FILES lattice/fine_qdpxx/aggregate_block_qdpxx.h
			   lattice/fine_qdpxx/aggregate_qdpxx.h
			   lattice/fine_qdpxx/clover_fermact_params_w.h
			   lattice/fine_qdpxx/clover_term_qdp_w.h
			   lattice/fine_qdpxx/dslashm_w.h
			   lattice/fine_qdpxx/invbicgstab_qdpxx.h
			   lattice/fine_qdpxx/invfgmres_qdpxx.h
			   lattice/fine_qdpxx/invmr_qdpxx.h
			   lattice/fine_qdpxx/mesfield.h
			   lattice/fine_qdpxx/mg_level_qdpxx.h
			   lattice/fine_qdpxx/mg_params_qdpxx.h
			   lattice/fine_qdpxx/qdpxx_helpers.h
			   lattice/fine_qdpxx/transf.h
			   lattice/fine_qdpxx/vcycle_qdpxx_coarse.h
			   lattice/fine_qdpxx/vcycle_recursive_qdpxx.h
			   lattice/fine_qdpxx/wilson_clover_linear_operator.h
		 DESTINATION include/lattice/fine_qdpxx)
            
install (FILES lattice/qphix/qphix_types.h
						   lattice/qphix/qphix_veclen.h
						   lattice/qphix/qphix_qdp_utils.h
						   lattice/qphix/qphix_clover_linear_operator.h
						   lattice/qphix/qphix_blas_wrappers.h
						   lattice/qphix/invfgmres_qphix.h
						   lattice/qphix/invbicgstab_qphix.h
						   lattice/qphix/invmr_qphix.h
				 DESTINATION include/lattice/qphix)
				 