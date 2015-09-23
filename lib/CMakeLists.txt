add_library(mg lattice/lattice.cpp
							 utils/print_utils.cpp)
target_compile_options(mg PUBLIC ${EXT_CXXFLAGS})
install(TARGETS mg DESTINATION lib)

