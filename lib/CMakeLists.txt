add_library(mg lattice/lattice.cpp)
target_compile_options(mg PUBLIC ${EXT_CXXFLAGS})
install(TARGETS mg DESTINATION lib)

