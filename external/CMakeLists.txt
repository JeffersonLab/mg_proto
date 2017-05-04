add_library(gtest_all gtest/gtest-all.cc)

target_compile_options(gtest_all PUBLIC ${EXT_CXXFLAGS})
