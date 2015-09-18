cmake_minimum_required (VERSION 2.6)
project (MG)

#A Version Number
set (MG_VERSION_MAJOR 1)
set (MG_VERSION_MINOR 0)


# Add a config.h like file to pass some of the
# CMake settings to the source

configure_file(
 "${PROJECT_SOURCE_DIR}/MG_config.h.in" 
 "${PROJECT_BINARY_DIR}/MG_config.h"
)

# Install the config file
install (FILES "${PROJECT_BINARY_DIR}/MG_config.h" DESTINATION include)

# Add the include directories on the search path
include_directories(include)
include_directories("${PROJECT_BINARY_DIR}")
include_directories(external)

# Add the include subdirectory for installation processing
add_subdirectory(include)

# Add the lib subdirectory to make the lib
add_subdirectory(lib)
add_subdirectory(test)

enable_testing()
add_test ( TestLattice test/test_lattice )
