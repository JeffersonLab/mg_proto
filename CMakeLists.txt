cmake_minimum_required (VERSION 3.0)
project (MG)

#A Version Number
set (MG_VERSION_MAJOR 1)
set (MG_VERSION_MINOR 0)

set (MG_DEFAULT_LOGLEVEL DEBUG CACHE STRING "Default Loglevel")


if( QMP_DIR ) 
  set( CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" "${QMP_DIR}")
endif( QMP_DIR ) 

# Find package QMP
find_package( QMP REQUIRED )

# Check for QMP
if ( ${QMP_C_FOUND} )
  include_directories(${QMP_C_INCLUDE_PATH})
  LIST(APPEND EXT_LIBS ${QMP_C_LINK_FLAGS} )
  LIST(APPEND EXT_LIBS ${QMP_C_LIBRARIES} )
  set(QMP_COMMS 1)
endif( ${QMP_C_FOUND} )

find_package( MPI REQUIRED )

# Deal with threads 
find_package(Threads REQUIRED)
if(THREADS_HAVE_PTHREAD_ARG)
  LIST(APPEND EXT_CXXFLAGS "-pthread")
endif()

if(CMAKE_THREAD_LIBS_INIT)
   LIST(APPEND EXT_LIBS ${CMAKE_THREAD_LIBS_INIT} )
endif()


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
add_subdirectory(external)
add_subdirectory(lib)
add_subdirectory(test)

enable_testing()
add_test ( NAME TestLattice COMMAND  test/test_lattice )
