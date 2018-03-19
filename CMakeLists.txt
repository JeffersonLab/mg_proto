cmake_minimum_required (VERSION 3.1)

project (MG)
include (CTest)

#A Version Number
set (MG_VERSION_MAJOR 1)
set (MG_VERSION_MINOR 0)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules/")

set (MG_DEFAULT_LOGLEVEL DEBUG CACHE STRING "Default Loglevel")
set (MG_USE_AVX512 CACHE BOOL FALSE)
set (MG_KOKKOS_DISPATCH CACHE STRING "FALSE")
set (MG_KOKKOS_USE_NEIGHBOR_TABLE CACHE BOOL TRUE)

if(MG_DEFAULT_ALIGNMENT)
  message(STATUS "User specified default alignment: ${MG_DEFAULT_ALIGNMENT}")
else(MG_DEFAULT_ALIGNMENT)
  message(STATUS "Setting Default alignment to 64")
  set (MG_DEFAULT_ALIGNMENT 64)
endif(MG_DEFAULT_ALIGNMENT )

if(MG_DEFAULT_CACHE_LINE_SIZE)
  message(STATUS "User specified default alignment: ${MG_DEFAULT_CACHE_LINE_SIZE}")
else(MG_DEFAULT_CACHE_LINE_SIZE)
  message(STATUS "Setting Default Cache Line Size to 64")
  set (MG_DEFAULT_CACHE_LINE_SIZE 64)
endif(MG_DEFAULT_CACHE_LINE_SIZE )

if( MG_KOKKOS_USE_NEIGHBOR_TABLE ) 
  message(STATUS "KOKKOS DSlash will use neighbor table")
else()
  message(STATUS "KOKKOS Dslash will not use neithbor table")
endif()


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
# list(APPEND CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -O3)

if( MG_USER_CXXFLAGS ) 
 list(APPEND CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${MG_USER_CXXFLAGS})
endif()

if ( MG_USE_KOKKOS )

  add_subdirectory(${CMAKE_SOURCE_DIR}/kokkos)
  include_directories(${Kokkos_SOURCE_DIR}/core/src)
  include_directories(${Kokkos_SOURCE_DIR}/containers/src)
  include_directories(${Kokkos_SOURCE_DIR}/algorithms/src)
  include_directories(${Kokkos_BINARY_DIR})
endif()

if( MG_KOKKOS_USE_FLAT_DISPATCH )
  set(MG_KOKKOS_USE_TEAM_DISPATCH FALSE)
  set(MG_KOKKOS_USE_MDRANGE_DISPATCH FALSE)
elseif( MG_KOKKOS_USE_TEAM_DISPATCH ) 
  set(MG_KOKKOS_USE_FLAT_DISPATCH FALSE)
  set(MG_KOKKOS_USE_MDRANGE_DISPATCH FALSE)
elseif( MG_KOKKOS_USE_MDRANGE_DISPATCH ) 
  set(MG_KOKKOS_USE_FLAT_DISPATCH FALSE)
  set(MG_KOKKOS_USE_TEAM_DISPATCH FALSE)
else()
  set(MG_KOKKOS_USE_FLAT_DISPATCH TRUE)
  set(MG_KOKKOS_USE_TEAM_DISPATCH FALSE)
  set(MG_KOKKOS_USE_MDRANGE_DISPATCH FALSE)
endif()

if( QMP_DIR ) 
  message(STATUS "Looking for QMP")
  set( CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" "${QMP_DIR}")

  # Find package QMP
  find_package( QMP )

  # Check for QMP
  if ( QMP_C_FOUND )
    include_directories(${QMP_C_INCLUDE_PATH})
    LIST(APPEND EXT_LIBS ${QMP_C_LINK_FLAGS} )
    LIST(APPEND EXT_LIBS ${QMP_C_LIBRARIES} )
    set( MG_QMP_COMMS TRUE )
    set( MG_QMP_INIT  TRUE )
  endif( QMP_C_FOUND )
else( QMP_DIR )
  message(STATUS "Not Looking for QMP")
  set( MG_QMP_COMMS FALSE)
  set( MG_QMP_INIT FALSE)
endif( QMP_DIR )

if( VTUNE_DIR )
  SET( MG_USE_VTUNE)
  include_directories(${VTUNE_DIR}/include)
  LIST(APPEND EXT_LIBS "-L${VTUNE_DIR}/lib64 -littnotify")
endif()

if( FAKE_COMMS )
   message(STATUS "Setting FAKE_COMMS")
   set( MG_FAKE_COMMS TRUE)
endif( FAKE_COMMS)

if( QDPXX_DIR ) 
  message(STATUS, "Looking for QDP++") 
  set( CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" "${QDPXX_DIR}")
  find_package( QDPXX )
  if( QDPXX_CXX_FOUND )
     include_directories(${QDPXX_CXX_INCLUDE_PATH})
     LIST(APPEND EXT_CXXFLAGS "${QDPXX_CXX_COMPILE_FLAGS}")
     LIST(REMOVE_DUPLICATES EXT_CXXFLAGS )
     LIST(APPEND EXT_LIBS ${QDPXX_CXX_LINK_FLAGS})
     LIST(APPEND EXT_LIBS ${QDPXX_CXX_LIBRARIES})
     set( MG_USE_QDPXX TRUE)
  endif( QDPXX_CXX_FOUND )
 else(QDPXX_DIR)
  message( STATUS "Not Looking for QDP++")
  set( MG_USE_QDPXX FALSE)
 endif(QDPXX_DIR)

if( MG_USE_QDPXX ) 
	if( ${QDPXX_PARALLEL_ARCH} STREQUAL "parscalar" )
  		message(STATUS "Enabling QDPXX_PARALLEL Tests")
  		set( MG_QDPXX_PARALLEL TRUE )
  		set( MG_QMP_COMMS TRUE )
	endif(${QDPXX_PARALLEL_ARCH} STREQUAL "parscalar" )
endif(MG_USE_QDPXX)

# Deal with threads 
find_package(Threads REQUIRED)

# Check if we are using QPhix
set( DEFAULT_QPHIX_TEST_ARGS "")
if( QPHIX_DIR )
  message(STATUS "Finding QPhiX")
  set( CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" "${QPHIX_DIR}")
  find_package(QPhiX REQUIRED)
  if( NOT QPhiX_FOUND )
     message( FATAL_ERROR "QPhiX was not successfully found")
  else()
     message( STATUS "QPhiX was successfully found")
  endif()
  include_directories(${QPhiX_INCLUDE_DIRS})
  set(MG_USE_QPHIX TRUE)
  
  set(DEFAULT_QPHIX_TEST_ARGS  -by 4 -bz 4 -pxy 0 -pxyz 0 -c 1 -sy 1 -sz 1 -minct 1)
  set(THREADED_QPHIX_TEST_ARGS -by 4 -bz 4 -pxy 0 -pxyz 0 -c 4 -sy 1 -sz 2 -minct 1)
  
endif()

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
