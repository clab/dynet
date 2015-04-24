project(cnn)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -O3 -g")

enable_testing()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

function(find_cudnn)
  set(CUDNN_ROOT "" CACHE PATH "CUDNN root path")
  find_path(CUDNN_INCLUDE_DIRS cudnn.h
    PATHS ${CUDNN_ROOT}
          ${CUDNN_ROOT}/include
    DOC "CUDNN include path")
  find_library(CUDNN_LIBRARIES NAMES libcudnn.so
    PATHS ${CUDNN_ROOT}
          ${CUDNN_ROOT}/lib
          ${CUDNN_ROOT}/lib64
    DOC "CUDNN library path")
  if(CUDNN_INCLUDE_DIRS AND CUDNN_LIBRARIES)
    set(CUDNN_FOUND TRUE PARENT_SCOPE)
    message(STATUS "Found CUDNN (include: ${CUDNN_INCLUDE_DIRS}, library: ${CUDNN_LIBRARIES})")
    mark_as_advanced(CUDNN_INCLUDE_DIRS CUDNN_LIBRARIES)
  else()
    MESSAGE(FATAL_ERROR "Failed to find CUDNN in path: ${CUDNN_ROOT} (Did you set CUDNN_ROOT properly?)")
  endif()
endfunction()

# look for Boost
find_package(Boost COMPONENTS program_options serialization unit_test_framework REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
set(LIBS ${LIBS} ${Boost_LIBRARIES})

if(BACKEND)
  message("BACKEND: ${BACKEND}")
else()
  message("BACKEND not specified, defaulting to eigen.")
  set(BACKEND "eigen")
endif()

if(BACKEND MATCHES "^thpp$")
  set(WITH_THPP_BACKEND 1)
elseif(BACKEND MATCHES "^eigen$")
  set(WITH_EIGEN_BACKEND 1)
elseif(BACKEND MATCHES "^minerva$")
  set(WITH_MINVERA_BACKEND 1)
else()
  message(SEND_ERROR "BACKEND must be one of thpp, eigen, or minerva.")
endif()

# look for Minerva
if(WITH_MINERVA_BACKEND)
  find_package(Minerva REQUIRED)
  include_directories(${MINERVA_INCLUDE_DIR})
  if (ENABLE_CUDA)
    set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_ROOT})
    find_package(CUDA REQUIRED)
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
    add_definitions(-DHAS_CUDA)
    find_cudnn()
    include_directories(SYSTEM ${CUDNN_INCLUDE_DIRS})
  else ()
    message(STATUS "CUDA not enabled (use -DENABLE_CUDA=1)")
  endif()
endif(WITH_MINERVA_BACKEND)

# look for thpp
if(WITH_THPP_BACKEND)
  find_package(TH++ REQUIRED)
  find_package(TH REQUIRED)
  include_directories(${THPP_INCLUDE_DIR})
  include_directories(${TH_INCLUDE_DIR})
  set(LIBS ${LIBS} ${TH_LIBRARIES} ${THPP_LIBRARIES})
  if (ENABLE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
    find_package(THC REQUIRED)
    include_directories(${THC_INCLUDE_DIR})
  else()
    message(STATUS "CUDA not enabled (use -DENABLE_CUDA=1)")
  endif()
  FIND_PACKAGE(OpenMP)
  IF(OPENMP_FOUND)
    MESSAGE(STATUS "Compiling with OpenMP support")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  ENDIF(OPENMP_FOUND)
endif(WITH_THPP_BACKEND)

# look for Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.cmake ${CMAKE_CURRENT_BINARY_DIR}/config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(cnn)
add_subdirectory(examples)

