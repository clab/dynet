project(dynet)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# DYNET uses Eigen which exploits modern CPU architectures. To get the
# best possible performance, the following are recommended:
#   1. use very recent versions of gcc or Clang to build
#   2. use very recent versions of Eigen (ideally the dev version)
#   3. try compiler options like -march=native or other architecture
#      flags (the compiler does not always make the best configuration
#      decisions without help)

# NOTE: This seems to be causing problems with linking before using
#       make install. It is allegedly preferred, but probably doesn't
#       suit our model of not installing the library most of the time.
set(CMAKE_MACOSX_RPATH 0)

function(find_mkl)
  set(MKL_ARCH intel64)
  find_path(MKL_INCLUDE_DIR mkl.h
            PATHS ${MKL_ROOT} ${MKL_ROOT}/include)
  find_library(MKL_CORE_LIB NAMES mkl_intel_lp64 mkl_intel_thread mkl_core
               PATHS ${MKL_ROOT} ${MKL_ROOT}/lib/${MKL_ARCH}
                     ${MKL_ROOT}/lib #OSX
               DOC "MKL core library path")

  find_library(MKL_COMPILER_LIB NAMES iomp5 libiomp5md
               PATHS ${MKL_ROOT} ${MKL_ROOT}/../compiler/lib/${MKL_ARCH}              #Windows
                     ${MKL_ROOT}/../compilers_and_libraries/linux/lib/${MKL_ARCH}_lin #Linux
                     ${MKL_ROOT}/../compilers_and_libraries/mac/lib                   #OSX
               DOC "MKL compiler lib (for threaded MKL)")

  if(MKL_INCLUDE_DIR AND MKL_CORE_LIB AND MKL_COMPILER_LIB)
    get_filename_component(MKL_CORE_LIB_DIR ${MKL_CORE_LIB} DIRECTORY)
    get_filename_component(MKL_COMPILER_LIB_DIR ${MKL_COMPILER_LIB} DIRECTORY)
    get_filename_component(MKL_COMPILER_LIB_FILE ${MKL_COMPILER_LIB} NAME)
    message(STATUS "Found MKL\n   * include: ${MKL_INCLUDE_DIR},\n   * core library dir: ${MKL_CORE_LIB_DIR},\n   * compiler library: ${MKL_COMPILER_LIB}")

    set(LIBS ${LIBS} mkl_rt ${MKL_COMPILER_LIB_FILE} PARENT_SCOPE)
    include_directories(${MKL_INCLUDE_DIR})
    link_directories(${MKL_CORE_LIB_DIR} ${MKL_COMPILER_LIB_DIR})
    set(MKL_LINK_DIRS ${MKL_CORE_LIB_DIR} ${MKL_COMPILER_LIB_DIR} PARENT_SCOPE) # Keeping this for python build
  else()
    message(FATAL_ERROR "Failed to find MKL in path: ${MKL_ROOT} (Did you set MKL_ROOT properly?)")
  endif()
endfunction()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

######## Cross-compiler, cross-platform options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_FAST_MATH")
if (MKL OR MKL_ROOT)
  if (DEFINED ENV{MKL_ROOT} AND NOT DEFINED MKL_ROOT)  # use env variable if not defined
    set(MKL_ROOT $ENV{MKL_ROOT})
  endif()
  find_mkl()  # sets include/lib directories and sets ${LIBS} needed for linking
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_USE_MKL_ALL")
endif()

# There are about 30 files compiled for GPU. We merge them into less
# for faster compilation. Set to 0 to mean no merging, or k to merge into k files. 
if(NOT DEFINED GPU_NUMFILES)
  if(MSVC)
    set(GPU_NUMFILES 1)   # MSVC does serial compilation of CUDA, so minimize overhead
  else()
    set(GPU_NUMFILES 4)   # Should probably be set to number of jobs in parallel make
  endif()
endif()


######## Platform-specific options
if(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX")   # Disable min/max macros in windef.h
endif()

######## Compiler-specific options
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W1 /MP")   # -Wall produces 20k warnings. Enable parallel compilation
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fno-finite-math-only -Wall -Wno-missing-braces -std=c++11 -g")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -fno-omit-frame-pointer")
  set(CMAKE_CXX_FLAGS_RELEASE "-funroll-loops -Ofast -march=native")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${PROJECT_SOURCE_DIR}/external/easyloggingpp/src)

function(find_cudnn)
  if (DEFINED ENV{CUDNN_ROOT} AND NOT DEFINED CUDNN_ROOT)  # use env variable if not defined
    set(CUDNN_ROOT $ENV{CUDNN_ROOT})
  elseif (DEFINED CUDA_TOOLKIT_ROOT_DIR AND NOT DEFINED CUDNN_ROOT)  # use env variable if not defined
    set(CUDNN_ROOT ${CUDA_TOOLKIT_ROOT_DIR})
  endif()
#  set(CUDNN_ROOT /usr/local/cuda CACHE PATH "CUDNN root path")
  find_path(CUDNN_INCLUDE_DIRS cudnn.h
    HINTS ${CUDNN_ROOT}
          ${CUDNN_ROOT}/include
    DOC "CUDNN include path")
  find_library(CUDNN_LIBRARIES NAMES libcudnn.so cudnn.lib
    PATHS ${CUDNN_ROOT}
          ${CUDNN_ROOT}/lib
          ${CUDNN_ROOT}/lib64
          ${CUDNN_ROOT}/lib/x64
    DOC "CUDNN library path")
  if(CUDNN_INCLUDE_DIRS AND CUDNN_LIBRARIES)
    set(CUDNN_FOUND TRUE PARENT_SCOPE)
    message(STATUS "Found CUDNN (include: ${CUDNN_INCLUDE_DIRS}, library: ${CUDNN_LIBRARIES})")
    mark_as_advanced(CUDNN_INCLUDE_DIRS CUDNN_LIBRARIES)
  else()
    MESSAGE(STATUS "Failed to find CUDNN in path: ${CUDNN_ROOT} (Did you set CUDNN_ROOT properly?)")
  endif()
endfunction()

# look for Boost
if(ENABLE_BOOST)
  message("-- Enabling Boost")
  if(DEFINED ENV{BOOST_ROOT})
    set(Boost_NO_SYSTEM_PATHS ON)
    if(DEFINED ${Boost_INCLUDE_DIR})
      get_filename_component(Boost_INCLUDE_DIR "${Boost_INCLUDE_DIR}" REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
    endif()
  endif()
  set(Boost_REALPATH ON)
  find_package(Boost COMPONENTS program_options regex serialization REQUIRED)
  message("-- Boost dir is " ${Boost_INCLUDE_DIR})
  include_directories(${Boost_INCLUDE_DIR})
  if(MSVC)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LIBPATH:${Boost_LIBRARY_DIRS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /LIBPATH:${Boost_LIBRARY_DIRS}")
  else()
    set(LIBS ${LIBS} ${Boost_LIBRARIES})
  endif()
endif()

if(BACKEND)
  message("-- BACKEND: ${BACKEND}")
else()
  message("-- BACKEND not specified, defaulting to eigen.")
  set(BACKEND "eigen")
endif()

if(BACKEND MATCHES "^eigen$")
  set(WITH_EIGEN_BACKEND 1)
elseif(BACKEND MATCHES "^cuda$")
  set(WITH_CUDA_BACKEND 1)
else()
  message(SEND_ERROR "BACKEND must be eigen or cuda")
endif()

if (WITH_CUDA_BACKEND)
  find_package(CUDA REQUIRED)
  set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_ROOT})
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  #list(APPEND CUDA_LIBRARIES /usr/lib64/libpthread.so)
  list(APPEND CUDA_LIBRARIES ${CUDA_curand_LIBRARY})
  MESSAGE("CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
  list(REMOVE_ITEM CUDA_LIBRARIES -lpthread)
  set(LIBS ${LIBS} ${CUDA_LIBRARIES})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_USE_GPU")
  find_cudnn()
  if(CUDNN_FOUND)
    include_directories(SYSTEM ${CUDNN_INCLUDE_DIRS})
    list(APPEND CUDA_LIBRARIES ${CUDNN_LIBRARIES})
    message("-- Successfully include CUDNN flags")
  else()
    message("-- CUDNN not found, some dependent functionalities will be disabled")
  endif()
endif()

# look for Eigen
if (DEFINED ENV{EIGEN3_INCLUDE_DIR} AND NOT DEFINED EIGEN3_INCLUDE_DIR) # use env variable if not set
  set(EIGEN3_INCLUDE_DIR $ENV{EIGEN3_INCLUDE_DIR})
endif()
get_filename_component(EIGEN3_INCLUDE_DIR "${EIGEN3_INCLUDE_DIR}" REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(EIGEN3_INCLUDE_DIR ${EIGEN3_INCLUDE_DIR} CACHE STRING "" FORCE)
message("-- Eigen dir is " ${EIGEN3_INCLUDE_DIR})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

FIND_PACKAGE(Threads REQUIRED)
set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.cmake ${CMAKE_CURRENT_BINARY_DIR}/config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(dynet)

if(ENABLE_CPP_EXAMPLES)
  add_subdirectory(tutorial)
  add_subdirectory(examples)
endif(ENABLE_CPP_EXAMPLES)

if(PYTHON)
  add_subdirectory(python)
endif(PYTHON)

if(ENABLE_SWIG)
  message("-- Enabling SWIG")
  add_subdirectory(contrib/swig)
endif(ENABLE_SWIG)

if(ENABLE_BOOST)
  add_subdirectory(tests)
  enable_testing()
endif()
