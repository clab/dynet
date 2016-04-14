project(cnn)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# CNN uses Eigen which exploits modern CPU architectures. To get the
# best possible performance, the following are recommended:
#   1. use very recent versions of gcc or Clang to build
#   2. use very recent versions of Eigen (ideally the dev version)
#   3. try compiler options like -march=native or other architecture
#      flags (the compiler does not always make the best configuration
#      decisions without help)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -funroll-loops -Wall -std=c++11 -Ofast -g -DEIGEN_FAST_MATH -march=native")

enable_testing()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${PROJECT_SOURCE_DIR}/external/easyloggingpp/src)

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
if(DEFINED ENV{BOOST_ROOT})
  set(Boost_NO_SYSTEM_PATHS ON)
endif()
set(Boost_REALPATH ON)
find_package(Boost COMPONENTS program_options regex serialization REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
set(LIBS ${LIBS} ${Boost_LIBRARIES})
# trouble shooting:
# if boost library cannot be found, in addition to install boost library
# check if environment variables are set
# 
# to set boost root and its library root in environment variable, use
# for example
# echo "export BOOST_LIBRARYDIR=/usr/local/lib" >> ~/.bashrc
# echo "export BOOST_ROOT=/cygdrive/d/tools/boost_1_58_0/boost_1_58_0" >> ~/.bashrc
# then run source ~/.bashrc to have those environment variable effective immediately

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
  add_definitions(-DHAVE_CUDA)
  #list(APPEND CUDA_LIBRARIES /usr/lib64/libpthread.so)
  MESSAGE("CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
  list(REMOVE_ITEM CUDA_LIBRARIES -lpthread)
  set(LIBS ${LIBS} ${CUDA_LIBRARIES})
  #find_cudnn()
  #include_directories(SYSTEM ${CUDNN_INCLUDE_DIRS})
endif()

# look for Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

FIND_PACKAGE(Threads REQUIRED)
set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.cmake ${CMAKE_CURRENT_BINARY_DIR}/config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(cnn)
add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(rnnlm)
enable_testing()
