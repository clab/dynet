project(cnn)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -O3")

enable_testing()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# look for Boost
find_package(Boost COMPONENTS program_options serialization unit_test_framework REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

# look for Minerva
find_package(Minerva)
if(MINERVA_FOUND)
  include_directories(${MINERVA_INCLUDE_DIR})
  set(HAVE_MINERVA_H 1)
endif(MINERVA_FOUND)

# look for Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.cmake ${CMAKE_CURRENT_BINARY_DIR}/config.h)

add_subdirectory(cnn)
add_subdirectory(examples)

