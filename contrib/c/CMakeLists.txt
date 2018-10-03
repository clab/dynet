cmake_minimum_required(VERSION 2.8)
project(dynet_c)

if (WITH_CUDA_BACKEND)
  set(DYNET_C_USE_CUDA 1)
endif()

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_BINARY_DIR}
  "../../dynet")
add_subdirectory(dynet_c)

if(ENABLE_BOOST)
  add_subdirectory(test)
  enable_testing()
endif()
