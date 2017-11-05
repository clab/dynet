CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

foreach(TARGET 0_multiply 1_linear_regression)
  add_executable(${TARGET} ${TARGET}.cc)
  target_link_libraries(${TARGET} dynet ${LIBS})
  if(UNIX AND NOT APPLE)
    target_link_libraries(${TARGET} rt)
  endif()
  if (WITH_CUDA_BACKEND)
    CUDA_ADD_CUBLAS_TO_TARGET(${TARGET})
  endif (WITH_CUDA_BACKEND)
endforeach()

