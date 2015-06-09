CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

#foreach(TARGET embed-cl encdec encdec-expr xor xor-xent rnnlm nlm)
foreach(TARGET encdec-expr) 
  ADD_EXECUTABLE(${TARGET} ${TARGET}.cc)
  target_link_libraries(${TARGET} cnn ${LIBS})
  if (WITH_CUDA_BACKEND)
    add_dependencies(${TARGET} cnncuda)
    target_link_libraries(${TARGET} cnncuda)
    CUDA_ADD_CUBLAS_TO_TARGET(${TARGET})
  endif (WITH_CUDA_BACKEND)
endforeach()

