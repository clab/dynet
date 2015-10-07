CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

foreach(TARGET tok-embed segrnn-sup poisson-regression tag-bilstm embed-cl encdec xor xor-xent rnnlm-aevb rnnlm nlm textcat rnnlm2 mp easy-char-lstm-lm)
  ADD_EXECUTABLE(${TARGET} ${TARGET}.cc)
  target_link_libraries(${TARGET} cnn ${LIBS})
  if (WITH_CUDA_BACKEND)
    add_dependencies(${TARGET} cnncuda)
    target_link_libraries(${TARGET} cnncuda)
    CUDA_ADD_CUBLAS_TO_TARGET(${TARGET})
  endif (WITH_CUDA_BACKEND)
endforeach()

