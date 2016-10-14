CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# segrnn-sup
foreach(TARGET mlc tok-embed poisson-regression tag-bilstm embed-cl encdec xor xor-xent xor-batch xor-batch-lookup rnnlm rnnlm-aevb rnnlm-cfsm rnnlm-batch nlm textcat rnnlm-mp read-write)
  ADD_EXECUTABLE(${TARGET} ${TARGET}.cc)
  if (WITH_CUDA_BACKEND)
    target_link_libraries(${TARGET} ${MKL_LIBS} gdynet ${LIBS})
    add_dependencies(${TARGET} dynetcuda)
    target_link_libraries(${TARGET} dynetcuda)
    CUDA_ADD_CUBLAS_TO_TARGET(${TARGET})
  else()
    target_link_libraries(${TARGET} ${MKL_LIBS} dynet ${LIBS})
  endif (WITH_CUDA_BACKEND)
  if(UNIX AND NOT APPLE)
    target_link_libraries(${TARGET} rt)
  endif()
endforeach()

