CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

if (NOT MSVC)
   set(MP_EXAMPLES rnnlm-mp xor-mp xor-simple-mp)
endif()


foreach(FOLDER encdec mlc mnist tok-embed poisson-regression tag-bilstm embed-cl xor xor-autobatch xor-xent xor-batch xor-batch-lookup rnnlm rnnlm-aevb rnnlm-cfsm rnnlm-batch nlm read-write segrnn-sup attention imdb rnnlm-batch-nce ${MP_EXAMPLES})
  set(TARGET train_${FOLDER})
  ADD_EXECUTABLE(${TARGET} cpp/${FOLDER}/${TARGET}.cc)
  if (WITH_CUDA_BACKEND)
    target_link_libraries(${TARGET} gdynet ${LIBS})
    CUDA_ADD_CUBLAS_TO_TARGET(${TARGET})
  else()
    target_link_libraries(${TARGET} dynet ${LIBS})
  endif (WITH_CUDA_BACKEND)
  if(UNIX AND NOT APPLE)
    target_link_libraries(${TARGET} rt)
  endif()
endforeach()

