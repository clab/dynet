CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

ADD_EXECUTABLE(embed_cl embed_cl.cc)
target_link_libraries(embed_cl cnn ${LIBS})

ADD_EXECUTABLE(encdec encdec.cc)
target_link_libraries(encdec cnn ${LIBS})

ADD_EXECUTABLE(xor xor.cc)
target_link_libraries(xor cnn ${LIBS})

ADD_EXECUTABLE(xor-xent xor-xent.cc)
target_link_libraries(xor-xent cnn ${LIBS})

ADD_EXECUTABLE(rnnlm rnnlm.cc)
target_link_libraries(rnnlm cnn ${LIBS})

ADD_EXECUTABLE(nlm nlm.cc)
target_link_libraries(nlm cnn ${LIBS})


