PROJECT(cnn:examples)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

ADD_EXECUTABLE(xor xor.cc)
target_link_libraries(xor cnn ${Boost_LIBRARIES})

ADD_EXECUTABLE(xor-xent xor-xent.cc)
target_link_libraries(xor-xent cnn ${Boost_LIBRARIES})

ADD_EXECUTABLE(rnnlm rnnlm.cc)
target_link_libraries(rnnlm cnn ${Boost_LIBRARIES})

ADD_EXECUTABLE(transnn transnn.cc)
target_link_libraries(transnn cnn ${Boost_LIBRARIES})

ADD_EXECUTABLE(nlm nlm.cc)
target_link_libraries(nlm cnn ${Boost_LIBRARIES})


