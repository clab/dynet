# ########## cnn library ##########
# Sources:
set(cnn_library_SRCS
    cnn.cc
    dict.cc
    dim.cc
    edges-common.cc
    gru.cc
    lstm.cc
    param-edges.cc
    rnn.cc
    rnn-state-machine.cc
    training.cc
)

# Headers:
set(cnn_library_HDRS
    cnn.h
    dict.h
    dim.h
    edges.h
    lstm.h
    model.h
    param-edges.h
    rnn.h
    rnn-state-machine.h
    saxe-init.h
    tensor.h
    timing.h
    training.h
)

if(WITH_EIGEN_BACKEND)
  list(APPEND cnn_library_SRCS
       backends/eigen/saxe-init.cc
       backends/eigen/init.cc
       backends/eigen/edges.cc
       backends/eigen/tensor.cc
       backends/eigen/model.cc)
endif(WITH_EIGEN_BACKEND)

file(GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} tests/*.cc)

#foreach(test_src ${TEST_SRCS})
  #Extract the filename without an extension (NAME_WE)
#  get_filename_component(testName ${test_src} NAME_WE)

  #Add compile target
#  add_executable(${testName} ${test_src})

  #link to Boost libraries AND your targets and dependencies
#  target_link_libraries(${testName} cnn ${LIBS})

#  set_target_properties(${testName} PROPERTIES 
#      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests.bin)

  #Finally add it to test execution - 
  #Notice the WORKING_DIRECTORY and COMMAND
#  add_test(NAME ${testName} 
#     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests.bin 
#     COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests.bin/${testName} )
#endforeach(test_src)

# actual target:
add_library(cnn ${cnn_library_SRCS} ${cnn_library_HDRS} ${LIBS})
# target_compile_features(cnn PRIVATE cxx_range_for)

