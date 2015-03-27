# ########## Project setup ##########
PROJECT(cnn CXX)

# ########## cnn library ##########
# Sources:
set(cnn_library_SRCS
    cnn.cc
    edges.cc
    lstm.cc
    lstm-fast.cc
    model.cc
    param-edges.cc
    rnn.cc
    saxe_init.cc
    training.cc
)

if(MINERVA_FOUND)
  # for now
  list(APPEND cnn_library_SRCS backends/eigen/eigen-backend.cc)
else(MINERVA_FOUND)
  list(APPEND cnn_library_SRCS backends/eigen/eigen-backend.cc)
endif(MINERVA_FOUND)

# Headers:
set(cnn_library_HDRS
    cnn.h
    edges.h
    lstm.h
    lstm-fast.h
    model.h
    param-edges.h
    rnn.h
    saxe_init.h
    tensor.h
    timing.h
    training.h
)

file(GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} tests/*.cc)

foreach(test_src ${TEST_SRCS})
  #Extract the filename without an extension (NAME_WE)
  get_filename_component(testName ${test_src} NAME_WE)

  #Add compile target
  add_executable(${testName} ${test_src})

  #link to Boost libraries AND your targets and dependencies
  target_link_libraries(${testName} ${Boost_LIBRARIES} cnn)

  set_target_properties(${testName} PROPERTIES 
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests.bin)

  #Finally add it to test execution - 
  #Notice the WORKING_DIRECTORY and COMMAND
  add_test(NAME ${testName} 
     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests.bin 
     COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests.bin/${testName} )
endforeach(test_src)

# actual target:
add_library(cnn ${cnn_library_SRCS} ${cnn_library_HDRS})
# target_compile_features(cnn PRIVATE cxx_range_for)

