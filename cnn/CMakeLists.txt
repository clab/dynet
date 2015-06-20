# ########## cnn library ##########
# Sources:
set(cnn_library_SRCS
    cnn.cc
    conv.cc
    dict.cc
    dim.cc
    exec.cc
    expr.cc
    graph.cc
#    gru.cc
    init.cc
    lstm.cc
    model.cc
    nodes.cc
    nodes-common.cc
    param-nodes.cc
    rnn.cc
    rnn-state-machine.cc
    saxe-init.cc
    shadow-params.cc
    tensor.cc
    training.cc
)

# Headers:
set(cnn_library_HDRS
    aligned-mem-pool.h
    c2w.h
    cnn.h
    conv.h
    cuda.h
    dict.h
    dim.h
    exec.h
    expr.h
    functors.h
    gpu-kernels.h
    gpu-ops.h
    graph.h
    gru.h
    init.h
    lstm.h
    model.h
    nodes.h
    param-nodes.h
    random.h
    rnn-state-machine.h
    rnn.h
    saxe-init.h
    shadow-params.h
    tensor.h
    timing.h
    training.h
)

if(WITH_CUDA_BACKEND)
  list(APPEND cnn_library_SRCS
       cuda.cc)
endif(WITH_CUDA_BACKEND)

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
if(WITH_CUDA_BACKEND)
  set(CUDA_SEPARABLE_COMPILATION ON)
  list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_20,code=sm_20;-gencode;arch=compute_30,code=sm_30;-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_52,code=compute_52;-std=c++11;-O2;-DVERBOSE")
  SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
  cuda_add_library(cnncuda STATIC gpu-ops.cu)
endif(WITH_CUDA_BACKEND)

install(FILES ${cnn_library_HDRS} DESTINATION include/cnn)
install(TARGETS cnn DESTINATION lib)

# target_compile_features(cnn PRIVATE cxx_range_for)

