# ########## dynet library ##########
# Sources:
set(dynet_library_SRCS
    aligned-mem-pool.cc
    cfsm-builder.cc
    deep-lstm.cc
    devices.cc
    dict.cc
    dim.cc
    dynet.cc
    exec.cc
    expr.cc
    fast-lstm.cc
    globals.cc
    grad-check.cc
    graph.cc
    gru.cc
    hsm-builder.cc
    init.cc
    io.cc
    lstm.cc
    mem.cc
    model.cc
    nodes-activations.cc
    nodes-affinetransform.cc
    nodes-argmax.cc
    nodes-arith-const.cc
    nodes-arith-cwise.cc
    nodes-arith-sum.cc
    nodes-arith-unary.cc
    nodes-concat.cc
    nodes-const.cc
    nodes-contract.cc
    nodes-conv.cc
    nodes-conv2d.cc
    nodes-dropout.cc
    nodes-flow.cc
    nodes-hinge.cc
    nodes-linalg.cc
    nodes-logsumexp.cc
    nodes-losses.cc
    nodes-lstm.cc
    nodes-matrixmultiply.cc
    nodes-maxpooling2d.cc
    nodes-minmax.cc
    nodes-moments.cc
    nodes-cumulative.cc
    nodes-normalization.cc
    nodes-norms.cc
    nodes-pickneglogsoftmax.cc
    nodes-random.cc
    nodes-select.cc
    nodes-similarities.cc
    nodes-softmaxes.cc
    nodes-to-device.cc
    nodes-trig.cc
    param-init.cc
    param-nodes.cc
    pretrain.cc
    rnn-state-machine.cc
    rnn.cc
    saxe-init.cc
    shadow-params.cc
    tensor.cc
    training.cc
    treelstm.cc
    weight-decay.cc
)
if(ENABLE_BOOST)
  list(APPEND dynet_library_SRCS mp.cc)
endif()

# Headers:
set(dynet_library_HDRS
aligned-mem-pool.h
c2w.h
cfsm-builder.h
cuda.h
cudnn-ops.h
deep-lstm.h
devices.h
device-structs.h
dict.h
dim.h
dynet.h
dynet-helper.h
except.h
exec.h
expr.h
fast-lstm.h
functors.h
globals.h
gpu-kernels.h
gpu-ops.h
grad-check.h
graph.h
gru.h
hsm-builder.h
index-tensor.h
init.h
io.h
lstm.h
matrix-multiply.h
mem.h
model.h
mp.h
nodes-activations.h
nodes-affinetransform.h
nodes-argmax.h
nodes-arith-const.h
nodes-arith-cwise.h
nodes-arith-sum.h
nodes-arith-unary.h
nodes-concat.h
nodes-const.h
nodes-contract.h
nodes-conv2d.h
nodes-conv.h
nodes-cumulative.h
nodes-def-macros.h
nodes-dropout.h
nodes-flow.h
nodes.h
nodes-hinge.h
nodes-impl-macros.h
nodes-linalg.h
nodes-logsumexp.h
nodes-losses.h
nodes-lstm.h
nodes-matrixmultiply.h
nodes-maxpooling2d.h
nodes-minmax.h
nodes-moments.h
nodes-normalization.h
nodes-norms.h
nodes-pickneglogsoftmax.h
nodes-random.h
nodes-select.h
nodes-similarities.h
nodes-softmaxes.h
nodes-to-device.h
nodes-trig.h
param-init.h
param-nodes.h
pretrain.h
rnn.h
rnn-state-machine.h
saxe-init.h
shadow-params.h
sig.h
simd-functors.h
str-util.h
tensor-eigen.h
tensor.h
timing.h
training.h
treelstm.h
virtual-cudnn.h
weight-decay.h
)
if(ENABLE_BOOST)
  list(APPEND dynet_library_HDRS mp.h)
endif()
  
set(dynet_gpu_mergeable_SRCS
    nodes-activations
    nodes-affinetransform
    nodes-argmax
    nodes-arith-const
    nodes-arith-cwise
    nodes-arith-sum
    nodes-arith-unary
    nodes-concat
    nodes-const
    nodes-contract
    nodes-conv2d
    nodes-conv
    nodes-dropout
    nodes-flow
    nodes-hinge
    nodes-linalg
    nodes-logsumexp
    nodes-losses
    nodes-lstm
    nodes-matrixmultiply
    nodes-maxpooling2d
    nodes-minmax
    nodes-moments
    nodes-cumulative
    nodes-normalization
    nodes-norms
    nodes-pickneglogsoftmax
    nodes-random
    nodes-select
    nodes-similarities
    nodes-softmaxes
    nodes-to-device
    nodes-trig
    param-nodes
    tensor
    training
    model)

set(dynet_gpu_SRCS
    cuda.cc
    cudnn-ops.cu
    gpu-ops.cu)

file(GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} tests/*.cc)
if (NOT MSVC)
  set(BUILD_SHARED_LIBS ON)
endif()

#foreach(test_src ${TEST_SRCS})
  #Extract the filename without an extension (NAME_WE)
#  get_filename_component(testName ${test_src} NAME_WE)

  #Add compile target
#  add_executable(${testName} ${test_src})

  #link to Boost libraries AND your targets and dependencies
#  target_link_libraries(${testName} dynet ${LIBS})

#  set_target_properties(${testName} PROPERTIES
#      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests.bin)

  #Finally add it to test execution -
  #Notice the WORKING_DIRECTORY and COMMAND
#  add_test(NAME ${testName}
#     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests.bin
#     COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests.bin/${testName} )
#endforeach(test_src)

if(WITH_CUDA_BACKEND)
  if(${GPU_NUMFILES} EQUAL 0)
    foreach(FILENAME ${dynet_gpu_mergeable_SRCS})
      set(MERGEFILE ${PROJECT_BINARY_DIR}/dynet/gpu-${FILENAME}.cu)
      file(WRITE ${MERGEFILE} "#include \"${PROJECT_SOURCE_DIR}/dynet/${FILENAME}.cc\"\n")
      list(APPEND dynet_gpu_SRCS ${MERGEFILE})
    endforeach()
  else()
    # merge into k merge files
    set(COUNTER 0)
    foreach(FILENAME ${dynet_gpu_mergeable_SRCS})
      math(EXPR MERGENUM "${COUNTER} % ${GPU_NUMFILES}")
      set(MERGEFILE ${PROJECT_BINARY_DIR}/dynet/gpu-merge${MERGENUM}.cu)
      if(${COUNTER} EQUAL ${MERGENUM})   # This is the first line of this merge file
        file(REMOVE ${MERGEFILE})        # Start with an empty file
        list(APPEND dynet_gpu_SRCS ${MERGEFILE})
      endif()
      file(APPEND ${MERGEFILE} "#include \"${PROJECT_SOURCE_DIR}/dynet/${FILENAME}.cc\"\n")
      math(EXPR COUNTER ${COUNTER}+1)
    endforeach()
  endif()

  # cuda flags
  set(CUDA_SEPARABLE_COMPILATION ON)

  if(CUDA_ARCH)
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_${CUDA_ARCH};-std=c++11;-DVERBOSE;-DEIGEN_USE_GPU;-DHAVE_CUDA;-D_FORCE_INLINES")
  else()
    list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_30,code=sm_30;-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_52,code=compute_52;-std=c++11;-DVERBOSE;-DEIGEN_USE_GPU;-DHAVE_CUDA;-D_FORCE_INLINES")
  endif()
  if(CUDNN_FOUND)
    list(APPEND CUDA_NVCC_FLAGS "-DHAVE_CUDNN")
  endif()
  if(CMAKE_COMPILER_IS_GNUCXX)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 4.9)
      # gcc 4.9 or later versions raise SEGV due to the optimization problem.
      # Use -O1 instead for now.
      list(APPEND CUDA_NVCC_FLAGS "-O1")
    else()
      list(APPEND CUDA_NVCC_FLAGS "-O2")
    endif()
  else()
    list(APPEND CUDA_NVCC_FLAGS "-O2")
  endif()
  if(MSVC)
    # If MSVC, we need the boost flag because nvcc doesn't properly parse part of the boost template definitions
    list(APPEND CUDA_NVCC_FLAGS "-DBOOST_NO_CXX11_ALLOCATOR")
    list(APPEND CUDA_NVCC_FLAGS_DEBUG "--compiler-options \"/MDd\"")
    list(APPEND CUDA_NVCC_FLAGS_RELEASE "--compiler-options \"/MD\"")
    SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
    cuda_add_library(dynet ${dynet_library_SRCS} ${dynet_library_HDRS} ${dynet_gpu_SRCS})
  else()
    SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
    cuda_add_library(dynet ${dynet_library_SRCS} ${dynet_library_HDRS} ${dynet_gpu_SRCS} OPTIONS --compiler-options "-fPIC")
  endif()
  set_target_properties(dynet PROPERTIES
                        COMPILE_DEFINITIONS HAVE_CUDA)
  if(CUDNN_FOUND)
    target_compile_definitions(dynet PRIVATE HAVE_CUDNN)
  endif()
  cuda_add_cublas_to_target(dynet)
  target_link_libraries(dynet ${LIBS})
else()
  # Build cpu library
  add_library(dynet ${dynet_library_SRCS} ${dynet_library_HDRS})
  target_link_libraries(dynet ${LIBS})
endif(WITH_CUDA_BACKEND)

install(FILES ${dynet_library_HDRS} DESTINATION include/dynet)
install(TARGETS dynet DESTINATION lib)
if(WITH_CUDA_BACKEND)
  install(TARGETS dynet DESTINATION lib)
endif(WITH_CUDA_BACKEND)

# target_compile_features(dynet PRIVATE cxx_range_for)
