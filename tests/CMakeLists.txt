find_package (Boost COMPONENTS system filesystem unit_test_framework REQUIRED)
include_directories (${TEST_SOURCE_DIR}/src
                     ${Boost_INCLUDE_DIRS}
                     )

add_definitions (-DBOOST_TEST_DYN_LINK)

# Sources:
set(test_dynet_SRCS
    test-nodes.cc
    test-trainers.cc
    test-io.cc
    test-params.cc
)

add_executable (test-dynet test-dynet.cc ${test_dynet_SRCS})

if (WITH_CUDA_BACKEND)
  target_link_libraries (test-dynet gdynet ${LIBS}
                         ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
                         )
  CUDA_ADD_CUBLAS_TO_TARGET(test-dynet)
else()
  target_link_libraries (test-dynet dynet ${LIBS}
                         ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
                         )
endif (WITH_CUDA_BACKEND)

add_test(test-dynet test-dynet)

