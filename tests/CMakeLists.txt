find_package (Boost COMPONENTS system filesystem unit_test_framework REQUIRED)
include_directories (${TEST_SOURCE_DIR}/src
                     ${Boost_INCLUDE_DIRS}
                     )

add_definitions (-DBOOST_TEST_DYN_LINK)

# Sources:
set(test_cnn_SRCS
    test-nodes.cc
    test-trainers.cc
)

add_executable (test-cnn test-cnn.cc ${test_cnn_SRCS})

if (WITH_CUDA_BACKEND)
  target_link_libraries (test-cnn gcnn ${LIBS}
                         ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
                         )
  add_dependencies(test-cnn cnncuda)
  target_link_libraries(test-cnn cnncuda)
  CUDA_ADD_CUBLAS_TO_TARGET(test-cnn)
else()
  target_link_libraries (test-cnn cnn ${LIBS}
                         ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
                         )
endif (WITH_CUDA_BACKEND)

add_test(test-cnn test-cnn)

