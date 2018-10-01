include(CTest)
enable_testing()

find_package(Boost COMPONENTS system filesystem unit_test_framework REQUIRED)
include_directories(${TEST_SOURCE_DIR}/src ${Boost_INCLUDE_DIRS})
add_definitions(-DBOOST_TEST_DYN_LINK)

if (DEFINED ENV{DYNET_TEST_DEVICES})  # use env variable as preprocessor macro
  add_definitions(-DDYNET_TEST_DEVICES=$ENV{DYNET_TEST_DEVICES})
endif()

foreach(TESTNAME dim dynet exec io mem nodes params tensor trainers trainers-io rnn softmax)
  add_executable(test-${TESTNAME} test-${TESTNAME}.cc)
  if (NOT MSVC)
    target_link_libraries(test-${TESTNAME} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
  endif()
  target_link_libraries(test-${TESTNAME} dynet ${LIBS} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
  if (WITH_CUDA_BACKEND)
    CUDA_ADD_CUBLAS_TO_TARGET(test-${TESTNAME})
  endif(WITH_CUDA_BACKEND)
  add_test(NAME test-${TESTNAME} COMMAND test-${TESTNAME})
endforeach()
