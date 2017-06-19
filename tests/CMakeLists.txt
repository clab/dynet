include(CTest)
enable_testing()

find_package(Boost COMPONENTS system filesystem unit_test_framework REQUIRED)
include_directories(${TEST_SOURCE_DIR}/src ${Boost_INCLUDE_DIRS})
add_definitions(-DBOOST_TEST_DYN_LINK)

foreach(TESTNAME dynet exec io mem nodes params tensor trainers rnn)

  add_executable(test-${TESTNAME} test-${TESTNAME}.cc)
  
  if (NOT MSVC)
    target_link_libraries(test-${TESTNAME} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
  endif()
  
  if (WITH_CUDA_BACKEND)
    target_link_libraries(test-${TESTNAME} gdynet ${LIBS} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
    CUDA_ADD_CUBLAS_TO_TARGET(test-${TESTNAME})
  else()
    target_link_libraries(test-${TESTNAME} dynet ${LIBS} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
  endif(WITH_CUDA_BACKEND)
  
  add_test(NAME test-${TESTNAME} COMMAND test-${TESTNAME})

endforeach()
