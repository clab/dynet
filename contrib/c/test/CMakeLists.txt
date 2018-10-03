include(CTest)

find_package(Boost COMPONENTS system filesystem unit_test_framework REQUIRED)
include_directories(${TEST_SOURCE_DIR}/src ${Boost_INCLUDE_DIRS})
add_definitions(-DBOOST_TEST_DYN_LINK)

function(dynet_c_test name)
  add_executable(test-c-${name} ${name}_test.cc)
  target_link_libraries(test-c-${name} dynet_c
      ${LIBS} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
  add_test(
      NAME test-c-${name}
      COMMAND test-c-${name}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endfunction()

dynet_c_test(dim)
dynet_c_test(exec)
dynet_c_test(params)
