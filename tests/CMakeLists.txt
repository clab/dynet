find_package (Boost COMPONENTS system filesystem unit_test_framework REQUIRED)
include_directories (${TEST_SOURCE_DIR}/src
                     ${Boost_INCLUDE_DIRS}
                     )

add_definitions (-DBOOST_TEST_DYN_LINK)

add_executable (test-cnn test-cnn.cc)

target_link_libraries (test-cnn cnn ${LIBS}
                       ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
                       )
add_test(test-cnn test-cnn)

