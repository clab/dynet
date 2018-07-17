configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/config.h
)
install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/config.h
  DESTINATION include/dynet_c
)

set(dynet_c_HDRS
  api.h
  define.h
  dim.h
  graph.h
  init.h
  internal.h
  tensor.h
)
set(
  dynet_c_SRCS
  dim.cc
  init.cc
  internal.cc
  tensor.cc
)

set(BUILD_SHARED_LIBS ON)
add_library(dynet_c ${dynet_c_HDRS} ${dynet_c_SRCS})
target_link_libraries(dynet_c dynet)
install(FILES ${dynet_c_HDRS} DESTINATION include/dynet_c)
install(TARGETS dynet_c DESTINATION lib)
