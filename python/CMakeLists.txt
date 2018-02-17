
function(get_filename_components DEST ITEMS PART)
  foreach(ITEM ${ITEMS})
    get_filename_component(ITEMPART "${ITEM}" ${PART})
    list(APPEND ${DEST} ${ITEMPART})
  endforeach()
  list(REMOVE_DUPLICATES ${DEST})
  set(${DEST} ${${DEST}} PARENT_SCOPE)
endfunction()

function(check_cython_version VERSION)
  # We use the Cython library directly within Python, make sure it is installed
  # and meets the minimum version required.
  execute_process(COMMAND ${PYTHON} -c "import Cython; print(Cython.__version__);" OUTPUT_VARIABLE CYTHON_VERSION RESULT_VARIABLE RETCODE)
  if(NOT "${RETCODE}" STREQUAL "0")
    message(FATAL_ERROR "Failed executing ${PYTHON}")
  endif()
  if(${CYTHON_VERSION} VERSION_LESS ${VERSION})
    message(FATAL_ERROR "DyNet needs at least Cython version ${VERSION}, found verson ${CYTHON_VERSION}")
  else()
    message(STATUS "Found Cython version ${CYTHON_VERSION}")
  endif()
endfunction()

if(!PYTHON)
  find_program(PYTHON "python")
endif()

if(PYTHON)
  # Verify Cython is installed with the given minimum version
  check_cython_version(0.23)
  
  # Things to copy the files
  file(GLOB DepFiles ${CMAKE_CURRENT_SOURCE_DIR}/dynet.py.in
                     ${CMAKE_CURRENT_SOURCE_DIR}/dynet_config.py
                     ${CMAKE_CURRENT_SOURCE_DIR}/_dynet.pyx
                     ${CMAKE_CURRENT_SOURCE_DIR}/_dynet.pxd
                     ${CMAKE_CURRENT_SOURCE_DIR}/dynet_viz.py
  )

  if (WITH_CUDA_BACKEND)
    message("--- CUDA: CUBLAS: ${CUDA_CUBLAS_LIBRARIES} RT: ${CUDA_LIBRARIES}")
    # Collect all of the library filenames and paths for setup.py
    get_filename_components(CUDA_CUBLAS_DIRS "${CUDA_CUBLAS_LIBRARIES}" PATH)
    get_filename_components(CUDA_CUBLAS_FILES "${CUDA_CUBLAS_LIBRARIES}" NAME)
    get_filename_components(CUDA_RT_DIRS "${CUDA_LIBRARIES}" PATH)
    get_filename_components(CUDA_RT_FILES "${CUDA_LIBRARIES}" NAME)
  endif()

  add_custom_target(pydynet_precopy)
  foreach(DepFile ${DepFiles})
    add_custom_command(TARGET pydynet_precopy PRE_BUILD
                       COMMAND ${CMAKE_COMMAND} -E
                           copy ${DepFile} ${CMAKE_CURRENT_BINARY_DIR})
  endforeach()
  # TODO: We should add a dependency, but this isn't working
  # add_dependencies(pydynet_precopy ${DepFiles})

  # Export environment variables for setup.py
  foreach(Var CMAKE_INSTALL_PREFIX PROJECT_SOURCE_DIR PROJECT_BINARY_DIR LIBS EIGEN3_INCLUDE_DIR MKL_LINK_DIRS WITH_CUDA_BACKEND CUDA_RT_FILES CUDA_RT_DIRS CUDA_CUBLAS_FILES CUDA_CUBLAS_DIRS MSVC)
    string(REPLACE ";" "\\\\;" Val "${${Var}}")
    message(${Var}="${Val}")
    list(APPEND EnvVars ${Var}="${Val}")
  endforeach()

  add_custom_command(
          OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/timestamp
          COMMAND ${PYTHON}
          ARGS ${CMAKE_SOURCE_DIR}/setup.py ${EnvVars} build_ext --inplace
          COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/timestamp
          DEPENDS ${DepFiles})

  add_custom_target(pydynet ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/timestamp)
  add_dependencies(pydynet pydynet_precopy)
  add_dependencies(pydynet_precopy dynet)

  execute_process(
      COMMAND git describe --tags --always
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE GIT_DESCRIBE
      OUTPUT_STRIP_TRAILING_WHITESPACE
      )

endif()

if (WITH_CUDA_BACKEND)
  set(HAVE_CUDA True)
else()
  set(HAVE_CUDA False)
endif()
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/dynet.py.in"
               "${CMAKE_CURRENT_BINARY_DIR}/dynet.py")
