
function(get_filename_components DEST ITEMS PART)
  foreach(ITEM ${ITEMS})
    get_filename_component(ITEMPART "${ITEM}" ${PART})
    list(APPEND ${DEST} ${ITEMPART})
  endforeach()
  list(REMOVE_DUPLICATES ${DEST})
  set(${DEST} ${${DEST}} PARENT_SCOPE)
endfunction()

if(!PYTHON)
  find_program(PYTHON "python")
endif()

if(PYTHON)
  # Things to copy the files
  file(GLOB DepFiles ${CMAKE_CURRENT_SOURCE_DIR}/dynet.py
                     ${CMAKE_CURRENT_SOURCE_DIR}/_dynet.pyx
                     ${CMAKE_CURRENT_SOURCE_DIR}/_dynet.pxd
                     ${CMAKE_CURRENT_SOURCE_DIR}/dynet_viz.py
                     ${CMAKE_CURRENT_SOURCE_DIR}/pybridge.h
  )

if (WITH_CUDA_BACKEND)
  message("--- CUDA: CUBLAS: ${CUDA_CUBLAS_LIBRARIES} RT: ${CUDA_LIBRARIES}")
  # Collect all of the library filenames and paths for setup.py
  get_filename_components(CUDA_CUBLAS_DIRS "${CUDA_CUBLAS_LIBRARIES}" PATH)
  get_filename_components(CUDA_CUBLAS_FILES "${CUDA_CUBLAS_LIBRARIES}" NAME)
  get_filename_components(CUDA_RT_DIRS "${CUDA_LIBRARIES}" PATH)
  get_filename_components(CUDA_RT_FILES "${CUDA_LIBRARIES}" NAME)
endif()

  add_custom_target(copy)
  foreach(DepFile ${DepFiles})
    add_custom_command(TARGET copy PRE_BUILD
                       COMMAND ${CMAKE_COMMAND} -E
                           copy ${DepFile} ${CMAKE_CURRENT_BINARY_DIR})
  endforeach()
  # TODO: We should add a dependency, but this isn&t working
  # add_dependencies(copy ${DepFiles})

  add_custom_command(TARGET copy PRE_BUILD
		  COMMAND ${CMAKE_COMMAND} -E
		  copy ${CMAKE_CURRENT_SOURCE_DIR}/_dynet.pyx ${CMAKE_CURRENT_BINARY_DIR}/_gdynet.pyx)


  # Things to perform setup
  set(SETUP_PY "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
  set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")

  configure_file(${SETUP_PY_IN} ${SETUP_PY})
  
  add_custom_command(
          OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/timestamp
          COMMAND ${PYTHON} 
          ARGS setup.py build_ext --inplace
          COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/timestamp
          DEPENDS ${DepFiles})

  add_custom_target(target ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/timestamp)
  add_dependencies(target copy)
  add_dependencies(copy dynet)
  if (WITH_CUDA_BACKEND)
  add_dependencies(copy gdynet)
  endif()

  install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY})")

endif()
