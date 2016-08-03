find_program(PYTHON "python")

if(PYTHON)
  set(SETUP_PY "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
  set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
  set(DEPS "${CMAKE_CURRENT_SOURCE_DIR}/pycnn.py")

  configure_file(${SETUP_PY_IN} ${SETUP_PY})

  add_custom_command(
          OUTPUT ${OUTPUT}/timestamp
          COMMAND ${PYTHON} 
          ARGS setup.py build
          COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}/timestamp
          DEPENDS ${DEPS})

  add_custom_target(target ALL DEPENDS ${OUTPUT}/timestamp)

  install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY})")

endif()
