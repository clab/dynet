#
# This will define:
#  MINERVA_FOUND
#  MINERVA_LIBRARY
#  MINERVA_INCLUDE_DIR

find_path(MINERVA_INCLUDE_DIR
    NAMES minerva.h
    PATHS ${MINERVA_PREFIX_PATH}/minerva)

find_library(MINERVA_LIBRARY
    NAMES minerva
    PATHS ${MINERVA_PREFIX_PATH}/release/lib)

if(NOT MINERVA_LIBRARY)
  find_library(MINERVA_LIBRARY
      NAMES minerva
      PATHS ${MINERVA_PREFIX_PATH}/debug/lib)
endif(NOT MINERVA_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Minerva DEFAULT_MSG MINERVA_LIBRARY MINERVA_INCLUDE_DIR)

