#include <dynet_c/config.h>

#include <dynet_c/internal.h>
#include <dynet_c/status.h>

#include <string>

using dynet_c::internal::ErrorHandler;

DYNET_C_STATUS dynetResetStatus() try {
  ErrorHandler::get_instance().reset();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetMessage(char *retval, size_t *size) try {
  DYNET_C_CHECK_NOT_NULL(size);
  dynet_c::internal::copy_string_to_array(
      ErrorHandler::get_instance().get_message(), retval, size);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS
