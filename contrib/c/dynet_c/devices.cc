#include <dynet_c/config.h>

#include <dynet/devices.h>
#include <dynet_c/internal.h>
#include <dynet_c/devices.h>

using dynet_c::internal::to_c_ptr;

DYNET_C_STATUS dynetGetGlobalDevice(
    const char *name, dynetDevice_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(newobj);
  if (name) {
    *newobj = to_c_ptr(dynet::get_device_manager()->get_global_device(name));
  } else {
    *newobj = to_c_ptr(dynet::get_device_manager()->get_global_device(""));
  }
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetNumDevices(uint32_t *retval) try {
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = dynet::get_device_manager()->num_devices();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS
