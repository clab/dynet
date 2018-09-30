#ifndef DYNET_C_DEVICES_H_
#define DYNET_C_DEVICES_H_

#include <dynet_c/define.h>

/**
 * Opaque type of Device.
 */
typedef struct dynetDevice dynetDevice_t;

/**
 * Retrieves a global device.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetGlobalDevice(
    const char *name, dynetDevice_t **newobj);

/**
 * Returns the number of global devices.
 * @param retval Pointer to receive the number,
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetNumDevices(uint32_t *retval);

#endif  // DYNET_C_DEVICES_H_
