#ifndef DYNET_C_STATUS_H_
#define DYNET_C_STATUS_H_

#include <dynet_c/define.h>

DYNET_C_API DYNET_C_STATUS dynetResetStatus();

DYNET_C_API DYNET_C_STATUS dynetGetMessage(char *retval, size_t *size);

#endif  // DYNET_C_STATUS_H_
