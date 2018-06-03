#ifndef DYNET_C_INIT_H_
#define DYNET_C_INIT_H_

#include <dynet_c/define.h>

/**
 * Opaque type of DynetParams.
 */
typedef struct dynetDynetParams_t;

/**
 * Creates a new DynetParams object.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateDynetParams(dynetDynetParams_t **newobj);

/**
 * Deletes the DynetParams object.
 * @param shape Pointer of a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetDeleteDynetParams(dynetDynetParams_t *dim);

#ifdef DYNET_C_USE_CUDA
#endif  // DYNET_C_USE_CUDA

#endif  // DYNET_C_INIT_H_
