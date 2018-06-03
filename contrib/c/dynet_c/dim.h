#ifndef DYNET_C_DIM_H_
#define DYNET_C_DIM_H_

#include <dynet_c/define.h>

/**
 * Opaque type of Dim.
 */
typedef struct dynetDim dynetDim_t;

/**
 * Creates a new Dim object.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateDim(dynetDim_t **newobj);

/**
 * Deletes the Dim object.
 * @param shape Pointer of a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetDeleteDim(dynetDim_t *dim);

#endif  // DYNET_C_DIM_H_
