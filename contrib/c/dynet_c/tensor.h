#ifndef DYNET_C_TENSOR_H_
#define DYNET_C_TENSOR_H_

#include <dynet_c/define.h>
#include <dynet_c/dim.h>

/**
 * Opaque type of Tensor.
 */
typedef struct dynetTensor dynetTensor_t;

/**
 * Deletes the Tensor object.
 * @param shape Pointer of a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetDeleteTensor(dynetTensor_t *tensor);

/**
 * Returns dim of the tensor.
 * @param tensor Pointer of a handler.
 * @param newobj Pointer to receive a Dim object.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetTensorDim(
    const dynetTensor_t *tensor, dynetDim_t **newobj);

/**
 * Retrieves one internal value in the tensor.
 * @param tensor Pointer of a handler.
 * @param retval Pointer to receive an internal float value.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetEvaluateTensorAsScalar(
    const dynetTensor_t *tensor, float *retval);

/**
 * Retrieves internal values in the tensor as a vector.
 * @param tensor Pointer of a handler.
 * @param retval Pointer to receive a list of the internal values.
 * @param size Pointer to receive the length of the array.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetEvaluateTensorAsArray(
    const dynetTensor_t *tensor, float *retval, size_t *size);

/**
 * Returns a string representation of the tensor.
 * @param tensor Pointer of a handler.
 * @param retval Pointer to receive the encoded string.
 * @param size Pointer to receive a length of the char sequence.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetRepresentTensorAsString(
    const dynetTensor_t *tensor, char *retval, size_t *size);

#endif  // DYNET_C_TENSOR_H_
