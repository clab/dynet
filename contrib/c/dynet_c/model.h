#ifndef DYNET_C_MODEL_H_
#define DYNET_C_MODEL_H_

#include <dynet_c/define.h>
#include <dynet_c/dim.h>
#include <dynet_c/model.h>
#include <dynet_c/tensor.h>

/**
 * Opaque type of Parameter.
 */
typedef struct dynetParameter dynetParameter_t;

/**
 * Opaque type of LookupParameter.
 */
typedef struct dynetLookupParameter dynetLookupParameter_t;

/**
 * Opaque type of ParameterCollection.
 */
typedef struct dynetParameterCollection dynetParameterCollection_t;

/**
 * Creates a new Parameter object.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateParameter(dynetParameter_t **newobj);

/**
 * Deletes the Parameter object.
 * @param param Pointer of a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetDeleteParameter(dynetParameter_t *param);

/**
 * Fills the Parameter with zero values.
 * @param param Pointer of a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetFillParameterWithZeros(dynetParameter_t *param);

/**
 * Returns dim of the parameter.
 * @param param Pointer of a handler.
 * @param newobj Pointer to receive a Dim object.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetParameterDim(
    const dynetParameter_t *param, dynetDim_t **newobj);

/**
 * Retrieves internal values in the parameter as a tensor.
 * @param param Pointer of a handler.
 * @param tensor Pointer to receive a tensor of the internal values.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetParameterValues(
    dynetParameter_t *param, dynetTensor_t *tensor);

/**
 * Sets update status of the parameter.
 * @param param Pointer of a handler.
 * @param b Update status value.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetSetParameterUpdated(
    dynetParameter_t *param, DYNET_C_BOOL b);

/**
 * Sets update status of the parameter.
 * @param param Pointer of a handler.
 * @param retval Pointer to receive an update status value.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetIsParameterUpdated(
    const dynetParameter_t *param, DYNET_C_BOOL *retval);

/**
 * Creates a new LookupParameter object.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateLookupParameter(
    dynetLookupParameter_t **newobj);

/**
 * Deletes the LookupParameter object.
 * @param param Pointer of a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetDeleteLookupParameter(
    dynetLookupParameter_t *param);

/**
 * Initializes one paticular column of the values in the LookupParameter.
 * @param param Pointer of a handler.
 * @param index Index of the column to be initialized/
 * @param value List of initial values.
 * @param n Number of values.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetInitializeLookupParameter(
    dynetLookupParameter_t *param, uint32_t index, const float *value,
    size_t n);

/**
 * Fills the LookupParameter with zero values.
 * @param param Pointer of a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetFillLookupParameterWithZeros(
    dynetLookupParameter_t *param);

/**
 * Returns dim of the parameter.
 * @param param Pointer of a handler.
 * @param newobj Pointer to receive a Dim object.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetLookupParameterDim(
    const dynetLookupParameter_t *param, dynetDim_t **newobj);

/**
 * Retrieves internal values in the parameter as a tensor.
 * @param param Pointer of a handler.
 * @param tensor Pointer to receive a tensor of the internal values.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetLookupParameterValues(
    dynetLookupParameter_t *param, dynetTensor_t *tensor);

/**
 * Sets update status of the parameter.
 * @param param Pointer of a handler.
 * @param b Update status value.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetSetLookupParameterUpdated(
    dynetLookupParameter_t *param, DYNET_C_BOOL b);

/**
 * Sets update status of the parameter.
 * @param param Pointer of a handler.
 * @param retval Pointer to receive an update status value.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetIsLookupParameterUpdated(
    const dynetLookupParameter_t *param, DYNET_C_BOOL *retval);

#endif  // DYNET_C_MODEL_H_
