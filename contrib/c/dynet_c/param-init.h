#ifndef DYNET_C_PARAM_INIT_H_
#define DYNET_C_PARAM_INIT_H_

#include <dynet_c/define.h>

/**
 * Opaque type of ParameterInit.
 */
typedef struct dynetParameterInit dynetParameterInit_t;

/**
 * Deletes the ParameterInit object.
 * @param init Pointer of a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetDeleteParameterInit(dynetParameterInit_t *init);

/**
 * Creates a new ParameterInitNormal object.
 * @param m Mean of the gaussian distribution.
 * @param v Variance of the gaussian distribution.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateParameterInitNormal(
    float m, float v, dynetParameterInit_t **newobj);

/**
 * Creates a new ParameterInitUniform object.
 * @param l Lower bound of the interval.
 * @param r Upper bound of the interval.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateParameterInitUniform(
    float l, float r, dynetParameterInit_t **newobj);

/**
 * Creates a new ParameterInitConst object.
 * @param c Constant value.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateParameterInitConst(
    float c, dynetParameterInit_t **newobj);

/**
 * Creates a new ParameterInitIdentity object.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateParameterInitIdentity(
    dynetParameterInit_t **newobj);

/**
 * Creates a new ParameterInitGlorot object.
 * @param is_lookup Boolean value identifying the parameter as a LookupParameter.
 * @param gain Scaling parameter.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateParameterInitGlorot(
    DYNET_C_BOOL is_lookup, float gain, dynetParameterInit_t **newobj);

/**
 * Creates a new ParameterInitSaxe object.
 * @param gain Scaling parameter.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateParameterInitSaxe(
    float gain, dynetParameterInit_t **newobj);

/**
 * Creates a new ParameterInitFromFile object.
 * @param f File name.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateParameterInitFromFile(
    const char *f, dynetParameterInit_t **newobj);

/**
 * Creates a new ParameterInitFromVector object.
 * @param value List of values to be used.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateParameterInitFromVector(
    const float *v, size_t n, dynetParameterInit_t **newobj);

#endif  // DYNET_C_PARAM_INIT_H_
