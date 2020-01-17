#ifndef DYNET_C_INIT_H_
#define DYNET_C_INIT_H_

#include <dynet_c/define.h>

/**
 * Opaque type of DynetParams.
 */
typedef struct dynetDynetParams dynetDynetParams_t;

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
DYNET_C_API DYNET_C_STATUS dynetDeleteDynetParams(dynetDynetParams_t *params);

/**
 * Sets the seed for random number generation.
 * @param params Pointer of a handler.
 * @param random_seed Random seed.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetSetDynetParamsRandomSeed(
    dynetDynetParams_t *params, uint32_t random_seed);

/**
 * Sets total memory to be allocated for DyNet.
 * @param params Pointer of a handler.
 * @param mem_descriptor Memory descriptor.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetSetDynetParamsMemDescriptor(
    dynetDynetParams_t *params, const char *mem_descriptor);

/**
 * Sets weight decay rate for L2 regularization.
 * @param params Pointer of a handler.
 * @param weight_decay Weight decay rate.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetSetDynetParamsWeightDecay(
    dynetDynetParams_t *params, float weight_decay);

/**
 * Specifies whether to autobatch or not.
 * @param params Pointer of a handler.
 * @param autobatch Whether to autobatch or not.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetSetDynetParamsAutobatch(
    dynetDynetParams_t *params, int32_t autobatch);

/**
 * Specifies whether to show autobatch debug info or not.
 * @param params Pointer of a handler.
 * @param profiling Whether to show autobatch debug info or not.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetSetDynetParamsProfiling(
    dynetDynetParams_t *params, int32_t profiling);

/**
 * Specifies whether to share parameters or not.
 * @param params Pointer of a handler.
 * @param shared_parameters Whether to share parameters or not.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetSetDynetParamsSharedParameters(
    dynetDynetParams_t *params, DYNET_C_BOOL shared_parameters);

/**
 * Specifies the number of requested GPUs.
 * @param params Pointer of a handler.
 * @param requested_gpus Number of requested GPUs.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetSetDynetParamsRequestedGpus(
    dynetDynetParams_t *params, int32_t requested_gpus);

/**
 * Builds a DynetParams object from command line arguments.
 * @param argc Command line arguments count
 * @param argv Command line arguments vector
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetExtractDynetParams(
    int32_t argc, char **argv, DYNET_C_BOOL shared_parameters,
    dynetDynetParams_t **newobj);

/**
 * Initializes DyNet.
 * @param params Pointer of a DynetParams.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetInitialize(dynetDynetParams_t *params);

/**
 * Resets random number generators.
 * @param seed Random seed.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetResetRng(uint32_t seed);

#endif  // DYNET_C_INIT_H_
