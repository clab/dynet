#include <dynet_c/config.h>

#include <dynet/init.h>
#include <dynet_c/init.h>
#include <dynet_c/internal.h>

using dynet_c::internal::to_c_ptr;
using dynet_c::internal::to_cpp_ptr;
using dynet_c::internal::to_c_ptr_from_value;

DYNET_C_STATUS dynetCreateDynetParams(dynetDynetParams_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::DynetParams());
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetDeleteDynetParams(dynetDynetParams_t *params) try {
  DYNET_C_CHECK_NOT_NULL(params);
  delete to_cpp_ptr(params);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSetDynetParamsRandomSeed(
    dynetDynetParams_t *params, uint32_t random_seed) try {
  DYNET_C_CHECK_NOT_NULL(params);
  to_cpp_ptr(params)->random_seed = random_seed;
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSetDynetParamsMemDescriptor(
    dynetDynetParams_t *params, const char *mem_descriptor) try {
  DYNET_C_CHECK_NOT_NULL(params);
  DYNET_C_CHECK_NOT_NULL(mem_descriptor);
  to_cpp_ptr(params)->mem_descriptor = mem_descriptor;
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSetDynetParamsWeightDecay(
    dynetDynetParams_t *params, float weight_decay) try {
  DYNET_C_CHECK_NOT_NULL(params);
  to_cpp_ptr(params)->weight_decay = weight_decay;
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSetDynetParamsAutobatch(
    dynetDynetParams_t *params, int32_t autobatch) try {
  DYNET_C_CHECK_NOT_NULL(params);
  to_cpp_ptr(params)->autobatch = autobatch;
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSetDynetParamsProfiling(
    dynetDynetParams_t *params, float profiling) try {
  DYNET_C_CHECK_NOT_NULL(params);
  to_cpp_ptr(params)->profiling = profiling;
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSetDynetParamsWeightDecay(
    dynetDynetParams_t *params, DYNET_C_BOOL shared_parameters) try {
  DYNET_C_CHECK_NOT_NULL(params);
  to_cpp_ptr(params)->shared_parameters = shared_parameters;
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSetDynetParamsRequestedGpus(
    dynetDynetParams_t *params, int32_t requested_gpus) try {
  DYNET_C_CHECK_NOT_NULL(params);
  to_cpp_ptr(params)->requested_gpus = requested_gpus;
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetExtractDynetParams(
    int32_t argc, char **argv, DYNET_C_BOOL shared_parameters,
    dynetDynetParams_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(argv);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::extract_dynet_params(argc, argv, shared_parameters));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetInitialize(dynetDynetParams_t *params) try {
  DYNET_C_CHECK_NOT_NULL(params);
  dynet::initialize(*to_cpp_ptr(params));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetResetRng(uint32_t seed) try {
  dynet::reset_rng(seed);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS
