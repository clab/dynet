#include <dynet_c/config.h>

#include <dynet/param-init.h>
#include <dynet_c/internal.h>
#include <dynet_c/param-init.h>

#include <vector>

using dynet_c::internal::to_c_ptr;
using dynet_c::internal::to_cpp_ptr;

DYNET_C_STATUS dynetDeleteParameterInit(dynetParameterInit_t *init) try {
  DYNET_C_CHECK_NOT_NULL(init);
  delete to_cpp_ptr(init);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateParameterInitNormal(
    float m, float v, dynetParameterInit_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::ParameterInitNormal(m, v));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateParameterInitUniform(
    float l, float r, dynetParameterInit_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::ParameterInitUniform(l, r));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateParameterInitConst(
    float c, dynetParameterInit_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::ParameterInitConst(c));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateParameterInitIdentity(
    dynetParameterInit_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::ParameterInitIdentity());
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateParameterInitGlorot(
    DYNET_C_BOOL is_lookup, float gain, dynetParameterInit_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::ParameterInitGlorot(is_lookup, gain));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateParameterInitSaxe(
    float gain, dynetParameterInit_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::ParameterInitSaxe(gain));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateParameterInitFromFile(
    const char *f, dynetParameterInit_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::ParameterInitFromFile(f));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateParameterInitFromVector(
    const float *v, size_t n, dynetParameterInit_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(v);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(
      new dynet::ParameterInitFromVector(std::vector<float>(v, v + n)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS
