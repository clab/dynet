#include <dynet_c/config.h>

#include <dynet/model.h>
#include <dynet_c/internal.h>
#include <dynet_c/model.h>

using dynet_c::internal::to_c_ptr;
using dynet_c::internal::to_cpp_ptr;
using dynet_c::internal::to_c_ptr_from_value;

DYNET_C_STATUS dynetCreateParameter(dynetParameter_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::Parameter());
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetDeleteParameter(dynetParameter_t *param) try {
  DYNET_C_CHECK_NOT_NULL(param);
  delete to_cpp_ptr(param);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetFillParameterWithZeros(dynetParameter_t *param) try {
  DYNET_C_CHECK_NOT_NULL(param);
  to_cpp_ptr(param)->zero();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetParameterDim(
    const dynetParameter_t *param, dynetDim_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(param);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(to_cpp_ptr(param)->dim());
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetParameterValues(
    dynetParameter_t *param, dynetTensor_t **tensor) try {
  DYNET_C_CHECK_NOT_NULL(param);
  DYNET_C_CHECK_NOT_NULL(tensor);
  *tensor = to_c_ptr(to_cpp_ptr(param)->values());
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSetParameterUpdated(
    dynetParameter_t *param, DYNET_C_BOOL b) try {
  DYNET_C_CHECK_NOT_NULL(param);
  to_cpp_ptr(param)->set_updated(b);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetIsParameterUpdated(
    const dynetParameter_t *param, DYNET_C_BOOL *retval) try {
  DYNET_C_CHECK_NOT_NULL(param);
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(param)->is_updated();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS
