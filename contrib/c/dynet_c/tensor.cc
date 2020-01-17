#include <dynet_c/config.h>

#include <dynet/tensor.h>
#include <dynet_c/internal.h>
#include <dynet_c/tensor.h>

#include <sstream>

using dynet_c::internal::to_c_ptr;
using dynet_c::internal::to_cpp_ptr;

DYNET_C_STATUS dynetDeleteTensor(dynetTensor_t *tensor) try {
  DYNET_C_CHECK_NOT_NULL(tensor);
  delete to_cpp_ptr(tensor);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetTensorDim(
    const dynetTensor_t *tensor, dynetDim_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(tensor);
  DYNET_C_CHECK_NOT_NULL(newobj);
  dynet::Dim *d = new dynet::Dim(to_cpp_ptr(tensor)->d);
  *newobj = to_c_ptr(d);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetEvaluateTensorAsScalar(
    const dynetTensor_t *tensor, float *retval) try {
  DYNET_C_CHECK_NOT_NULL(tensor);
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = dynet::as_scalar(*to_cpp_ptr(tensor));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetEvaluateTensorAsArray(
    const dynetTensor_t *tensor, float *retval, size_t *size) try {
  DYNET_C_CHECK_NOT_NULL(tensor);
  DYNET_C_CHECK_NOT_NULL(size);
  dynet_c::internal::copy_vector_to_array(
      dynet::as_vector(*to_cpp_ptr(tensor)), retval, size);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_API DYNET_C_STATUS dynetRepresentTensorAsString(
    const dynetTensor_t *tensor, char *retval, size_t *size) try {
  DYNET_C_CHECK_NOT_NULL(tensor);
  DYNET_C_CHECK_NOT_NULL(size);
  std::stringstream ss;
  dynet::operator<<(ss, *to_cpp_ptr(tensor));
  dynet_c::internal::copy_string_to_array(ss.str(), retval, size);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS
