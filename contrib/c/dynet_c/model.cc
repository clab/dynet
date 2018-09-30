#include <dynet_c/config.h>

#include <dynet/devices.h>
#include <dynet/model.h>
#include <dynet_c/internal.h>
#include <dynet_c/model.h>

#include <string>

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

DYNET_C_STATUS dynetGetParameterGradients(
    dynetParameter_t *param, dynetTensor_t **tensor) try {
  DYNET_C_CHECK_NOT_NULL(param);
  DYNET_C_CHECK_NOT_NULL(tensor);
  *tensor = to_c_ptr(to_cpp_ptr(param)->gradients());
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

DYNET_C_STATUS dynetCreateLookupParameter(
    dynetLookupParameter_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::LookupParameter());
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetDeleteLookupParameter(dynetLookupParameter_t *param) try {
  DYNET_C_CHECK_NOT_NULL(param);
  delete to_cpp_ptr(param);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetFillLookupParameterWithZeros(
    dynetLookupParameter_t *param) try {
  DYNET_C_CHECK_NOT_NULL(param);
  to_cpp_ptr(param)->zero();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetLookupParameterDim(
    const dynetLookupParameter_t *param, dynetDim_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(param);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(to_cpp_ptr(param)->dim());
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetLookupParameterValues(
    dynetLookupParameter_t *param, dynetTensor_t **tensors, size_t *size) try {
  DYNET_C_CHECK_NOT_NULL(param);
  DYNET_C_CHECK_NOT_NULL(size);
  dynet_c::internal::move_vector_to_array_of_c_ptrs(
      to_cpp_ptr(param)->values(), tensors, size);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSetLookupParameterUpdated(
    dynetLookupParameter_t *param, DYNET_C_BOOL b) try {
  DYNET_C_CHECK_NOT_NULL(param);
  to_cpp_ptr(param)->set_updated(b);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetIsLookupParameterUpdated(
    const dynetLookupParameter_t *param, DYNET_C_BOOL *retval) try {
  DYNET_C_CHECK_NOT_NULL(param);
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(param)->is_updated();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateParameterCollection(
    dynetParameterCollection_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::ParameterCollection());
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetDeleteParameterCollection(
    dynetParameterCollection_t *pc) try {
  DYNET_C_CHECK_NOT_NULL(pc);
  delete to_cpp_ptr(pc);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetParameterCollectionGradientL2Norm(
    const dynetParameterCollection_t *pc, float *retval) try {
  DYNET_C_CHECK_NOT_NULL(pc);
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(pc)->gradient_l2_norm();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSetParameterCollectionWeightDecayLambda(
    dynetParameterCollection_t *pc, float lambda) try {
  DYNET_C_CHECK_NOT_NULL(pc);
  to_cpp_ptr(pc)->set_weight_decay_lambda(lambda);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetParameterCollectionWeightDecayLambda(
    const dynetParameterCollection_t *pc, float *retval) try {
  DYNET_C_CHECK_NOT_NULL(pc);
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(pc)->get_weight_decay_lambda();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetAddParametersToParameterCollection(
    dynetParameterCollection_t *pc, const dynetDim_t *d,
    const dynetParameterInit_t *init, const char *name, dynetDevice_t *device,
    dynetParameter_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(pc);
  DYNET_C_CHECK_NOT_NULL(d);
  DYNET_C_CHECK_NOT_NULL(newobj);
  const std::string name_str = name ? name : "";
  dynet::Device *device_ptr = device ?
      to_cpp_ptr(device) : dynet::default_device;
  if (init) {
    *newobj = to_c_ptr_from_value(to_cpp_ptr(pc)->add_parameters(
        *to_cpp_ptr(d), *to_cpp_ptr(init), name_str, device_ptr));
  } else {
    *newobj = to_c_ptr_from_value(to_cpp_ptr(pc)->add_parameters(
        *to_cpp_ptr(d), name_str, device_ptr));
  }
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetAddLookupParametersToParameterCollection(
    dynetParameterCollection_t *pc, uint32_t n, const dynetDim_t *d,
    const dynetParameterInit_t *init, const char *name, dynetDevice_t *device,
    dynetLookupParameter_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(pc);
  DYNET_C_CHECK_NOT_NULL(d);
  DYNET_C_CHECK_NOT_NULL(newobj);
  const std::string name_str = name ? name : "";
  dynet::Device *device_ptr = device ?
      to_cpp_ptr(device) : dynet::default_device;
  if (init) {
    *newobj = to_c_ptr_from_value(to_cpp_ptr(pc)->add_lookup_parameters(
        n, *to_cpp_ptr(d), *to_cpp_ptr(init), name_str, device_ptr));
  } else {
    *newobj = to_c_ptr_from_value(to_cpp_ptr(pc)->add_lookup_parameters(
        n, *to_cpp_ptr(d), name_str, device_ptr));
  }
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetAddSubcollectionToParameterCollection(
    dynetParameterCollection_t *pc, const char *name,
    dynetParameterCollection_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(pc);
  DYNET_C_CHECK_NOT_NULL(newobj);
  if (name) {
    *newobj = to_c_ptr_from_value(to_cpp_ptr(pc)->add_subcollection(name));
  } else {
    *newobj = to_c_ptr_from_value(to_cpp_ptr(pc)->add_subcollection());
  }
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetParameterCollectionParameterCount(
    const dynetParameterCollection_t *pc, size_t *retval) try {
  DYNET_C_CHECK_NOT_NULL(pc);
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(pc)->parameter_count();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS
