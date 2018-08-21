#include <dynet_c/config.h>

#include <vector>

#include <dynet/expr.h>
#include <dynet_c/internal.h>
#include <dynet_c/expr.h>

using dynet_c::internal::to_c_ptr;
using dynet_c::internal::to_cpp_ptr;
using dynet_c::internal::to_c_ptr_from_value;

DYNET_C_STATUS dynetCreateExpression(dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::Expression());
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetDeleteExpression(dynetExpression_t *expr) try {
  DYNET_C_CHECK_NOT_NULL(expr);
  delete to_cpp_ptr(expr);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetExpressionDim(
    const dynetExpression_t *expr, const dynetDim_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(expr);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(&to_cpp_ptr(expr)->dim());
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetExpressionValue(
    const dynetExpression_t *expr, const dynetTensor_t **tensor) try {
  DYNET_C_CHECK_NOT_NULL(expr);
  DYNET_C_CHECK_NOT_NULL(tensor);
  *tensor = to_c_ptr(&to_cpp_ptr(expr)->value());
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetExpressionGradient(
    const dynetExpression_t *expr, const dynetTensor_t **tensor) try {
  DYNET_C_CHECK_NOT_NULL(expr);
  DYNET_C_CHECK_NOT_NULL(tensor);
  *tensor = to_c_ptr(&to_cpp_ptr(expr)->gradient());
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyInputScalar(
    dynetComputationGraph_t *g, float s, dynetDevice_t *device,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(g);
  DYNET_C_CHECK_NOT_NULL(newobj);
  dynet::Device *device_ptr = device ?
      to_cpp_ptr(device) : dynet::default_device;
  *newobj = to_c_ptr_from_value(dynet::input(*to_cpp_ptr(g), s, device_ptr));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyInput(
    dynetComputationGraph_t *g, const dynetDim_t *d, const float *data,
    size_t n, dynetDevice_t *device, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(g);
  DYNET_C_CHECK_NOT_NULL(d);
  DYNET_C_CHECK_NOT_NULL(data);
  DYNET_C_CHECK_NOT_NULL(newobj);
  dynet::Device *device_ptr = device ?
      to_cpp_ptr(device) : dynet::default_device;
  *newobj = to_c_ptr_from_value(dynet::input(
      *to_cpp_ptr(g), *to_cpp_ptr(d), std::vector<float>(data, data + n),
      device_ptr));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyInputSparse(
    dynetComputationGraph_t *g, const dynetDim_t *d, const uint32_t *ids,
    size_t n_ids, const float *data, size_t n_data, float defdata,
    dynetDevice_t *device, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(g);
  DYNET_C_CHECK_NOT_NULL(d);
  DYNET_C_CHECK_NOT_NULL(ids);
  DYNET_C_CHECK_NOT_NULL(data);
  DYNET_C_CHECK_NOT_NULL(newobj);
  dynet::Device *device_ptr = device ?
      to_cpp_ptr(device) : dynet::default_device;
  *newobj = to_c_ptr_from_value(dynet::input(
      *to_cpp_ptr(g), *to_cpp_ptr(d), std::vector<uint32_t>(ids, ids + n_ids),
      std::vector<float>(data, data + n_data), defdata, device_ptr));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyOneHot(
    dynetComputationGraph_t *g, uint32_t d, const uint32_t *ids, size_t n,
    dynetDevice_t *device, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(g);
  DYNET_C_CHECK_NOT_NULL(ids);
  DYNET_C_CHECK_NOT_NULL(newobj);
  dynet::Device *device_ptr = device ?
      to_cpp_ptr(device) : dynet::default_device;
  *newobj = to_c_ptr_from_value(dynet::one_hot(
      *to_cpp_ptr(g), d, std::vector<uint32_t>(ids, ids + n), device_ptr));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyParameter(
    dynetComputationGraph_t *g, dynetParameter_t *p,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(g);
  DYNET_C_CHECK_NOT_NULL(p);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::parameter(*to_cpp_ptr(g), *to_cpp_ptr(p)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyLookupParameter(
    dynetComputationGraph_t *g, dynetLookupParameter_t *lp,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(g);
  DYNET_C_CHECK_NOT_NULL(lp);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::parameter(*to_cpp_ptr(g), *to_cpp_ptr(lp)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyConstParameter(
    dynetComputationGraph_t *g, dynetParameter_t *p,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(g);
  DYNET_C_CHECK_NOT_NULL(p);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::const_parameter(*to_cpp_ptr(g), *to_cpp_ptr(p)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyConstLookupParameter(
    dynetComputationGraph_t *g, dynetLookupParameter_t *lp,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(g);
  DYNET_C_CHECK_NOT_NULL(lp);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::const_parameter(*to_cpp_ptr(g), *to_cpp_ptr(lp)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyLookupOne(
    dynetComputationGraph_t *g, dynetLookupParameter_t *p, uint32_t index,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(g);
  DYNET_C_CHECK_NOT_NULL(p);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::lookup(*to_cpp_ptr(g), *to_cpp_ptr(p), index));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyLookup(
    dynetComputationGraph_t *g, dynetLookupParameter_t *p,
    const uint32_t *indices, size_t n, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(g);
  DYNET_C_CHECK_NOT_NULL(p);
  DYNET_C_CHECK_NOT_NULL(indices);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::lookup(*to_cpp_ptr(g), *to_cpp_ptr(p),
      std::vector<uint32_t>(indices, indices + n)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyConstLookupOne(
    dynetComputationGraph_t *g, dynetLookupParameter_t *p, uint32_t index,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(g);
  DYNET_C_CHECK_NOT_NULL(p);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::const_lookup(*to_cpp_ptr(g), *to_cpp_ptr(p), index));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyConstLookup(
    dynetComputationGraph_t *g, dynetLookupParameter_t *p,
    const uint32_t *indices, size_t n, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(g);
  DYNET_C_CHECK_NOT_NULL(p);
  DYNET_C_CHECK_NOT_NULL(indices);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::const_lookup(*to_cpp_ptr(g), *to_cpp_ptr(p),
      std::vector<uint32_t>(indices, indices + n)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyZeros(
    dynetComputationGraph_t *g, const dynetDim_t *d,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(g);
  DYNET_C_CHECK_NOT_NULL(d);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::zeros(*to_cpp_ptr(g), *to_cpp_ptr(d)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyOnes(
    dynetComputationGraph_t *g, const dynetDim_t *d,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(g);
  DYNET_C_CHECK_NOT_NULL(d);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::ones(*to_cpp_ptr(g), *to_cpp_ptr(d)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyConstant(
    dynetComputationGraph_t *g, const dynetDim_t *d, float val,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(g);
  DYNET_C_CHECK_NOT_NULL(d);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::constant(*to_cpp_ptr(g), *to_cpp_ptr(d), val));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyRandomNormal(
    dynetComputationGraph_t *g, const dynetDim_t *d, float mean, float stddev,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(g);
  DYNET_C_CHECK_NOT_NULL(d);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::random_normal(*to_cpp_ptr(g), *to_cpp_ptr(d), mean, stddev));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyRandomBernoulli(
    dynetComputationGraph_t *g, const dynetDim_t *d, float p, float scale,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(g);
  DYNET_C_CHECK_NOT_NULL(d);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::random_bernoulli(*to_cpp_ptr(g), *to_cpp_ptr(d), p, scale));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyRandomUniform(
    dynetComputationGraph_t *g, const dynetDim_t *d, float left, float right,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(g);
  DYNET_C_CHECK_NOT_NULL(d);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::random_uniform(*to_cpp_ptr(g), *to_cpp_ptr(d), left, right));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyRandomGumbel(
    dynetComputationGraph_t *g, const dynetDim_t *d, float mu, float beta,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(g);
  DYNET_C_CHECK_NOT_NULL(d);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::random_gumbel(*to_cpp_ptr(g), *to_cpp_ptr(d), mu, beta));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS
