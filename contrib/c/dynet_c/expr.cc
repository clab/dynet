#include <dynet_c/config.h>

#include <dynet/expr.h>
#include <dynet_c/internal.h>
#include <dynet_c/expr.h>

#include <vector>

using dynet_c::internal::to_c_ptr;
using dynet_c::internal::to_cpp_ptr;
using dynet_c::internal::to_c_ptr_from_value;

#define DYNET_C_IMPL_UNARY_FUNC(name, cpp_func) \
DYNET_C_STATUS dynetApply##name( \
    const dynetExpression_t *x, dynetExpression_t **newobj) try { \
  DYNET_C_CHECK_NOT_NULL(x); \
  DYNET_C_CHECK_NOT_NULL(newobj); \
  *newobj = to_c_ptr_from_value(dynet::cpp_func(*to_cpp_ptr(x))); \
  return DYNET_C_OK; \
} DYNET_C_HANDLE_EXCEPTIONS \

#define DYNET_C_IMPL_BINARY_FUNC(name, cpp_func) \
DYNET_C_STATUS dynetApply##name( \
    const dynetExpression_t *x, const dynetExpression_t *y, \
    dynetExpression_t **newobj) try { \
  DYNET_C_CHECK_NOT_NULL(x); \
  DYNET_C_CHECK_NOT_NULL(y); \
  DYNET_C_CHECK_NOT_NULL(newobj); \
  *newobj = to_c_ptr_from_value( \
      dynet::cpp_func(*to_cpp_ptr(x), *to_cpp_ptr(y))); \
  return DYNET_C_OK; \
} DYNET_C_HANDLE_EXCEPTIONS \

DYNET_C_STATUS dynetCreateExpression(dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::Expression());
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCloneExpression(
    const dynetExpression_t *src, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(src);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::Expression(*to_cpp_ptr(src)));
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
    dynetComputationGraph_t *g, const dynetParameter_t *p,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(g);
  DYNET_C_CHECK_NOT_NULL(p);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::const_parameter(*to_cpp_ptr(g), *to_cpp_ptr(p)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyConstLookupParameter(
    dynetComputationGraph_t *g, const dynetLookupParameter_t *lp,
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
    dynetComputationGraph_t *g, const dynetLookupParameter_t *p,
    uint32_t index, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(g);
  DYNET_C_CHECK_NOT_NULL(p);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::const_lookup(*to_cpp_ptr(g), *to_cpp_ptr(p), index));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyConstLookup(
    dynetComputationGraph_t *g, const dynetLookupParameter_t *p,
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

DYNET_C_STATUS dynetApplyNegative(
    const dynetExpression_t *x, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(-(*to_cpp_ptr(x)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyAdd(
    const dynetExpression_t *x, const dynetExpression_t *y,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(y);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(*to_cpp_ptr(x) + *to_cpp_ptr(y));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyAddConst(
    const dynetExpression_t *x, float y, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(*to_cpp_ptr(x) + y);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyAddExpr(
    float x, const dynetExpression_t *y, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(y);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(x + *to_cpp_ptr(y));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplySubtract(
    const dynetExpression_t *x, const dynetExpression_t *y,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(y);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(*to_cpp_ptr(x) - *to_cpp_ptr(y));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplySubtractConst(
    const dynetExpression_t *x, float y, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(*to_cpp_ptr(x) - y);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplySubtractExpr(
    float x, const dynetExpression_t *y, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(y);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(x - *to_cpp_ptr(y));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyMultiply(
    const dynetExpression_t *x, const dynetExpression_t *y,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(y);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(*to_cpp_ptr(x) * *to_cpp_ptr(y));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyMultiplyConst(
    const dynetExpression_t *x, float y, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(*to_cpp_ptr(x) * y);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyMultiplyExpr(
    float x, const dynetExpression_t *y, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(y);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(x * *to_cpp_ptr(y));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyDivide(
    const dynetExpression_t *x, const dynetExpression_t *y,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(y);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(*to_cpp_ptr(x) / *to_cpp_ptr(y));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyDivideConst(
    const dynetExpression_t *x, float y, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(*to_cpp_ptr(x) / y);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyAffineTransform(
    const dynetExpression_t *const *xs, size_t n,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(xs);
  DYNET_C_CHECK_NOT_NULL(newobj);
  std::vector<dynet::Expression> _xs;
  dynet_c::internal::copy_array_of_c_ptrs_to_vector(xs, n, &_xs);
  *newobj = to_c_ptr_from_value(dynet::affine_transform(_xs));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplySum(
    const dynetExpression_t *const *xs, size_t n,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(xs);
  DYNET_C_CHECK_NOT_NULL(newobj);
  std::vector<dynet::Expression> _xs;
  dynet_c::internal::copy_array_of_c_ptrs_to_vector(xs, n, &_xs);
  *newobj = to_c_ptr_from_value(dynet::sum(_xs));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplySumElems(
    const dynetExpression_t *x, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::sum_elems(*to_cpp_ptr(x)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyMomentElems(
    const dynetExpression_t *x, uint32_t r, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::moment_elems(*to_cpp_ptr(x), r));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyMeanElems(
    const dynetExpression_t *x, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::mean_elems(*to_cpp_ptr(x)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyStdElems(
    const dynetExpression_t *x, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::std_elems(*to_cpp_ptr(x)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplySumBatches(
    const dynetExpression_t *x, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::sum_batches(*to_cpp_ptr(x)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyMomentBatches(
    const dynetExpression_t *x, uint32_t r, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::moment_batches(*to_cpp_ptr(x), r));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyMeanBatches(
    const dynetExpression_t *x, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::mean_batches(*to_cpp_ptr(x)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyStdBatches(
    const dynetExpression_t *x, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::std_batches(*to_cpp_ptr(x)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplySumDim(
    const dynetExpression_t *x, const uint32_t *dims, size_t n_dims,
    DYNET_C_BOOL b, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(dims);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::sum_dim(
      *to_cpp_ptr(x), std::vector<uint32_t>(dims, dims + n_dims), b));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyCumsum(
    const dynetExpression_t *x, uint32_t d, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(d);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::cumsum(*to_cpp_ptr(x), d));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyMomentDim(
    const dynetExpression_t *x, const uint32_t *dims, size_t n_dims,
    uint32_t r, DYNET_C_BOOL b, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(dims);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::moment_dim(
      *to_cpp_ptr(x), std::vector<uint32_t>(dims, dims + n_dims), r, b));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyMeanDim(
    const dynetExpression_t *x, const uint32_t *dims, size_t n_dims,
    DYNET_C_BOOL b, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(dims);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::mean_dim(
      *to_cpp_ptr(x), std::vector<uint32_t>(dims, dims + n_dims), b));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyStdDim(
    const dynetExpression_t *x, const uint32_t *dims, size_t n_dims,
    DYNET_C_BOOL b, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(dims);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::std_dim(
      *to_cpp_ptr(x), std::vector<uint32_t>(dims, dims + n_dims), b));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyAverage(
    const dynetExpression_t *const *xs, size_t n,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(xs);
  DYNET_C_CHECK_NOT_NULL(newobj);
  std::vector<dynet::Expression> _xs;
  dynet_c::internal::copy_array_of_c_ptrs_to_vector(xs, n, &_xs);
  *newobj = to_c_ptr_from_value(dynet::average(_xs));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_IMPL_UNARY_FUNC(Sqrt, sqrt);
DYNET_C_IMPL_UNARY_FUNC(Abs, abs);
DYNET_C_IMPL_UNARY_FUNC(Erf, erf);
DYNET_C_IMPL_UNARY_FUNC(Asin, asin);
DYNET_C_IMPL_UNARY_FUNC(Acos, acos);
DYNET_C_IMPL_UNARY_FUNC(Atan, atan);
DYNET_C_IMPL_UNARY_FUNC(Sin, sin);
DYNET_C_IMPL_UNARY_FUNC(Cos, cos);
DYNET_C_IMPL_UNARY_FUNC(Tan, tan);
DYNET_C_IMPL_UNARY_FUNC(Sinh, sinh);
DYNET_C_IMPL_UNARY_FUNC(Cosh, cosh);
DYNET_C_IMPL_UNARY_FUNC(Tanh, tanh);
DYNET_C_IMPL_UNARY_FUNC(Asinh, asinh);
DYNET_C_IMPL_UNARY_FUNC(Acosh, acosh);
DYNET_C_IMPL_UNARY_FUNC(Atanh, atanh);
DYNET_C_IMPL_UNARY_FUNC(Exp, exp);
DYNET_C_IMPL_UNARY_FUNC(Square, square);
DYNET_C_IMPL_UNARY_FUNC(Cube, cube);
DYNET_C_IMPL_UNARY_FUNC(LogSigmoid, log_sigmoid);
DYNET_C_IMPL_UNARY_FUNC(Lgamma, lgamma);
DYNET_C_IMPL_UNARY_FUNC(Log, log);
DYNET_C_IMPL_UNARY_FUNC(Logistic, logistic);
DYNET_C_IMPL_UNARY_FUNC(Rectify, rectify);

DYNET_C_STATUS dynetApplyElu(
    const dynetExpression_t *x, float alpha, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::elu(*to_cpp_ptr(x), alpha));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_IMPL_UNARY_FUNC(Selu, selu);

DYNET_C_STATUS dynetApplySilu(
    const dynetExpression_t *x, float beta, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::silu(*to_cpp_ptr(x), beta));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_IMPL_UNARY_FUNC(Softsign, softsign);

DYNET_C_IMPL_BINARY_FUNC(Pow, pow);
DYNET_C_IMPL_BINARY_FUNC(Bmin, min);
DYNET_C_IMPL_BINARY_FUNC(Bmax, max);

DYNET_C_STATUS dynetApplyMax(
    const dynetExpression_t *const *xs, size_t n,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(xs);
  DYNET_C_CHECK_NOT_NULL(newobj);
  std::vector<dynet::Expression> _xs;
  dynet_c::internal::copy_array_of_c_ptrs_to_vector(xs, n, &_xs);
  *newobj = to_c_ptr_from_value(dynet::max(_xs));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_IMPL_BINARY_FUNC(DotProduct, dot_product);

DYNET_C_STATUS dynetApplyCircConv(
    const dynetExpression_t *u, const dynetExpression_t *v,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(u);
  DYNET_C_CHECK_NOT_NULL(v);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::circ_conv(*to_cpp_ptr(u), *to_cpp_ptr(v)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyCircCorr(
    const dynetExpression_t *u, const dynetExpression_t *v,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(u);
  DYNET_C_CHECK_NOT_NULL(v);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::circ_corr(*to_cpp_ptr(u), *to_cpp_ptr(v)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_IMPL_BINARY_FUNC(Cmult, cmult);
DYNET_C_IMPL_BINARY_FUNC(Cdiv, cdiv);

DYNET_C_STATUS dynetApplyColwiseAdd(
    const dynetExpression_t *x, const dynetExpression_t *bias,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(bias);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::colwise_add(*to_cpp_ptr(x), *to_cpp_ptr(bias)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyRoundWithZeroGradientMode(
    const dynetExpression_t *x, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::round(*to_cpp_ptr(x), dynet::zero_gradient));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyRoundWithStraightThroughGradientMode(
    const dynetExpression_t *x, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::round(*to_cpp_ptr(x), dynet::straight_through_gradient));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyCeilWithZeroGradientMode(
    const dynetExpression_t *x, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::ceil(*to_cpp_ptr(x), dynet::zero_gradient));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyCeilWithStraightThroughGradientMode(
    const dynetExpression_t *x, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::ceil(*to_cpp_ptr(x), dynet::straight_through_gradient));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyFloorWithZeroGradientMode(
    const dynetExpression_t *x, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::floor(*to_cpp_ptr(x), dynet::zero_gradient));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyFloorWithStraightThroughGradientMode(
    const dynetExpression_t *x, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::floor(*to_cpp_ptr(x), dynet::straight_through_gradient));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplySoftmax(
    const dynetExpression_t *x, uint32_t d, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::softmax(*to_cpp_ptr(x), d));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_IMPL_UNARY_FUNC(LogSoftmax, log_softmax);

DYNET_C_STATUS dynetApplyRestrictedLogSoftmax(
    const dynetExpression_t *x, const uint32_t *restriction, size_t n,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(restriction);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::log_softmax(
      *to_cpp_ptr(x), std::vector<uint32_t>(restriction, restriction + n)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyLogsumexpDim(
    const dynetExpression_t *x, uint32_t d, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::logsumexp_dim(*to_cpp_ptr(x), d));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyLogsumexp(
    const dynetExpression_t *const *xs, size_t n,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(xs);
  DYNET_C_CHECK_NOT_NULL(newobj);
  std::vector<dynet::Expression> _xs;
  dynet_c::internal::copy_array_of_c_ptrs_to_vector(xs, n, &_xs);
  *newobj = to_c_ptr_from_value(dynet::logsumexp(_xs));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyPickneglogsoftmaxOne(
    const dynetExpression_t *x, uint32_t v, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::pickneglogsoftmax(*to_cpp_ptr(x), v));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyPickneglogsoftmax(
    const dynetExpression_t *x, const uint32_t *v, size_t n,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(v);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::pickneglogsoftmax(
      *to_cpp_ptr(x), std::vector<uint32_t>(v, v + n)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyHingeOne(
    const dynetExpression_t *x, uint32_t index, float m,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::hinge(*to_cpp_ptr(x), index, m));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyHinge(
    const dynetExpression_t *x, const uint32_t *indices, size_t n, float m,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(indices);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::hinge(
      *to_cpp_ptr(x), std::vector<uint32_t>(indices, indices + n), m));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyHingeDimOne(
    const dynetExpression_t *x, const uint32_t *indices, size_t n, uint32_t d,
    float m, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(indices);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::hinge_dim(
      *to_cpp_ptr(x), std::vector<uint32_t>(indices, indices + n), d, m));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyHingeDim(
    const dynetExpression_t *x, const uint32_t *indices, size_t n, uint32_t d,
    float m, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(indices);
  DYNET_C_CHECK_NOT_NULL(newobj);
  const dynet::Expression *cpp_x = to_cpp_ptr(x);
  uint32_t batch = cpp_x->dim().batch_size();
  uint32_t n_elems = n / batch;
  std::vector<std::vector<uint32_t>> indices_m;
  indices_m.reserve(batch);
  for (uint32_t i = 0; i < batch; i++) {
    std::vector<uint32_t> v;
    v.reserve(n_elems);
    for (uint32_t j = 0; j < n_elems; j++) {
      v.push_back(*(indices + i * n_elems + j));
    }
    indices_m.push_back(v);
  }
  *newobj = to_c_ptr_from_value(dynet::hinge_dim(*cpp_x, indices_m, d, m));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_IMPL_UNARY_FUNC(Sparsemax, sparsemax);

DYNET_C_STATUS dynetApplySparsemaxLoss(
    const dynetExpression_t *x, const uint32_t *target_support, size_t n,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(target_support);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::sparsemax_loss(
      *to_cpp_ptr(x),
      std::vector<uint32_t>(target_support, target_support + n)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_IMPL_BINARY_FUNC(ConstrainedSoftmax, constrained_softmax);
DYNET_C_IMPL_UNARY_FUNC(SquaredNorm, squared_norm);
DYNET_C_IMPL_UNARY_FUNC(L2Norm, l2_norm);
DYNET_C_IMPL_BINARY_FUNC(SquaredDistance, squared_distance);
DYNET_C_IMPL_BINARY_FUNC(L1Distance, l1_distance);

DYNET_C_STATUS dynetApplyHuberDistance(
    const dynetExpression_t *x, const dynetExpression_t *y, float c,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(y);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::huber_distance(*to_cpp_ptr(x), *to_cpp_ptr(y), c));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_IMPL_BINARY_FUNC(BinaryLogLoss, binary_log_loss);

DYNET_C_STATUS dynetApplyPairwiseRankLoss(
    const dynetExpression_t *x, const dynetExpression_t *y, float m,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(y);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::pairwise_rank_loss(*to_cpp_ptr(x), *to_cpp_ptr(y), m));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyPoissonLoss(
    const dynetExpression_t *x, uint32_t y, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::poisson_loss(*to_cpp_ptr(x), y));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_IMPL_UNARY_FUNC(Nobackprop, nobackprop);
DYNET_C_IMPL_UNARY_FUNC(FlipGradient, flip_gradient);

DYNET_C_STATUS dynetApplyScaleGradient(
    const dynetExpression_t *x, float lambd, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::scale_gradient(*to_cpp_ptr(x), lambd));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyArgmaxWithZeroGradientMode(
    const dynetExpression_t *x, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::argmax(*to_cpp_ptr(x), dynet::zero_gradient));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyArgmaxWithStraightThroughGradientMode(
    const dynetExpression_t *x, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::argmax(*to_cpp_ptr(x), dynet::straight_through_gradient));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyReshape(
    const dynetExpression_t *x, const dynetDim_t *d,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(d);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::reshape(*to_cpp_ptr(x), *to_cpp_ptr(d)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyTranspose(
    const dynetExpression_t *x, const uint32_t *dims, size_t n,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  if (dims) {
    *newobj = to_c_ptr_from_value(dynet::transpose(
        *to_cpp_ptr(x), std::vector<uint32_t>(dims, dims + n)));
  } else {
    *newobj = to_c_ptr_from_value(dynet::transpose(*to_cpp_ptr(x)));
  }
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplySelectRows(
    const dynetExpression_t *x, const uint32_t *rows, size_t n,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(rows);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::select_rows(
      *to_cpp_ptr(x), std::vector<uint32_t>(rows, rows + n)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplySelectCols(
    const dynetExpression_t *x, const uint32_t *cols, size_t n,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(cols);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::select_cols(
      *to_cpp_ptr(x), std::vector<uint32_t>(cols, cols + n)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyPickOne(
    const dynetExpression_t *x, uint32_t v, uint32_t d,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::pick(*to_cpp_ptr(x), v, d));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyPick(
    const dynetExpression_t *x, const uint32_t *v, size_t n, uint32_t d,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(v);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::pick(
      *to_cpp_ptr(x), std::vector<uint32_t>(v, v + n), d));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyPickRange(
    const dynetExpression_t *x, uint32_t s, uint32_t e, uint32_t d,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::pick_range(*to_cpp_ptr(x), s, e, d));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyPickBatchElem(
    const dynetExpression_t *x, uint32_t v, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::pick_batch_elem(*to_cpp_ptr(x), v));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyPickBatchElems(
    const dynetExpression_t *x, const uint32_t *v, size_t n,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(v);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::pick_batch_elems(
      *to_cpp_ptr(x), std::vector<uint32_t>(v, v + n)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyStridedSelect(
    const dynetExpression_t *x, const int32_t *strides, size_t n_strides,
    const int32_t *from, size_t n_from, const int32_t *to, size_t n_to,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(strides);
  DYNET_C_CHECK_NOT_NULL(newobj);
  std::vector<int32_t> from_v = from ?
      std::vector<int32_t>(from, from + n_from) : std::vector<int32_t>();
  std::vector<int32_t> to_v = to ?
      std::vector<int32_t>(to, to + n_to) : std::vector<int32_t>();
  *newobj = to_c_ptr_from_value(dynet::strided_select(
      *to_cpp_ptr(x), std::vector<int32_t>(strides, strides + n_strides),
      from_v, to_v));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyConcatenateToBatch(
    const dynetExpression_t *const *xs, size_t n,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(xs);
  DYNET_C_CHECK_NOT_NULL(newobj);
  std::vector<dynet::Expression> _xs;
  dynet_c::internal::copy_array_of_c_ptrs_to_vector(xs, n, &_xs);
  *newobj = to_c_ptr_from_value(dynet::concatenate_to_batch(_xs));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyConcatenateCols(
    const dynetExpression_t *const *xs, size_t n,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(xs);
  DYNET_C_CHECK_NOT_NULL(newobj);
  std::vector<dynet::Expression> _xs;
  dynet_c::internal::copy_array_of_c_ptrs_to_vector(xs, n, &_xs);
  *newobj = to_c_ptr_from_value(dynet::concatenate_cols(_xs));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyConcatenate(
    const dynetExpression_t *const *xs, size_t n, uint32_t d,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(xs);
  DYNET_C_CHECK_NOT_NULL(newobj);
  std::vector<dynet::Expression> _xs;
  dynet_c::internal::copy_array_of_c_ptrs_to_vector(xs, n, &_xs);
  *newobj = to_c_ptr_from_value(dynet::concatenate(_xs, d));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyMaxDim(
    const dynetExpression_t *x, uint32_t d, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::max_dim(*to_cpp_ptr(x), d));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyMinDim(
    const dynetExpression_t *x, uint32_t d, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::min_dim(*to_cpp_ptr(x), d));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyNoise(
    const dynetExpression_t *x, float stddev, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::noise(*to_cpp_ptr(x), stddev));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyDropout(
    const dynetExpression_t *x, float p, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::dropout(*to_cpp_ptr(x), p));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyDropoutDim(
    const dynetExpression_t *x, uint32_t d, float p,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::dropout_dim(*to_cpp_ptr(x), d, p));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyDropoutBatch(
    const dynetExpression_t *x, float p, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::dropout_batch(*to_cpp_ptr(x), p));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyBlockDropout(
    const dynetExpression_t *x, float p, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::block_dropout(*to_cpp_ptr(x), p));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyFilter1dNarrow(
    const dynetExpression_t *x, const dynetExpression_t *f,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(f);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::filter1d_narrow(*to_cpp_ptr(x), *to_cpp_ptr(f)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyKmaxPooling(
    const dynetExpression_t *x, uint32_t k, uint32_t d,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::kmax_pooling(*to_cpp_ptr(x), k, d));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyFoldRows(
    const dynetExpression_t *x, uint32_t nrows,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::fold_rows(*to_cpp_ptr(x), nrows));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_IMPL_UNARY_FUNC(AverageCols, average_cols);

DYNET_C_STATUS dynetApplyKmhNgram(
    const dynetExpression_t *x, uint32_t n, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::kmh_ngram(*to_cpp_ptr(x), n));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyConv2d(
    const dynetExpression_t *x, const dynetExpression_t *f,
    const uint32_t *stride, size_t n, DYNET_C_BOOL is_valid,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(f);
  DYNET_C_CHECK_NOT_NULL(stride);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::conv2d(*to_cpp_ptr(x), *to_cpp_ptr(f),
          std::vector<uint32_t>(stride, stride + n), is_valid));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyConv2dWithBias(
    const dynetExpression_t *x, const dynetExpression_t *f,
    const dynetExpression_t *b, const uint32_t *stride, size_t n,
    DYNET_C_BOOL is_valid, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(f);
  DYNET_C_CHECK_NOT_NULL(b);
  DYNET_C_CHECK_NOT_NULL(stride);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::conv2d(*to_cpp_ptr(x), *to_cpp_ptr(f), *to_cpp_ptr(b),
          std::vector<uint32_t>(stride, stride + n), is_valid));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyMaxpooling2d(
    const dynetExpression_t *x, const uint32_t *ksize, size_t n_ksize,
    const uint32_t *stride, size_t n_stride, DYNET_C_BOOL is_valid,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(ksize);
  DYNET_C_CHECK_NOT_NULL(stride);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::maxpooling2d(
      *to_cpp_ptr(x), std::vector<uint32_t>(ksize, ksize + n_ksize),
      std::vector<uint32_t>(stride, stride + n_stride), is_valid));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_IMPL_BINARY_FUNC(Contract3d1d, contract3d_1d);

DYNET_C_STATUS dynetApplyContract3d1dWithBias(
    const dynetExpression_t *x, const dynetExpression_t *y,
    const dynetExpression_t *b, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(y);
  DYNET_C_CHECK_NOT_NULL(b);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::contract3d_1d(*to_cpp_ptr(x), *to_cpp_ptr(y), *to_cpp_ptr(b)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyContract3d1d1d(
    const dynetExpression_t *x, const dynetExpression_t *y,
    const dynetExpression_t *z, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(y);
  DYNET_C_CHECK_NOT_NULL(z);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::contract3d_1d_1d(*to_cpp_ptr(x), *to_cpp_ptr(y), *to_cpp_ptr(z)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplyContract3d1d1dWithBias(
    const dynetExpression_t *x, const dynetExpression_t *y,
    const dynetExpression_t *z, const dynetExpression_t *b,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(y);
  DYNET_C_CHECK_NOT_NULL(z);
  DYNET_C_CHECK_NOT_NULL(b);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(dynet::contract3d_1d_1d(
      *to_cpp_ptr(x), *to_cpp_ptr(y), *to_cpp_ptr(z), *to_cpp_ptr(b)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_IMPL_UNARY_FUNC(Inverse, inverse);

DYNET_C_IMPL_UNARY_FUNC(Logdet, logdet);

DYNET_C_IMPL_BINARY_FUNC(TraceOfProduct, trace_of_product);

DYNET_C_STATUS dynetApplyLayerNorm(
    const dynetExpression_t *x, const dynetExpression_t *g,
    const dynetExpression_t *b, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(g);
  DYNET_C_CHECK_NOT_NULL(b);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::layer_norm(*to_cpp_ptr(x), *to_cpp_ptr(g), *to_cpp_ptr(b)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_IMPL_BINARY_FUNC(WeightNorm, weight_norm);

DYNET_C_STATUS dynetApplyToDevice(
    const dynetExpression_t *x, dynetDevice_t *device,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(device);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      dynet::to_device(*to_cpp_ptr(x), to_cpp_ptr(device)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

#undef DYNET_C_IMPL_UNARY_FUNC
#undef DYNET_C_IMPL_BINARY_FUNC
