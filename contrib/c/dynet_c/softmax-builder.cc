#include <dynet_c/config.h>

#include <dynet/cfsm-builder.h>
#include <dynet_c/internal.h>
#include <dynet_c/softmax-builder.h>

#include <vector>

using dynet_c::internal::to_c_ptr;
using dynet_c::internal::to_cpp_ptr;
using dynet_c::internal::to_c_ptr_from_value;

DYNET_C_STATUS dynetDeleteSoftmaxBuilder(dynetSoftmaxBuilder_t *builder) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  delete to_cpp_ptr(builder);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetResetSoftmaxBuilderGraph(
    dynetSoftmaxBuilder_t *builder, dynetComputationGraph_t *cg,
    DYNET_C_BOOL update) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  DYNET_C_CHECK_NOT_NULL(cg);
  to_cpp_ptr(builder)->new_graph(*to_cpp_ptr(cg), update);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplySoftmaxBuilderNegLogSoftmaxOne(
    dynetSoftmaxBuilder_t *builder, const dynetExpression_t *rep,
    uint32_t classidx, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  DYNET_C_CHECK_NOT_NULL(rep);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      to_cpp_ptr(builder)->neg_log_softmax(*to_cpp_ptr(rep), classidx));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetApplySoftmaxBuilderNegLogSoftmax(
    dynetSoftmaxBuilder_t *builder, const dynetExpression_t *rep,
    const uint32_t *classidxs, size_t n, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  DYNET_C_CHECK_NOT_NULL(rep);
  DYNET_C_CHECK_NOT_NULL(classidxs);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      to_cpp_ptr(builder)->neg_log_softmax(
          *to_cpp_ptr(rep), std::vector<uint32_t>(classidxs, classidxs + n)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSampleFromSoftmaxBuilder(
    dynetSoftmaxBuilder_t *builder, const dynetExpression_t *rep,
    uint32_t *retval) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  DYNET_C_CHECK_NOT_NULL(rep);
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(builder)->sample(*to_cpp_ptr(rep));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetSoftmaxBuilderFullLogDistribution(
    dynetSoftmaxBuilder_t *builder, const dynetExpression_t *rep,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  DYNET_C_CHECK_NOT_NULL(rep);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      to_cpp_ptr(builder)->full_log_distribution(*to_cpp_ptr(rep)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetSoftmaxBuilderFullLogits(
    dynetSoftmaxBuilder_t *builder, const dynetExpression_t *rep,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  DYNET_C_CHECK_NOT_NULL(rep);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      to_cpp_ptr(builder)->full_logits(*to_cpp_ptr(rep)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetSoftmaxBuilderParameterCollection(
    dynetSoftmaxBuilder_t *builder, dynetParameterCollection_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(&to_cpp_ptr(builder)->get_parameter_collection());
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateStandardSoftmaxBuilder(
    uint32_t rep_dim, uint32_t num_classes, dynetParameterCollection_t *pc,
    DYNET_C_BOOL bias, dynetSoftmaxBuilder_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(pc);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::StandardSoftmaxBuilder(
      rep_dim, num_classes, *to_cpp_ptr(pc), bias));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateStandardSoftmaxBuilderFromParameters(
    dynetParameter_t *p_w, dynetParameter_t *p_b,
    dynetSoftmaxBuilder_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(p_w);
  DYNET_C_CHECK_NOT_NULL(newobj);
  if (p_b) {
    *newobj = to_c_ptr(
        new dynet::StandardSoftmaxBuilder(*to_cpp_ptr(p_w), *to_cpp_ptr(p_b)));
  } else {
    *newobj = to_c_ptr(new dynet::StandardSoftmaxBuilder(*to_cpp_ptr(p_w)));
  }
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS
