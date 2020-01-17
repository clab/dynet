#include <dynet_c/config.h>

#include <dynet/dynet.h>
#include <dynet_c/internal.h>
#include <dynet_c/graph.h>

using dynet_c::internal::to_c_ptr;
using dynet_c::internal::to_cpp_ptr;

DYNET_C_STATUS dynetCreateComputationGraph(
    dynetComputationGraph_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::ComputationGraph());
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetDeleteComputationGraph(dynetComputationGraph_t *cg) try {
  DYNET_C_CHECK_NOT_NULL(cg);
  delete to_cpp_ptr(cg);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetClearComputationGraph(dynetComputationGraph_t *cg) try {
  DYNET_C_CHECK_NOT_NULL(cg);
  to_cpp_ptr(cg)->clear();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSetComputationGraphCheckpoint(
    dynetComputationGraph_t *cg) try {
  DYNET_C_CHECK_NOT_NULL(cg);
  to_cpp_ptr(cg)->checkpoint();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetRevertComputationGraph(dynetComputationGraph_t *cg) try {
  DYNET_C_CHECK_NOT_NULL(cg);
  to_cpp_ptr(cg)->revert();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetForwardExprOnComputationGraph(
    dynetComputationGraph_t *cg, const dynetExpression_t *last,
    const dynetTensor_t **retval) try {
  DYNET_C_CHECK_NOT_NULL(cg);
  DYNET_C_CHECK_NOT_NULL(last);
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = to_c_ptr(&to_cpp_ptr(cg)->forward(*to_cpp_ptr(last)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetForwardExprIncrementallyOnComputationGraph(
    dynetComputationGraph_t *cg, const dynetExpression_t *last,
    const dynetTensor_t **retval) try {
  DYNET_C_CHECK_NOT_NULL(cg);
  DYNET_C_CHECK_NOT_NULL(last);
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = to_c_ptr(&to_cpp_ptr(cg)->incremental_forward(*to_cpp_ptr(last)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetExprValueFromComputationGraph(
    dynetComputationGraph_t *cg, const dynetExpression_t *expr,
    const dynetTensor_t **retval) try {
  DYNET_C_CHECK_NOT_NULL(cg);
  DYNET_C_CHECK_NOT_NULL(expr);
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = to_c_ptr(&to_cpp_ptr(cg)->get_value(*to_cpp_ptr(expr)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetExprGradientFromComputationGraph(
    dynetComputationGraph_t *cg, const dynetExpression_t *expr,
    const dynetTensor_t **retval) try {
  DYNET_C_CHECK_NOT_NULL(cg);
  DYNET_C_CHECK_NOT_NULL(expr);
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = to_c_ptr(&to_cpp_ptr(cg)->get_gradient(*to_cpp_ptr(expr)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetInvalidateComputationGraph(
    dynetComputationGraph_t *cg) try {
  DYNET_C_CHECK_NOT_NULL(cg);
  to_cpp_ptr(cg)->invalidate();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetBackwardExprOnComputationGraph(
    dynetComputationGraph_t *cg, const dynetExpression_t *last) try {
  DYNET_C_CHECK_NOT_NULL(cg);
  DYNET_C_CHECK_NOT_NULL(last);
  to_cpp_ptr(cg)->backward(*to_cpp_ptr(last));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetPrintComputationGraphViz(
    const dynetComputationGraph_t *cg) try {
  DYNET_C_CHECK_NOT_NULL(cg);
  to_cpp_ptr(cg)->print_graphviz();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetDumpComputationGraph(
    dynetComputationGraph_t *cg, const char *filename,
    DYNET_C_BOOL show_values, DYNET_C_BOOL show_gradients,
    DYNET_C_BOOL nan_check_only) try {
  DYNET_C_CHECK_NOT_NULL(cg);
  DYNET_C_CHECK_NOT_NULL(filename);
  to_cpp_ptr(cg)->dump(filename, show_values, show_gradients, nan_check_only);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS
