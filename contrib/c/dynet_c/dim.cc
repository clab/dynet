#include <dynet_c/config.h>

#include <dynet/dim.h>
#include <dynet_c/internal.h>
#include <dynet_c/dim.h>

#include <sstream>
#include <vector>

using dynet_c::internal::to_c_ptr;
using dynet_c::internal::to_cpp_ptr;
using dynet_c::internal::to_c_ptr_from_value;

DYNET_C_STATUS dynetCreateDim(dynetDim_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::Dim());
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateDimWithDimensions(
    const uint32_t *dims, size_t n, dynetDim_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(dims);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::Dim(std::vector<long>(dims, dims + n)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateDimWithDimensionsAndBatch(
    const uint32_t *dims, size_t n, uint32_t batch, dynetDim_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(dims);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(
      new dynet::Dim(std::vector<long>(dims, dims + n), batch));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCloneDim(
    const dynetDim_t *src, dynetDim_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(src);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::Dim(*to_cpp_ptr(src)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetDeleteDim(dynetDim_t *dim) try {
  DYNET_C_CHECK_NOT_NULL(dim);
  delete to_cpp_ptr(dim);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetDimTotalSize(
    const dynetDim_t *dim, uint32_t *retval) try {
  DYNET_C_CHECK_NOT_NULL(dim);
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(dim)->size();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetDimBatchSize(
    const dynetDim_t *dim, uint32_t *retval) try {
  DYNET_C_CHECK_NOT_NULL(dim);
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(dim)->batch_size();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSumDimDimensions(
    const dynetDim_t *dim, uint32_t *retval) try {
  DYNET_C_CHECK_NOT_NULL(dim);
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(dim)->sum_dims();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetTruncateDim(
    const dynetDim_t *dim, dynetDim_t **new_dim) try {
  DYNET_C_CHECK_NOT_NULL(dim);
  DYNET_C_CHECK_NOT_NULL(new_dim);
  *new_dim = to_c_ptr_from_value(to_cpp_ptr(dim)->truncate());
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetResizeDim(dynetDim_t *dim, uint32_t i) try {
  DYNET_C_CHECK_NOT_NULL(dim);
  to_cpp_ptr(dim)->resize(i);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetDimNDimensions(
    const dynetDim_t *dim, uint32_t *retval) try {
  DYNET_C_CHECK_NOT_NULL(dim);
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(dim)->ndims();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetDimRows(const dynetDim_t *dim, uint32_t *retval) try {
  DYNET_C_CHECK_NOT_NULL(dim);
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(dim)->rows();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetDimCols(const dynetDim_t *dim, uint32_t *retval) try {
  DYNET_C_CHECK_NOT_NULL(dim);
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(dim)->cols();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetDimBatchElems(
    const dynetDim_t *dim, uint32_t *retval) try {
  DYNET_C_CHECK_NOT_NULL(dim);
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(dim)->batch_elems();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetDimDimensionSize(
    const dynetDim_t *dim, uint32_t i, uint32_t *retval) try {
  DYNET_C_CHECK_NOT_NULL(dim);
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(dim)->size(i);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSetDimDimensionSize(
    dynetDim_t *dim, uint32_t i, uint32_t s) try {
  DYNET_C_CHECK_NOT_NULL(dim);
  to_cpp_ptr(dim)->set(i, s);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetTransposeDim(
    const dynetDim_t *dim, dynetDim_t **new_dim) try {
  DYNET_C_CHECK_NOT_NULL(dim);
  DYNET_C_CHECK_NOT_NULL(new_dim);
  *new_dim = to_c_ptr_from_value(to_cpp_ptr(dim)->transpose());
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetIsDimEqualTo(
    const dynetDim_t *dim, const dynetDim_t *other, DYNET_C_BOOL *retval) try {
  DYNET_C_CHECK_NOT_NULL(dim);
  DYNET_C_CHECK_NOT_NULL(other);
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = dynet::operator==(*to_cpp_ptr(dim), *to_cpp_ptr(other));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetIsNotDimEqualTo(
    const dynetDim_t *dim, const dynetDim_t *other, DYNET_C_BOOL *retval) try {
  DYNET_C_CHECK_NOT_NULL(dim);
  DYNET_C_CHECK_NOT_NULL(other);
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = dynet::operator!=(*to_cpp_ptr(dim), *to_cpp_ptr(other));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_API DYNET_C_STATUS dynetRepresentDimAsString(
    const dynetDim_t *dim, char *retval, size_t *size) try {
  DYNET_C_CHECK_NOT_NULL(dim);
  DYNET_C_CHECK_NOT_NULL(size);
  std::stringstream ss;
  dynet::operator<<(ss, *to_cpp_ptr(dim));
  dynet_c::internal::copy_string_to_array(ss.str(), retval, size);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS
