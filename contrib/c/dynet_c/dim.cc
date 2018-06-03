#include <dynet_c/config.h>

#include <dynet/dim.h>
#include <dynet_c/internal.h>
#include <dynet_c/dim.h>

using dynet::Dim;
using dynet_c::internal::to_c_ptr;
using dynet_c::internal::to_cpp_ptr;

DYNET_C_STATUS dynetCreateDim(dynetDim_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new Dim());
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetDeleteDim(dynetDim_t *dim) try {
  DYNET_C_CHECK_NOT_NULL(dim);
  delete to_cpp_ptr(dim);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS
