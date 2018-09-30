#include <dynet_c/config.h>

#include <dynet/io.h>
#include <dynet_c/internal.h>
#include <dynet_c/io.h>

using dynet_c::internal::to_c_ptr;
using dynet_c::internal::to_cpp_ptr;
using dynet_c::internal::to_c_ptr_from_value;

DYNET_C_STATUS dynetCreateTextFileSaver(
    const char *filename, DYNET_C_BOOL append,
    dynetTextFileSaver_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(filename);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::TextFileSaver(filename, append));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetDeleteTextFileSaver(dynetTextFileSaver_t *saver) try {
  DYNET_C_CHECK_NOT_NULL(saver);
  delete to_cpp_ptr(saver);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSaveParameterCollection(
    dynetTextFileSaver_t *saver, const dynetParameterCollection_t *model,
    const char *key) try {
  DYNET_C_CHECK_NOT_NULL(saver);
  DYNET_C_CHECK_NOT_NULL(model);
  if (key) {
    to_cpp_ptr(saver)->save(*to_cpp_ptr(model), key);
  } else {
    to_cpp_ptr(saver)->save(*to_cpp_ptr(model));
  }
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSaveParameter(
    dynetTextFileSaver_t *saver, const dynetParameter_t *param,
    const char *key) try {
  DYNET_C_CHECK_NOT_NULL(saver);
  DYNET_C_CHECK_NOT_NULL(param);
  if (key) {
    to_cpp_ptr(saver)->save(*to_cpp_ptr(param), key);
  } else {
    to_cpp_ptr(saver)->save(*to_cpp_ptr(param));
  }
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSaveLookupParameter(
    dynetTextFileSaver_t *saver, const dynetLookupParameter_t *param,
    const char *key) try {
  DYNET_C_CHECK_NOT_NULL(saver);
  DYNET_C_CHECK_NOT_NULL(param);
  if (key) {
    to_cpp_ptr(saver)->save(*to_cpp_ptr(param), key);
  } else {
    to_cpp_ptr(saver)->save(*to_cpp_ptr(param));
  }
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateTextFileLoader(
    const char *filename, dynetTextFileLoader_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(filename);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::TextFileLoader(filename));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetDeleteTextFileLoader(dynetTextFileLoader_t *loader) try {
  DYNET_C_CHECK_NOT_NULL(loader);
  delete to_cpp_ptr(loader);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetPopulateParameterCollection(
    dynetTextFileLoader_t *loader, dynetParameterCollection_t *model,
    const char *key) try {
  DYNET_C_CHECK_NOT_NULL(loader);
  DYNET_C_CHECK_NOT_NULL(model);
  if (key) {
    to_cpp_ptr(loader)->populate(*to_cpp_ptr(model), key);
  } else {
    to_cpp_ptr(loader)->populate(*to_cpp_ptr(model));
  }
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetPopulateParameter(
    dynetTextFileLoader_t *loader, dynetParameter_t *param,
    const char *key) try {
  DYNET_C_CHECK_NOT_NULL(loader);
  DYNET_C_CHECK_NOT_NULL(param);
  if (key) {
    to_cpp_ptr(loader)->populate(*to_cpp_ptr(param), key);
  } else {
    to_cpp_ptr(loader)->populate(*to_cpp_ptr(param));
  }
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetPopulateLookupParameter(
    dynetTextFileLoader_t *loader, dynetLookupParameter_t *param,
    const char *key) try {
  DYNET_C_CHECK_NOT_NULL(loader);
  DYNET_C_CHECK_NOT_NULL(param);
  if (key) {
    to_cpp_ptr(loader)->populate(*to_cpp_ptr(param), key);
  } else {
    to_cpp_ptr(loader)->populate(*to_cpp_ptr(param));
  }
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetLoadParameterFromParameterCollection(
    dynetTextFileLoader_t *loader, dynetParameterCollection_t *model,
    const char *key, dynetParameter_t **param) try {
  DYNET_C_CHECK_NOT_NULL(loader);
  DYNET_C_CHECK_NOT_NULL(key);
  DYNET_C_CHECK_NOT_NULL(param);
  *param = to_c_ptr_from_value(
      to_cpp_ptr(loader)->load_param(*to_cpp_ptr(model), key));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetLoadLookupParameterFromParameterCollection(
    dynetTextFileLoader_t *loader, dynetParameterCollection_t *model,
    const char *key, dynetLookupParameter_t **param) try {
  DYNET_C_CHECK_NOT_NULL(loader);
  DYNET_C_CHECK_NOT_NULL(key);
  DYNET_C_CHECK_NOT_NULL(param);
  *param = to_c_ptr_from_value(
      to_cpp_ptr(loader)->load_lookup_param(*to_cpp_ptr(model), key));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS
