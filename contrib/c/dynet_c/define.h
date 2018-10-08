#ifndef DYNET_C_DEFINE_H_
#define DYNET_C_DEFINE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define DYNET_C_EXTERN extern "C"
#else
#define DYNET_C_EXTERN extern
#endif  // __cplusplus

#if defined(__GNUC__) || defined(__clang__)
#define DYNET_C_EXPORT __attribute__((visibility("default")))
#elif defined(_MSC_VER)
#ifdef DYNET_C_DLLEXPORT
#define DYNET_C_EXPORT __declspec(dllexport)
#else
#define DYNET_C_EXPORT __declspec(dllimport)
#endif  // DYNET_C_DLLEXPORT
#else
#define DYNET_C_EXPORT
#endif  // __GNUC__, __clang__, _MSC_VER

#define DYNET_C_API DYNET_C_EXTERN DYNET_C_EXPORT

/*
 * Boolean type.
 */
typedef uint32_t DYNET_C_BOOL;

/*
 * Boolean values.
 * `DYNET_C_TRUE` can not be compared with any `DYNET_C_BOOL` values.
 * Only substituting `DYNET_C_TRUE` to `DYNET_C_BOOL` variables is
 * allowed.
 */
#define DYNET_C_FALSE 0
#define DYNET_C_TRUE 1

/*
 * Return codes.
 */
typedef uint32_t DYNET_C_STATUS;
#define DYNET_C_OK 0
#define DYNET_C_ERROR -1

#endif  // DYNET_C_DEFINE_H_
