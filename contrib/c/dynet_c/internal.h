#ifndef DYNET_C_INTERNAL_H_
#define DYNET_C_INTERNAL_H_

#include <dynet/init.h>
#include <dynet/dim.h>
#include <dynet/tensor.h>
#include <dynet/model.h>
#include <dynet/io.h>
#include <dynet/param-init.h>
#include <dynet/dynet.h>
#include <dynet/training.h>
#include <dynet/devices.h>
#include <dynet/expr.h>
#include <dynet/rnn.h>
#include <dynet/cfsm-builder.h>

#include <dynet_c/define.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define DYNET_C_PTR_TO_PTR(cpp_name, c_name) \
inline c_name *to_c_ptr(dynet::cpp_name *instance) { \
  return reinterpret_cast<c_name*>(instance); \
} \
inline const c_name *to_c_ptr(const dynet::cpp_name *instance) { \
  return reinterpret_cast<const c_name*>(instance); \
} \
inline dynet::cpp_name *to_cpp_ptr(c_name *instance) { \
  return reinterpret_cast<dynet::cpp_name*>(instance); \
} \
inline const dynet::cpp_name *to_cpp_ptr(const c_name *instance) { \
  return reinterpret_cast<const dynet::cpp_name*>(instance); \
}

#define DYNET_C_VAL_TO_PTR(cpp_name, c_name) \
inline c_name *to_c_ptr_from_value(dynet::cpp_name &&instance) { \
  return reinterpret_cast<c_name*>( \
      new dynet::cpp_name(std::forward<dynet::cpp_name>(instance))); \
}

#define DYNET_C_HANDLE_EXCEPTIONS \
catch (const std::exception &err) { \
  return dynet_c::internal::ErrorHandler::get_instance().handle(err); \
}

#define DYNET_C_CHECK_NOT_NULL(var) \
if (!var) { \
  DYNET_C_THROW_ERROR("Argument `" #var "` must not be null."); \
}

#define DYNET_C_THROW_ERROR(cmds) { \
  std::stringstream ss; \
  ss << cmds; \
  throw dynet_c::internal::Error(__FILE__, __LINE__, ss.str()); \
}

struct dynetDynetParams;
struct dynetDim;
struct dynetTensor;
struct dynetParameter;
struct dynetLookupParameter;
struct dynetParameterCollection;
struct dynetTextFileSaver;
struct dynetTextFileLoader;
struct dynetParameterInit;
struct dynetComputationGraph;
struct dynetTrainer;
struct dynetDevice;
struct dynetExpression;
struct dynetRNNBuilder;
struct dynetSoftmaxBuilder;

namespace dynet_c {

namespace internal {

class Error : public std::exception {
  Error() = delete;

 public:
  Error(const std::string &file, std::uint32_t line, const std::string &message)
  : file_(file), line_(line), msg_(message) {
    std::stringstream ss;
    ss << file_ << ": " << line_ << ": " << msg_;
    full_msg_ = ss.str();
  }

  const char *what() const noexcept override { return full_msg_.c_str(); }

 private:
  std::string file_;
  std::uint32_t line_;
  std::string msg_;
  std::string full_msg_;
};

template<typename T>
using Throwable = typename std::enable_if<
    std::is_base_of<std::exception, T>::value>::type;

class ErrorHandler {
 public:
  ErrorHandler() noexcept : exception_(nullptr), message_("OK") {}
  ~ErrorHandler() = default;

  template<typename T, typename = Throwable<T>>
  ::DYNET_C_STATUS handle(const T &e) {
    exception_ = std::make_exception_ptr(e);
    message_ = e.what();
    return DYNET_C_ERROR;
  }

  std::exception rethrow() {
    if (has_exception()) {
      std::rethrow_exception(exception_);
    } else {
      throw std::bad_exception();
    }
  }

  void reset() noexcept {
    exception_ = nullptr;
    message_ = "OK";
  }

  bool has_exception() const noexcept {
    return !exception_;
  }

  const char *get_message() const noexcept {
    return message_.c_str();
  }

  static ErrorHandler &get_instance();

 private:
  std::exception_ptr exception_;
  std::string message_;
};

DYNET_C_PTR_TO_PTR(DynetParams, dynetDynetParams);
DYNET_C_VAL_TO_PTR(DynetParams, dynetDynetParams);
DYNET_C_PTR_TO_PTR(Dim, dynetDim);
DYNET_C_VAL_TO_PTR(Dim, dynetDim);
DYNET_C_PTR_TO_PTR(Tensor, dynetTensor);
DYNET_C_VAL_TO_PTR(Tensor, dynetTensor);
DYNET_C_PTR_TO_PTR(Parameter, dynetParameter);
DYNET_C_VAL_TO_PTR(Parameter, dynetParameter);
DYNET_C_PTR_TO_PTR(LookupParameter, dynetLookupParameter);
DYNET_C_VAL_TO_PTR(LookupParameter, dynetLookupParameter);
DYNET_C_PTR_TO_PTR(ParameterCollection, dynetParameterCollection);
DYNET_C_VAL_TO_PTR(ParameterCollection, dynetParameterCollection);
DYNET_C_PTR_TO_PTR(TextFileSaver, dynetTextFileSaver);
DYNET_C_PTR_TO_PTR(TextFileLoader, dynetTextFileLoader);
DYNET_C_PTR_TO_PTR(ParameterInit, dynetParameterInit);
DYNET_C_PTR_TO_PTR(ComputationGraph, dynetComputationGraph);
DYNET_C_PTR_TO_PTR(Trainer, dynetTrainer);
DYNET_C_PTR_TO_PTR(Device, dynetDevice);
DYNET_C_PTR_TO_PTR(Expression, dynetExpression);
DYNET_C_VAL_TO_PTR(Expression, dynetExpression);
DYNET_C_PTR_TO_PTR(RNNBuilder, dynetRNNBuilder);
DYNET_C_PTR_TO_PTR(SoftmaxBuilder, dynetSoftmaxBuilder);

template<typename S, typename T>
inline void move_vector_to_array_of_c_ptrs(
    std::vector<S> *src, T **target, std::size_t *size) {
  DYNET_C_CHECK_NOT_NULL(src);
  DYNET_C_CHECK_NOT_NULL(size);
  if (target) {
    if (*size < src->size()) {
      DYNET_C_THROW_ERROR("Size is not enough to move a vector.");
    }
    std::transform(std::make_move_iterator(src->begin()),
                   std::make_move_iterator(src->end()),
                   target,
                   [](S &&x) {
                     return to_c_ptr_from_value(std::forward<S>(x));
                   });
  } else {
    *size = src->size();
  }
}

template<typename S, typename T>
inline void copy_array_of_c_ptrs_to_vector(
    const S *const *src, std::size_t size, std::vector<T> *target) {
  DYNET_C_CHECK_NOT_NULL(src);
  DYNET_C_CHECK_NOT_NULL(target);
  const T *const *_src = reinterpret_cast<const T *const *>(src);
  const std::vector<const T*> src_v = std::vector<const T*>(_src, _src + size);
  std::transform(src_v.begin(),
                 src_v.end(),
                 std::back_inserter(*target),
                 [](const T *x) {
                   return *x;
                 });
}

template<typename T>
inline void copy_vector_to_array(
    const std::vector<T> &src, T *array, std::size_t *size) {
  if (array) {
    if (*size < src.size()) {
      DYNET_C_THROW_ERROR("Size is not enough to copy a vector.");
    }
    std::copy(src.begin(), src.end(), array);
  } else {
    *size = src.size();
  }
}

inline void copy_string_to_array(
    const std::string &str, char *buffer, std::size_t *size) {
  if (buffer) {
    if (*size <= str.length()) {
      DYNET_C_THROW_ERROR("Size is not enough to copy a string.");
    }
    std::snprintf(buffer, *size, "%s", str.c_str());
  } else {
    *size = str.length() + 1u;
  }
}

}  // namespace internal

}  // namespace dynet_c

#endif  // DYNET_C_INTERNAL_H_
