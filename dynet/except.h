#ifndef DYNET_EXCEPT_H_
#define DYNET_EXCEPT_H_

#include <stdexcept>
#include <sstream>

namespace dynet {

// if DYNET exhausts its memory pool
class out_of_memory : public std::runtime_error {
 public:
  out_of_memory(const std::string& what_arg) : runtime_error(what_arg) {}
};

// this error occurs when some logic is
// attempted to execut on a CUDA backend but the
// logic has not been implemented.
class cuda_not_implemented : public std::logic_error {
 public:
  cuda_not_implemented(const std::string& what_arg) : logic_error(what_arg) {}
};

// this is thrown when cuda returns an error (bad arguments, memory, state, etc)
class cuda_exception : public std::runtime_error {
 public:
  cuda_exception(const std::string& what_arg) : runtime_error(what_arg) {}
};
} // namespace dynet

#if !defined(DYNET_DEBUG_LEVEL) || DYNET_DEBUG_LEVEL == 1
  #define DYNET_INVALID_ARG(msg) do {      \
    std::ostringstream oss;                 \
    oss << msg;                             \
    throw std::invalid_argument(oss.str()); \
  } while (0);

  #define DYNET_INVALID_ARG_CHECK(cond, msg) do { \
    if (cond) {                                    \
      std::ostringstream oss;                      \
      oss << msg;                                  \
      throw std::invalid_argument(oss.str()); }    \
  } while (0);

  #define DYNET_ASSERT(expr, msg)

#elif DYNET_DEBUG_LEVEL == 0
  #define DYNET_INVALID_ARG(msg)
  #define DYNET_INVALID_ARG_CHECK(cond, msg)
  #define DYNET_ASSERT(expr, msg)

#else
  #define DYNET_INVALID_ARG(msg) do {       \
    std::ostringstream oss;                 \
    oss << msg;                             \
    throw std::invalid_argument(oss.str()); \
  } while (0);

  #define DYNET_INVALID_ARG_CHECK(cond, msg) do { \
    if (cond) {                                   \
      std::ostringstream oss;                     \
      oss << msg;                                 \
      throw std::invalid_argument(oss.str()); }   \
  } while (0);

  #define DYNET_ASSERT(expr, msg) do {       \
    if(!(expr)) {                            \
      std::ostringstream oss;                \
      oss << msg;                            \
      throw std::runtime_error(oss.str()); } \
  } while (0);
#endif

#define DYNET_RUNTIME_ERR(msg) do {             \
    std::ostringstream oss;                     \
    oss << msg;                                 \
    throw std::runtime_error(oss.str()); }      \
  while (0);

#endif
