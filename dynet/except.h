#ifndef DYNET_EXCEPT_H_
#define DYNET_EXCEPT_H_

#include <stdexcept>

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

#endif
