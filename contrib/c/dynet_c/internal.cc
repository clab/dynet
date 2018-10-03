#include <dynet_c/config.h>

#include <dynet_c/internal.h>

namespace dynet_c {

namespace internal {

static thread_local ErrorHandler error_handler;

ErrorHandler &ErrorHandler::get_instance() {
  return error_handler;
}

}  // namespace internal

}  // namespace dynet_c
