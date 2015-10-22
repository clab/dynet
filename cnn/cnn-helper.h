#ifndef CNN_HELPER_H_
#define CNN_HELPER_H_

#include <string>

/// helper functions

namespace cnn {

/**
    this fix a compilation problem in cygwin
*/
#if defined(__CYGWIN__)
  template <typename T>
    inline std::string to_string(T value)
    {
      std::ostringstream os;
      os << value;
      return os.str();
    }
#endif

} // namespace cnn

#endif
