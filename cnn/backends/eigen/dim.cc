#include "cnn/backends/eigen/dim.h"

#include <iostream>

namespace cnn {

std::ostream& operator<<(std::ostream& os, const Dim& d) {
  os << '{';
  int c = 0;
  for (auto v : d.d) {
    if (!v) break;
    if (c) os << ',';
    os << v;
    ++c;
  }
  return os << '}';
}

} // namespace cnn

