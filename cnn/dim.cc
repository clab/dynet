#include "cnn/dim.h"

#include <iostream>

using namespace std;

namespace cnn {

ostream& operator<<(ostream& os, const Dim& d) {
  os << '{';
  for (unsigned i = 0; i < d.nd; ++i) {
    if (i) os << ',';
    os << d.d[i];
  }
  if(d.bd != 1) os << 'X' << d.bd;
  return os << '}';
}

ostream& operator<<(ostream& os, const vector<Dim>& ds) {
  os << '[';
  for (unsigned i = 0; i < ds.size(); ++i)
    os << (i ? " " : "") << ds[i];
  return os << ']';
}

} // namespace cnn

