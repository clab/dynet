#include "cnn/backends/thpp/tensor.h"

#include <iostream>

using namespace std;

ostream& operator<<(ostream& os, const Dim& d) {
  os << '{';
  for (unsigned i = 0; i < d.v.size(); ++i)
    os << (i ? ", " : "") << d.v[i];
  return os << '}';
}

