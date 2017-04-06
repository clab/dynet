#include "dynet/dim.h"

#include <iostream>

using namespace std;

namespace dynet {

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

istream& operator>>(istream& is, Dim& d) {
  char place_holder;
  int nd = 0;
  is >> place_holder;
  d.resize(DYNET_MAX_TENSOR_DIM);
  bool batch_flag = false;
  unsigned i = 0;
  for (; i < DYNET_MAX_TENSOR_DIM + 1; ++i) {
    if (i) {
      is >> place_holder;
      if (place_holder == 'X') {
        batch_flag = true;
        break;
      } else if (place_holder == '}') {
        break;
      }
    }
    is >> d.d[i];
  }
  d.resize(i);
  if (batch_flag) {
    is >> d.bd >> place_holder;
  }
  return is;
}

DYNET_SERIALIZE_COMMIT(Dim, DYNET_SERIALIZE_DEFINE(nd, d))
DYNET_SERIALIZE_IMPL(Dim)

} // namespace dynet

