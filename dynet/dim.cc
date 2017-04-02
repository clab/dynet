#include "dynet/dim.h"

#include <iostream>

using namespace std;

namespace dynet {

ostream& operator<<(ostream& os, const Dim& d) {
  os << d.nd << '-' << '{';
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
  is >> nd >> place_holder;
  is >> place_holder;
  d.resize(nd);
  for (unsigned i = 0; i < nd; ++i) {
    if (i) is >> place_holder;
    is >> d.d[i];
  }
  is >> place_holder;
  if (place_holder == 'X') {
    is >> d.bd >> place_holder;
  } else {
    is >> place_holder;
  }
  return is;
}

DYNET_SERIALIZE_COMMIT(Dim, DYNET_SERIALIZE_DEFINE(nd, d))
DYNET_SERIALIZE_IMPL(Dim)

} // namespace dynet

