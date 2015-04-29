#ifndef EIGEN_DIM_H
#define EIGEN_DIM_H

#include <initializer_list>
#include <iostream>

#include "cnn/backends/eigen/eigen-serialization.h"
#define CNN_MAX_TENSOR_DIM 8

namespace cnn {

struct Dim {
  Dim() { d[0] = 0; }
  explicit Dim(int m) { d[0] = m; d[1] = 0; }
  Dim(const Dim& o) { std::memcpy(d, o.d, sizeof(d)); }
  Dim(int m, int n) { d[0] = m; d[1] = n; d[2] = 0; }
  inline int size() const { int p = 1; const int* pd=d; while(*pd) { p *= *pd; ++pd; } return p; }
  int ndims() const { int nd = 0; const int* pd = d; while(*pd) { nd++; pd++; } return nd; }
  inline unsigned Prod() const { return size(); }
  int rows() const { return d[0]; }
  int cols() const { return d[1] ? d[1] : 1; }
  Dim(std::initializer_list<long> x) {
    int c = 0;
    for(auto v : x) d[c++] = v;
    d[c] = 0;
  }
  int operator[](unsigned i) const {
    return d[i];
  }
  int size(unsigned i) const {
    return d[i];
  }
  Dim transpose() const { return Dim({d[1],d[0]}); }
  int d[CNN_MAX_TENSOR_DIM];
 private:
  friend class boost::serialization::access;
  template<class Archive> void serialize(Archive& ar, const unsigned int) {
    ar & d;
    ar & d;
  }
};

inline bool operator==(const Dim& a, const Dim& b) {
  if (a.size() != b.size()) return false;
  for (unsigned i = 0; i < CNN_MAX_TENSOR_DIM && !(a.d[i] == 0 && b.d[i] == 0); ++i)
    if (a.d[i] != b.d[i]) return false;
  return true;
}

inline bool operator!=(const Dim& a, const Dim& b) { return !(a == b); }
inline std::ostream& operator<<(std::ostream& os, const Dim& d) {
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
#endif
