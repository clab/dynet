#ifndef EIGEN_DIM_H
#define EIGEN_DIM_H

#include <initializer_list>
#include <iosfwd>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#define CNN_MAX_TENSOR_DIM 8

namespace cnn {

struct Dim {
  Dim() { d[0] = 0; }
  explicit Dim(int m) { d[0] = m; d[1] = 0; }
  Dim(int m, int n) { d[0] = m; d[1] = n; d[2] = 0; }
  Dim(std::initializer_list<long> x) {
    int c = 0;
    for(auto v : x) d[c++] = v;
    d[c] = 0;
  }
  inline int size() const { int p = 1; const unsigned short* pd=d; while(*pd) { p *= *pd; ++pd; } return p; }
  inline int ndims() const { int nd = 0; const unsigned short* pd = d; while(*pd) { nd++; pd++; } return nd; }
  inline unsigned Prod() const { return size(); }
  inline int rows() const { return d[0]; }
  inline int cols() const { return d[1] ? d[1] : 1; }
  inline int operator[](unsigned i) const { return d[i]; }
  inline int size(unsigned i) const { return d[i]; }
  inline Dim transpose() const { return Dim({d[1],d[0]}); }
  unsigned short d[CNN_MAX_TENSOR_DIM];
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

std::ostream& operator<<(std::ostream& os, const Dim& d);

} // namespace cnn

#endif
