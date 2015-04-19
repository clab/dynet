#ifndef _CNN_THPP_DIM_H_
#define _CNN_THPP_DIM_H_

#include <iosfwd>
#include <vector>
#include "thpp/Range.h"

struct Dim {
  Dim() : v() {}
  Dim(Dim&& o) = default;
  Dim(const Dim& o) = default;
  Dim(std::initializer_list<long> x) : v(x) {}
  Dim(const thpp::LongRange& r) : v(r.begin(), r.end()) {}
  long size() const {
    long p = 1;
    for (auto x : v) p *= x;
    return p;
  }
  long size(size_t i) const {
    return (*this)[i];
  }
  long& operator[](long i) { return v[i]; }
  long operator[](long i) const { return v[i]; }
  long ndims() const { return v.size(); }
  long rows() const { return v[0]; };
  long cols() const { return v[1]; };
  operator const std::vector<long>& () const {
    return v;
  }
  std::vector<long> v;
};

inline bool operator==(const Dim& a, const Dim& b) { return a.v == b.v; }
inline bool operator!=(const Dim& a, const Dim& b) { return a.v != b.v; }

std::ostream& operator<<(std::ostream& os, const Dim& d);

#endif
