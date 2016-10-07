#ifndef DYNET_DIM_H
#define DYNET_DIM_H

#include <cassert>
#include <initializer_list>
#include <type_traits>
#include <stdexcept>
#include <iosfwd>
#include <cstring>
#include <vector>

#define DYNET_MAX_TENSOR_DIM 7

namespace boost { namespace serialization { class access; } }

namespace dynet {

struct Dim {
  Dim() : nd(0), bd(1) {}
  // explicit Dim(unsigned int m) : nd(1), bd(1) { d[0] = m; }
  // TODO: The constructors for dimensions w/ and w/o batches is not intuitive.
  //       can this be fixed in some way?
  // Dim(unsigned int m, unsigned int n) : nd(2), bd(1) { d[0] = m; d[1] = n; }
  Dim(std::initializer_list<unsigned int> x) : nd(0), bd(1) {
    for(auto v : x) d[nd++] = v;
  }
  Dim(std::initializer_list<unsigned int> x, unsigned int b) : nd(0), bd(b) {
    for(auto v : x) d[nd++] = v;
  }
  Dim(const std::vector<long> & x) : nd(0), bd(1) {
     for(auto v : x) d[nd++] = v;
  }
  Dim(const std::vector<long> & x, unsigned int b) : nd(0), bd(b) {
     for(auto v : x) d[nd++] = v;
  }
  inline unsigned int size() const {
    return batch_size() * bd;
  }
  inline unsigned int batch_size() const {
    unsigned int p = 1;
    for (unsigned int i = 0; i < nd; ++i) p *= d[i];
    return p;
  }
  inline unsigned int sum_dims() const {
    unsigned int p = 0;
    for (unsigned int i = 0; i < nd; ++i) p += d[i];
    return p;
  }
  inline Dim truncate() const {
    Dim r = *this;
    unsigned int m = 1;
    unsigned int s = size();
    for (unsigned int i = 1; i < s; ++i)
      if (size(i) > 1) m = i + 1;
    r.resize(m);
    return r;
  }
  inline Dim single_batch() const {
    Dim r = *this;
    r.bd = 1;
    return r;
  }
  inline void resize(unsigned int i) { nd = i; }
  inline unsigned int ndims() const { return nd; }
  inline unsigned int rows() const { return d[0]; }
  inline unsigned int cols() const { return nd > 1 ? d[1] : 1; }
  inline unsigned int batch_elems() const { return bd; }
  inline void set(unsigned int i, unsigned int s) { assert(i < nd); assert(s > 0); d[i] = s; }
  inline unsigned int operator[](unsigned int i) const { return i < nd ? d[i] : 1; }
  inline unsigned int size(unsigned int i) const { return (*this)[i]; }
  inline Dim transpose() const {
    if (nd == 1) { return Dim({1, d[0]}, bd); }
    else if (nd == 2) { return Dim({d[1], d[0]}, bd); }
    throw std::invalid_argument("Cannot transpose Dim object with more than 2 dimensions");
  }
  unsigned int d[DYNET_MAX_TENSOR_DIM];
  unsigned int nd;
  unsigned int bd;
 private:
  friend class boost::serialization::access;
  template<class Archive> void serialize(Archive& ar, const unsigned int);
};

//static_assert(std::is_trivially_copyable<Dim>::value, "Dim must be trivially copyable");

inline bool operator==(const Dim& a, const Dim& b) {
  if (a.nd != b.nd || a.bd != b.bd) return false;
  return std::memcmp(a.d, b.d, a.nd) == 0;
}

inline bool operator!=(const Dim& a, const Dim& b) { return !(a == b); }

std::ostream& operator<<(std::ostream& os, const Dim& d);
std::ostream& operator<<(std::ostream& os, const std::vector<Dim>& ds);

} // namespace dynet

#endif
