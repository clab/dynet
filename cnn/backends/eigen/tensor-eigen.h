#ifndef CNN_TENSOR_EIGEN_H_
#define CNN_TENSOR_EIGEN_H_

#include <initializer_list>

#include <Eigen/Eigen>
#include "cnn/backends/eigen/eigen-serialization.h"

namespace cnn {

#define EIGEN_BACKEND 1

typedef Eigen::MatrixXf Tensor;
typedef float real;

struct Dim {
  Dim() : rows(1), cols(1) {}
  explicit Dim(unsigned m) : rows(m), cols(1) {}
  Dim(unsigned m, unsigned n) : rows(m), cols(n) {}
  Dim(const std::initializer_list<unsigned>& x) {
    unsigned c = 0;
    for (auto v : x) {
      if (c == 0) rows = v;
      if (c == 1) cols = v;
      ++c;
    }
    if (c > 2) {
      std::cerr << "Dim class doesn't support more than two dimensions\n";
      abort();
    }
  }
  int operator[](unsigned i) const {
    if (i == 0) return rows;
    if (i == 1) return cols;
    abort();
  }
  unsigned short rows;
  unsigned short cols;
  Dim transpose() const { return Dim(cols,rows); }
 private:
  friend class boost::serialization::access;
  template<class Archive> void serialize(Archive& ar, const unsigned int) {
    ar & rows;
    ar & cols;
  }
};

inline Dim operator*(const Dim& a, const Dim& b) {
  assert(a.cols == b.rows);
  return Dim(a.rows, b.cols);
}

inline bool operator==(const Dim& a, const Dim& b) { return (a.rows == b.rows && a.cols == b.cols); }
inline bool operator!=(const Dim& a, const Dim& b) { return !(a == b); }

inline std::ostream& operator<<(std::ostream& os, const Dim& d) {
  return os << '(' << d.rows << ',' << d.cols << ')';
}

inline Dim size(const Tensor& m) { return Dim(m.rows(), m.cols()); }

inline Tensor Zero(const Dim& d) { return Eigen::MatrixXf::Zero(d.rows, d.cols); }
inline Tensor Random(const Dim& d) { return Eigen::MatrixXf::Random(d.rows, d.cols) * (sqrt(6) / sqrt(d.cols + d.rows)); }
inline Tensor Random(const Dim& d, double scale) { return Eigen::MatrixXf::Random(d.rows, d.cols) * scale; }

// column major constructor
inline Tensor Ccm(const Dim&d, const std::initializer_list<real>& v) {
  Tensor m = Zero(d);
  int cc = 0;
  int cr = 0;
  for (const auto& x : v) {
    m(cr, cc) = x;
    ++cr;
    if (cr == d.rows) { cr = 0; ++cc; }
  }
  return m;
}

} // namespace cnn

#include "cnn/backends/eigen/eigen-backend.h"

#endif
