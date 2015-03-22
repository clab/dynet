#ifndef CNN_TENSOR_H_
#define CNN_TENSOR_H_

#include <iostream>
#include <initializer_list>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include "cnn/eigen-serialization.h"
#include <Eigen/Eigen>

namespace cnn {

typedef Eigen::MatrixXf Matrix;
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

inline Dim size(const Matrix& m) { return Dim(m.rows(), m.cols()); }

inline Matrix Zero(const Dim& d) { return Matrix::Zero(d.rows, d.cols); }
inline Matrix Random(const Dim& d) { return Matrix::Random(d.rows, d.cols) * (sqrt(6) / sqrt(d.cols + d.rows)); }
//inline Matrix Random(const Dim& d) { return Matrix::Random(d.rows, d.cols) * 0.08; }
inline Matrix Random(const Dim& d, double scale) { return Matrix::Random(d.rows, d.cols) * scale; }

// column major constructor
inline Matrix Ccm(const Dim&d, const std::initializer_list<real>& v) {
  Matrix m = Matrix::Zero(d.rows, d.cols);
  int cc = 0;
  int cr = 0;
  for (const auto& x : v) {
    m(cr, cc) = x;
    ++cc;
    if (cc == d.cols) { cc = 0; ++cr; }
  }
  return m;
}

} // namespace cnn

#endif
