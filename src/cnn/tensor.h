#ifndef CNN_TENSOR_H_
#define CNN_TENSOR_H_

#include <iostream>
#include <Eigen/Eigen>

namespace cnn {

typedef Eigen::MatrixXd Matrix;
typedef double real;

struct Dim {
  Dim() : rows(1), cols(1) {}
  explicit Dim(unsigned m) : rows(m), cols(1) {}
  Dim(unsigned m, unsigned n) : rows(m), cols(n) {}
  unsigned short rows;
  unsigned short cols;
  Dim transpose() const { return Dim(cols,rows); }
};

inline Dim operator*(const Dim& a, const Dim& b) {
  assert(a.cols == b.rows);
  return Dim(a.rows, b.cols);
}

inline std::ostream& operator<<(std::ostream& os, const Dim& d) {
  return os << '(' << d.rows << ',' << d.cols << ')';
}

inline Matrix Zero(const Dim& d) { return Matrix::Zero(d.rows, d.cols); }
inline Matrix Random(const Dim& d) { return Matrix::Random(d.rows, d.cols) / sqrt(d.rows); }
inline Matrix Random(const Dim& d, double scale) { return Matrix::Random(d.rows, d.cols) * scale; }

} // namespace cnn

#endif
