#include "cnn/eigen-backend.h"

#include <cmath>

using namespace std;

namespace cnn {

Matrix Elewise::Ln(const Matrix& x) {
  return x.array().log();
}

Matrix Elewise::Exp(const Matrix& x) {
  return x.array().exp();
}

Matrix Elewise::SigmoidForward(const Matrix& x) {
  Matrix fx = x;
  for (unsigned i = 0; i < fx.rows(); ++i)
    for (unsigned j = 0; j < fx.cols(); ++j)
      fx(i,j) = 1.f / (1.f + expf(-x(i,j)));
  return fx;
}

Matrix Elewise::SigmoidBackward(const Matrix& diff, const Matrix& top, const Matrix& bottom) {
  const unsigned rows = top.rows();
  const unsigned cols = top.cols();
  Matrix dfdx(rows, cols);
  for (unsigned i = 0; i < rows; ++i)
    for (unsigned j = 0; j < cols; ++j)
      dfdx(i,j) = (1.f - top(i,j)) * top(i,j);
  return dfdx.cwiseProduct(diff);
}

Matrix Elewise::ReluForward(const Matrix& x) {
  Matrix fx = x;
  for (unsigned i = 0; i < fx.rows(); ++i)
    for (unsigned j = 0; j < fx.cols(); ++j)
      if (fx(i,j) < 0) fx(i,j) = 0;
  return fx;
}

Matrix Elewise::ReluBackward(const Matrix& diff, const Matrix& top, const Matrix& bottom) {
  Matrix dfdx = diff;
  const unsigned rows = diff.rows();
  const unsigned cols = diff.cols();
  for (unsigned i = 0; i < rows; ++i)
    for (unsigned j = 0; j < cols; ++j)
      if (!top(i,j)) dfdx(i,j) = 0;
  return dfdx;
}

Matrix Elewise::TanhForward(const Matrix& x) {
  Matrix fx = x;
  for (unsigned i = 0; i < fx.rows(); ++i)
    for (unsigned j = 0; j < fx.cols(); ++j)
      fx(i,j) = tanhf(fx(i,j));
  return fx;
}

Matrix Elewise::TanhBackward(const Matrix& diff, const Matrix& top, const Matrix& bottom) {
  const unsigned rows = top.rows();
  const unsigned cols = top.cols();
  Matrix dfdx(rows, cols);
  for (unsigned i = 0; i < rows; ++i)
    for (unsigned j = 0; j < cols; ++j)
      dfdx(i,j) = 1.f - top(i,j) * top(i,j);
  return dfdx.cwiseProduct(diff);
}

inline real logsumexp(const Matrix& x) {
  real m = x(0,0);
  for (unsigned i = 1; i < x.rows(); ++i) {
    real r = x(i,0);
    if (r > m) m = r;
  }
  real z = 0;
  for (unsigned i = 0; i < x.rows(); ++i)
    z += expf(x(i,0) - m);
  return m + logf(z);
}

Matrix Convolution::SoftmaxForward(const Matrix& src, SoftmaxAlgorithm algorithm) {
  const unsigned rows = src.rows();
  assert(src.cols() == 1);
  const real logz = logsumexp(src);
  Matrix fx(rows, 1);
  for (unsigned i = 0; i < rows; ++i)
    fx(i,0) = expf(src(i,0) - logz);
  return fx;
}

Matrix Convolution::SoftmaxBackward(const Matrix& diff, const Matrix& top, SoftmaxAlgorithm algorithm) {
  // d softmax(x)_i / d x_j = softmax(x)_i * (1 - softmax(x)_i) if i == j
  // d softmax(x)_i / d x_j = -softmax(x)_i * softmax(x)_j if i != j
  const unsigned rows = top.rows();

  real off_diag_sum = 0;
  for (unsigned i = 0; i < rows; ++i)
    off_diag_sum -= top(i, 0) * diff(i, 0);

  Matrix dEdx = Matrix::Zero(rows, 1);
  for (unsigned i = 0; i < rows; ++i)
    dEdx(i, 0) = (off_diag_sum + diff(i, 0)) * top(i, 0);
  return dEdx;
}

} // namespace cnn

