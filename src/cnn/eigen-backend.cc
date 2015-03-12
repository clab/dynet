#include "cnn/eigen-backend.h"

#include <cmath>

using namespace std;

namespace cnn {

Matrix Elewise::SigmoidForward(const Matrix& x) {
  Matrix fx = x;
  for (unsigned i = 0; i < fx.rows(); ++i)
    for (unsigned j = 0; j < fx.cols(); ++j)
      fx(i,j) = 1. / (1. + exp(-x(i,j)));
  return fx;
}

Matrix Elewise::SigmoidBackward(const Matrix& diff, const Matrix& top, const Matrix& bottom) {
  const unsigned rows = top.rows();
  const unsigned cols = top.cols();
  Matrix dfdx(rows, cols);
  for (unsigned i = 0; i < rows; ++i)
    for (unsigned j = 0; j < cols; ++j)
      dfdx(i,j) = (1. - top(i,j)) * top(i,j);
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
      fx(i,j) = tanh(fx(i,j));
  return fx;
}

Matrix Elewise::TanhBackward(const Matrix& diff, const Matrix& top, const Matrix& bottom) {
  const unsigned rows = top.rows();
  const unsigned cols = top.cols();
  Matrix dfdx(rows, cols);
  for (unsigned i = 0; i < rows; ++i)
    for (unsigned j = 0; j < cols; ++j)
      dfdx(i,j) = 1. - top(i,j) * top(i,j);
  return dfdx.cwiseProduct(diff);
}

} // namespace cnn

