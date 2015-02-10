#include "cnn/edges.h"

#include <sstream>

using namespace std;

namespace cnn {

// TODO move the implementation of all the standard functional edges into here

inline real logsumexp(const Matrix& x) {
  real m = x(0,0);
  for (unsigned i = 1; i < x.rows(); ++i) {
    real r = x(i,0);
    if (r > m) m = r;
  }
  real z = 0;
  for (unsigned i = 0; i < x.rows(); ++i)
    z += exp(x(i,0) - m);
  return m + log(z);
}

string MatrixMultiply::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << " * " << arg_names[1];
  return s.str();
}

Matrix MatrixMultiply::forward(const vector<const Matrix*>& xs) const {
  assert(xs.size() == 2);
  return (*xs[0]) * (*xs[1]);
}

Matrix MatrixMultiply::backward(const vector<const Matrix*>& xs,
                                const Matrix& fx,
                                const Matrix& dEdf,
                                unsigned i) const {
  assert(i < 2);
  if (i == 0) {
    return dEdf * xs[1]->transpose();
  } else {
    return xs[0]->transpose() * dEdf;
  }
}

string Negate::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << '-' << arg_names[0];
  return s.str();
}

Matrix Negate::forward(const vector<const Matrix*>& xs) const {
  assert(xs.size() == 1);
  return -(*xs[0]);
}

Matrix Negate::backward(const vector<const Matrix*>& xs,
                        const Matrix& fx,
                        const Matrix& dEdf,
                        unsigned i) const {
  assert(i == 0);
  return -dEdf;
}

string Rectify::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "ReLU(" << arg_names[0] << ')';
  return s.str();
}

Matrix Rectify::forward(const vector<const Matrix*>& xs) const {
  assert(xs.size() == 1);
  return xs[0]->cwiseMax(Matrix::Zero(xs[0]->rows(), xs[0]->cols()));
}

Matrix Rectify::backward(const vector<const Matrix*>& xs,
                        const Matrix& fx,
                        const Matrix& dEdf,
                        unsigned i) const {
  Matrix dEdx = fx;
  for (unsigned i = 0; i < dEdx.rows(); ++i)
    for (unsigned j = 0; j < dEdx.cols(); ++j)
      if (dEdx(i,j)) dEdx(i,j) = dEdf(i,j);
  return dEdx;
}

string HardTanh::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "hardtanh(" << arg_names[0] << ')';
  return s.str();
}

inline real hardtanh(real x) {
  if (x > 1.) return 1;
  if (x < -1.) return -1;
  return x;
}

Matrix HardTanh::forward(const vector<const Matrix*>& xs) const {
  assert(xs.size() == 1);
  Matrix fx = *xs[0];
  for (unsigned i = 0; i < fx.rows(); ++i)
    for (unsigned j = 0; j < fx.cols(); ++j)
      fx(i,j) = hardtanh(fx(i,j));
  return fx;
}

Matrix HardTanh::backward(const vector<const Matrix*>& xs,
                          const Matrix& fx,
                          const Matrix& dEdf,
                          unsigned i) const {
  const Matrix& x = *xs[0];
  Matrix dEdx = x * 0;
  for (unsigned i = 0; i < dEdx.rows(); ++i)
    for (unsigned j = 0; j < dEdx.cols(); ++j)
      if (x(i,j) > -1. && x(i,j) < -1.) dEdx(i,j) = dEdf(i,j);
  return dEdx;
}

} // namespace cnn
