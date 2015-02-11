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

string LogSoftmax::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "log_softmax(" << arg_names[0] << ')';
  return s.str();
}

Matrix LogSoftmax::forward(const vector<const Matrix*>& xs) const {
  assert(xs.size() == 1);
  const Matrix& x = *xs.front();
  const unsigned rows = x.rows();
  assert(x.cols() == 1);
  const real logz = logsumexp(x);
  Matrix fx(rows, 1);
  for (unsigned i = 0; i < rows; ++i)
    fx(i,0) = x(i,0) - logz;
  return fx;
}

Matrix LogSoftmax::backward(const vector<const Matrix*>& xs,
                            const Matrix& fx,
                            const Matrix& dEdf,
                            unsigned i) const {
  assert(i == 0);
  const Matrix& x = *xs.front();
  const unsigned rows = x.rows();
  Matrix dEdx(rows, 1);
  double z = 0;
  for (unsigned i = 0; i < rows; ++i)
    z += dEdf(i, 0);
  for (unsigned i = 0; i < rows; ++i)
    dEdx(i, 0) = dEdf(i, 0) - exp(fx(i, 0)) * z;
  return dEdx;
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

string CwiseMultiply::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << " \\cdot " << arg_names[1];
  return s.str();
}

Matrix CwiseMultiply::forward(const vector<const Matrix*>& xs) const {
  assert(xs.size() == 2);
  return xs[0]->cwiseProduct(*xs[1]);
}

Matrix CwiseMultiply::backward(const vector<const Matrix*>& xs,
                               const Matrix& fx,
                               const Matrix& dEdf,
                               unsigned i) const {
  assert(i < 2);
  if (i == 0) {
    return dEdf.cwiseProduct(*xs[1]);
  } else {
    return dEdf.cwiseProduct(*xs[0]);
  }
}

string Multilinear::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0];
  for (unsigned i = 1; i < arg_names.size(); i += 2)
    s << " + " << arg_names[i] << " * " << arg_names[i+1];
  return s.str();
}

Matrix Multilinear::forward(const vector<const Matrix*>& xs) const {
  assert(xs.size() % 2 == 1);
  Matrix fx = *xs.front();
  //cerr << "Multilinear\n";
  //for (unsigned i = 0; i < xs.size(); i++)
  //  cerr << " (" << xs[i]->rows() << "," << xs[i]->cols() << ")\n";
  for (unsigned i = 1; i < xs.size(); i += 2)
    fx += (*xs[i]) * (*xs[i + 1]);
  return fx;
}

Matrix Multilinear::backward(const vector<const Matrix*>& xs,
                             const Matrix& fx,
                             const Matrix& dEdf,
                             unsigned i) const {
  assert(i < xs.size());
  if (i == 0) return dEdf;
  if (i % 2 == 1) return dEdf * xs[i+1]->transpose();
  return xs[i-1]->transpose() * dEdf;
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
