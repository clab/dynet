#include "cnn/edges.h"

#include <limits>
#include <cmath>
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

inline real logsumexp(const Matrix& x, const vector<unsigned>& denom) {
  real m = x(denom[0],0);
  for (auto i : denom) {
    real r = x(i,0);
    if (r > m) m = r;
  }
  real z = 0;
  for (auto i : denom)
    z += exp(x(i,0) - m);
  return m + log(z);
}

string Identity::as_string(const vector<string>& arg_names) const {
  return arg_names[0];
}

Matrix Identity::forward(const vector<const Matrix*>& xs) const {
  assert(xs.size() == 1);
  return *xs.front();
}

Matrix Identity::backward(const vector<const Matrix*>& xs,
                  const Matrix& fx,
                  const Matrix& dEdf,
                  unsigned i) const {
  assert(i == 0);
  return dEdf;
}

string MaxPooling1D::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "maxpool1d(" << arg_names.front() << ",w=" << width << ")";
  return os.str();
}

Matrix MaxPooling1D::forward(const vector<const Matrix*>& xs) const {
  assert(xs.size() == 1);
  const Matrix& x = *xs.front();
  const unsigned x_rows = x.rows();
  assert(x.cols() == 1);
  const unsigned fx_rows = x_rows / width;
  ind.resize(fx_rows);
  Matrix fx = Matrix::Zero(fx_rows, 1);
  for (unsigned i = 0; i < fx_rows; ++i) {
    unsigned from = i * width;
    unsigned to = from + width;
    if (to > x_rows) to = x_rows;
    real best = x(from, 0);
    unsigned bestr = from;
    for (unsigned r = from + 1; r < to; ++r) {
      if (x(r, 0) > best) {
        best = x(r,0);
        bestr = r;
      }
    }
    ind[i] = bestr;
    fx(i, 0) = best;
  }
  return fx;
}

Matrix MaxPooling1D::backward(const vector<const Matrix*>& xs,
                  const Matrix& fx,
                  const Matrix& dEdf,
                  unsigned i) const {
  const Matrix& x = *xs.front();
  const unsigned x_rows = x.rows();
  Matrix dEdx = Matrix::Zero(x_rows, 1);
  const unsigned fx_rows = x_rows / width;
  assert(fx_rows == ind.size());
  assert(fx_rows == dEdf.rows());
  for (unsigned i = 0; i < fx_rows; ++i)
    dEdx(ind[i], 0) = dEdf(i, 0);
  return dEdx;
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

string RestrictedLogSoftmax::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "r_log_softmax(" << arg_names[0] << ')';
  return s.str();
}

Matrix RestrictedLogSoftmax::forward(const vector<const Matrix*>& xs) const {
  assert(xs.size() == 1);
  assert(denom.size() > 0);
  const Matrix& x = *xs.front();
  const unsigned rows = x.rows();
  assert(x.cols() == 1);
  const real logz = logsumexp(x, denom);
  Matrix fx(rows, 1);
  for (unsigned i = 0; i < rows; ++i)
    fx(i,0) = -numeric_limits<real>::infinity();
  for (auto i : denom)
    fx(i,0) = x(i,0) - logz;
  if (denom.size() == 1) fx(denom.front(), 0) = 0;
  return fx;
}

Matrix RestrictedLogSoftmax::backward(const vector<const Matrix*>& xs,
                            const Matrix& fx,
                            const Matrix& dEdf,
                            unsigned i) const {
  assert(i == 0);
  const Matrix& x = *xs.front();
  const unsigned rows = x.rows();
  Matrix dEdx = Matrix::Zero(rows, 1);
  double z = 0;
  for (auto i : denom)
    z += dEdf(i, 0);
  for (auto i : denom)
    dEdx(i, 0) = dEdf(i, 0) - exp(fx(i, 0)) * z;
  return dEdx;
}

// x_1 is a vector
// y = (x_1)_{*pval}
string PickElement::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "pick(" << arg_names[0] << ',' << *pval << ')';
  return s.str();
}

Matrix PickElement::forward(const vector<const Matrix*>& xs) const {
  assert(xs.size() == 1);
  const Matrix& x = *xs.front();
  assert(x.cols() == 1);
  assert(*pval < x.rows());
  Matrix fx(1,1);
  fx(0,0) = x(*pval, 0);
  return fx;
}

// derivative is 0 in all dimensions except 1 for the selected element
Matrix PickElement::backward(const vector<const Matrix*>& xs,
                    const Matrix& fx,
                    const Matrix& dEdf,
                    unsigned i) const {
  assert(i == 0);
  assert(dEdf.rows() == 1);
  assert(dEdf.cols() == 1);
  const Matrix& x = *xs.front();

  // TODO should be sparse
  Matrix dEdx = Matrix::Zero(x.rows(), 1); 
  dEdx(*pval,0) = dEdf(0,0);
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
  for (unsigned i = 1; i < xs.size(); i += 2) {
    if (xs[i]->cols() == 1 && xs[i+1]->cols() == 1)
      fx += xs[i]->cwiseProduct(*xs[i + 1]);
    else
      fx += (*xs[i]) * (*xs[i + 1]);
  }
  return fx;
}

Matrix Multilinear::backward(const vector<const Matrix*>& xs,
                             const Matrix& fx,
                             const Matrix& dEdf,
                             unsigned i) const {
  assert(i < xs.size());
  if (i == 0) return dEdf;
  if (i % 2 == 1) {  // is a matrix
    if (xs[i]->cols() == 1)  // diagonal matrix
      return dEdf.cwiseProduct(*xs[i+1]);
    else
      return dEdf * xs[i+1]->transpose();
  }
  // is a vector
  if (xs[i-1]->cols() == 1)  // xs[i-1] is a diagonal matrix
    return xs[i-1]->cwiseProduct(dEdf);
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
  return Elewise::ReluForward(*xs.front());
}

Matrix Rectify::backward(const vector<const Matrix*>& xs,
                        const Matrix& fx,
                        const Matrix& dEdf,
                        unsigned i) const {
  return Elewise::ReluBackward(dEdf, fx, *xs.front());
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

// you could do this with LogisticSigmoid, Softmax or a variety of other
// functions, but this is often useful.
// x_1 must be a scalar that is a value between 0 and 1
// target_y is a value between 0 and 1
// y = ty * log(x_1) + (1 - ty) * log(x_1)
string BinaryLogLoss::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "binary_log_loss(" << arg_names[0] << ", " << *ptarget_y << ')';
  return os.str();
}

Matrix BinaryLogLoss::forward(const vector<const Matrix*>& xs) const {
  assert(xs.size() == 1);
  assert(xs.front()->cols() == 1);
  assert(xs.front()->rows() == 1);
  const real y_pred = (*xs.front())(0,0);
  assert(y_pred >= 0.);
  assert(y_pred <= 1.);
  const real ty = *ptarget_y;
  assert(ty >= 0.);
  assert(ty <= 1.);
  Matrix fx = *xs.front();
  real& res = fx(0,0);
  res = 0;
  if (ty > 0.) res -= ty * log(y_pred);
  if ((1 - ty) > 0.) res -= (1 - ty) * log1p(-y_pred);
  return fx;
}

Matrix BinaryLogLoss::backward(const vector<const Matrix*>& xs,
                  const Matrix& fx,
                  const Matrix& dEdf,
                  unsigned i) const {
  const real y_pred = (*xs.front())(0,0);
  const real ty = *ptarget_y;
  real scale = 0;
  if (ty > 0.) scale -= ty / y_pred;
  if ((1 - ty) >= 0.) scale += (1 - ty) / (1 - y_pred);
  return dEdf * scale;
}

} // namespace cnn
