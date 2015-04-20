#include "cnn/edges.h"

#include <limits>
#include <cmath>
#include <sstream>

using namespace std;

namespace cnn {

string InnerProduct3D_1D::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "inner(" << arg_names[0] << "," << arg_names[1] << ") + " << arg_names[2];
  return s.str();
}

Tensor InnerProduct3D_1D::forward(const vector<const Tensor*>& xs) const {
  assert(xs.size() == 3);
  Tensor fx;
  fx.inner(*xs[0], *xs[1]);
  fx += *xs[2];
  return fx;
}

Tensor InnerProduct3D_1D::backward(const vector<const Tensor*>& xs,
                                   const Tensor& fx,
                                   const Tensor& dEdf,
                                   unsigned i) const {
  assert(i < 3);
  if (i == 2) return dEdf;
  const int ii = dEdf.size(0);
  const int jj = dEdf.size(1);
  const int kk = xs[1]->size(0);
  Tensor dEdx;
  if (i == 0) {
//   (dE/dA)_ijk = (dE/dY)_ij * L_k
    dEdx.resize({ii, jj, kk});
    const real* x1 = xs[1]->data();
    for (int i = 0; i < ii; ++i) {
      for (int j = 0; j < jj; ++j) {
        const real d = dEdf.at({i,j});
        for (int k = 0; k < kk; ++k)
          dEdx.at({i,j,k}) = d * x1[k];
      }
    }
    return dEdx;
  }
//   (dE/dB)_k = (dE/dY)_ij * A_ijk
  dEdx.resize({kk});
  dEdx.zero();
  const Tensor& x0 = *xs[0];
  for (int i = 0; i < ii; ++i) {
    for (int j = 0; j < jj; ++j) {
      const real d = dEdf.at({i,j});
      for (int k = 0; k < kk; ++k)
        dEdx.at({k}) += d * x0.at({i,j,k});
    }
  }
  return dEdx;
}

string CwiseMultiply::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << " \\cdot " << arg_names[1];
  return s.str();
}

Tensor CwiseMultiply::forward(const vector<const Tensor*>& xs) const {
  assert(xs.size() == 2);
  Tensor fx;
  fx.cmul(*xs[0], *xs[1]);
  return fx;
}

Tensor CwiseMultiply::backward(const vector<const Tensor*>& xs,
                               const Tensor& fx,
                               const Tensor& dEdf,
                               unsigned i) const {
  assert(i < 2);
  Tensor dEdx;
  if (i == 0) {
    dEdx.cmul(dEdf, *xs[1]);
  } else {
    dEdx.cmul(dEdf, *xs[0]);
  }
  return dEdx;
}

string Negate::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << '-' << arg_names[0];
  return s.str();
}

Tensor Negate::forward(const vector<const Tensor*>& xs) const {
  assert(xs.size() == 1);
  return -(*xs[0]);
}

Tensor Negate::backward(const vector<const Tensor*>& xs,
                        const Tensor& fx,
                        const Tensor& dEdf,
                        unsigned i) const {
  assert(i == 0);
  return -dEdf;
}

string LogSoftmax::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "log_softmax(" << arg_names[0] << ')';
  return s.str();
}

Tensor LogSoftmax::forward(const vector<const Tensor*>& xs) const {
  assert(xs.size() == 1);
  const Tensor& x = *xs.front();
  Tensor fx;
  fx.softmax(x);
  fx.log();
  return fx;
}

Tensor LogSoftmax::backward(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i) const {
  assert(i == 0);

  Tensor u;
  u.exp(fx);
  Tensor dEdu;
  dEdu.cdiv(dEdf, u);
  Tensor dEdx;
  dEdx.softmax_backward(dEdu, u);
  return dEdx;
}

string Softmax::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "softmax(" << arg_names[0] << ')';
  return s.str();
}

Tensor Softmax::forward(const vector<const Tensor*>& xs) const {
  assert(xs.size() == 1);
  const Tensor& x = *xs.front();
  Tensor fx;
  fx.softmax(x);
  return fx;
}

Tensor Softmax::backward(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i) const {
  assert(i == 0);
  Tensor dEdx;
  dEdx.softmax_backward(dEdf, fx);
  return dEdx;
}

string OneMinusX::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "1 - " << arg_names[0];
  return s.str();
}

Tensor OneMinusX::forward(const vector<const Tensor*>& xs) const {
  assert(xs.size() == 1);
  const Tensor& x = *xs[0];
  Tensor fx;
  fx.one_minus(x);
  return fx;
}

Tensor OneMinusX::backward(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i) const {
  return -dEdf;
};

string Concatenate::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "concat(" << arg_names[0];
  for (unsigned i = 1; i < arg_names.size(); ++i) {
    os << ',' << arg_names[i];
  }
  os << ')';
  return os.str();
}

Tensor Concatenate::forward(const vector<const Tensor*>& xs) const {
  assert(xs.size() > 0);
  assert(xs.front()->ndims() == 1);
  Tensor prev = *xs[0];
  src_row_indices.resize(xs.size());
  src_row_indices[0] = 0;
  for (unsigned i = 1; i < xs.size(); ++i) {
    src_row_indices[i] = prev.size(0);
    Tensor t;
    t.cat(prev, *xs[i], 0);
    prev = t;
  }
  return prev;
}

Tensor Concatenate::backward(const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i) const {
  Tensor dEdx;
  unsigned rows = xs[i]->size(0);
  dEdx.narrow(dEdf,0,src_row_indices[i], rows);
  return dEdx;
}

string ConcatenateColumns::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "concat_cols(" << arg_names[0];
  for (unsigned i = 1; i < arg_names.size(); ++i) {
    os << ',' << arg_names[i];
  }
  os << ')';
  return os.str();
}

Tensor ConcatenateColumns::forward(const vector<const Tensor*>& xs) const {
  assert(xs.size() > 0);
  assert(xs.front()->ndims() == 1);

  Tensor prev = *xs[0];
  for (unsigned i = 1; i < xs.size(); ++i) {
    Tensor t;
    t.cat(prev, *xs[i], 1);
    prev = t;
  }
  return prev;
}

Tensor ConcatenateColumns::backward(const vector<const Tensor*>& xs,
                                    const Tensor& fx,
                                    const Tensor& dEdf,
                                    unsigned i) const {
  Tensor dEdx;
  dEdx.narrow(dEdf,1,i,1);
  return dEdx;
}

string SquaredEuclideanDistance::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "|| " << arg_names[0] << " - " << arg_names[1] << " ||^2";
  return s.str();
}

Tensor SquaredEuclideanDistance::forward(const vector<const Tensor*>& xs) const {
  assert(xs.size() == 2);
  Tensor res({1});
  Tensor diff = *xs[0] - *xs[1];
  diff.cmul(diff);
  res.at(0) = diff.sumall();
  return res;
}

Tensor SquaredEuclideanDistance::backward(const vector<const Tensor*>& xs,
                                 const Tensor& fx,
                                 const Tensor& dEdf,
                                 unsigned i) const {
  assert(i < 2);
  real scale = dEdf.at(0) * 2;
  if (i == 1) scale = -scale;
  Tensor dEdx = *xs[0] - *xs[1];
  dEdx *= scale;
  return dEdx;
}

string Sum::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0];
  for (unsigned i = 1; i < arg_names.size(); ++i)
    s << " + " << arg_names[i];
  return s.str();
}

Tensor Sum::forward(const vector<const Tensor*>& xs) const {
  assert(xs.size() > 0);
  Tensor fx = *xs[0];
  fx.force(Tensor::UNIQUE);
  for (unsigned i = 1; i < xs.size(); ++i)
    fx += *xs[i];
  return fx;
}

Tensor Sum::backward(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i) const {
  return dEdf;
};

string Multilinear::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0];
  for (unsigned i = 1; i < arg_names.size(); i += 2)
    s << " + " << arg_names[i] << " * " << arg_names[i+1];
  return s.str();
}

Tensor Multilinear::forward(const vector<const Tensor*>& xs) const {
  assert(xs.size() % 2 == 1);
  Tensor fx = *xs[0];
  fx.force(Tensor::UNIQUE);
  for (unsigned i = 1; i < xs.size(); i += 2)
    fx.addmv(1, 1, *xs[i], *xs[i+1]);
  return fx;
}

Tensor Multilinear::backward(const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i) const {
  assert(i < xs.size());
  if (i == 0) return dEdf;
  Tensor dEdx;
  dEdx.resizeAs(*xs[i]);
  dEdx.zero();
  // (TODO currently only vector supported, should probably support matrix with addmm)
  if (i % 2 == 1) {  // dif wrt matrix
    dEdx.addr(0, 1, dEdf, *xs[i+1]);
  } else {
    // dif wrt to right arg of multiplication
    Tensor xt = *xs[i-1];
    xt.transpose();
    dEdx.addmv(0, 1, xt, dEdf);
  }
  return dEdx;
}

string MatrixMultiply::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << " * " << arg_names[1];
  return s.str();
}

Tensor MatrixMultiply::forward(const vector<const Tensor*>& xs) const {
  assert(xs.size() == 2);
  Tensor fx;
  if (xs[1]->ndims() == 1) {
    fx.resize({xs[0]->size(0)});
    fx.zero();
    fx.addmv(0, 1, *xs[0], *xs[1]);
  } else {
    fx.resize({xs[0]->size(0), xs[1]->size(1)});
    fx.zero();
    fx.addmm(0, 1, *xs[0], *xs[1]);
  }
  return fx;
}

Tensor MatrixMultiply::backward(const vector<const Tensor*>& xs,
                                const Tensor& fx,
                                const Tensor& dEdf,
                                unsigned i) const {
  assert(i < 2);
  Tensor dEdx;
  dEdx.resizeAs(*xs[i]);
  dEdx.zero();
  if (i == 0) { // diff wrt a matrix
    Tensor xt = *xs[1];
    if (xt.ndims() == 1) {  // coeff is vector
      dEdx.addr(0, 1, dEdf, xt);
    } else { // coeff is matrix
      xt.transpose();
      dEdx.addmm(0, 1, dEdf, xt);
    }
  } else { // diff wrt to second argument
    Tensor xt = *xs[0];  // x[0] is a matrix, transpose it
    xt.transpose();
    if (dEdf.ndims() > 1) { // matrix
      dEdx.addmm(0, 1, xt, dEdf);
    } else { // vector
      dEdx.addmv(0, 1, xt, dEdf);
    }
  }
  return dEdx;
}

std::string Tanh::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "tanh(" << arg_names[0] << ')';
  return s.str();
}

Tensor Tanh::forward(const vector<const Tensor*>& xs) const {
  assert(xs.size() == 1);
  const Tensor& x = *xs.front();
  Tensor fx;
  fx.tanh(x);
  return fx;
}

Tensor Tanh::backward(const vector<const Tensor*>& xs,
                      const Tensor& fx,
                      const Tensor& dEdf,
                      unsigned i) const {
  assert(i == 0);
  Tensor dEdx;
  dEdx.tanh_backward(dEdf, fx);
  return dEdx;
#if 0
  assert(i == 0);
  Tensor o; o.resizeAs(fx);
  o.fill(1);
  Tensor y = fx;
  y.force(Tensor::UNIQUE);
  y.cmul(y);
  Tensor dEdx;
  dEdx.cmul(dEdf, o - y);
  return dEdx;
#endif
}

std::string LogisticSigmoid::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "\\sigma(" << arg_names[0] << ')';
  return s.str();
}

Tensor LogisticSigmoid::forward(const vector<const Tensor*>& xs) const {
  assert(xs.size() == 1);
  const Tensor& x = *xs.front();
  Tensor fx;
  fx.logistic_sigmoid(x);
  return fx;
}

Tensor LogisticSigmoid::backward(const vector<const Tensor*>& xs,
                      const Tensor& fx,
                      const Tensor& dEdf,
                      unsigned i) const {
  assert(i == 0);
  Tensor dEdx;
  dEdx.logistic_sigmoid_backward(dEdf, fx);
  return dEdx;
}

// x_1 is a vector
// y = (x_1)_{*pval}
string PickElement::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "pick(" << arg_names[0] << ',' << *pval << ')';
  return s.str();
}

Tensor PickElement::forward(const vector<const Tensor*>& xs) const {
  assert(xs.size() == 1);
  const Tensor& x = *xs.front();
  assert(x.ndims() == 1);
  Tensor fx;
  fx.narrow(x, 0, *pval, 1);
  return fx;
}

// derivative is 0 in all dimensions except 1 for the selected element
Tensor PickElement::backward(const vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i) const {
  assert(i == 0);
  assert(dEdf.isScalar());
  const Tensor& x = *xs.front();

  // TODO should be sparse
  Tensor dEdx = Zero(Dim({x.size(0)})); 
  dEdx.at({*pval}) = dEdf.front();
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

Tensor BinaryLogLoss::forward(const vector<const Tensor*>& xs) const {
  assert(xs.size() == 1);
  assert(xs.front()->isScalar());
  const real y_pred = (*xs.front()).front();
  assert(y_pred >= real(0));
  assert(y_pred <= real(1));
  const real ty = *ptarget_y;
  assert(ty >= real(0));
  assert(ty <= real(1));
  real res = 0;
  if (ty > 0.) res -= ty * log(y_pred);
  if ((1 - ty) > 0.) res -= (1 - ty) * log1p(-y_pred);
  Tensor fx;
  fx.resizeAs(*xs.front());
  fx.fill(res);
  return fx;
}

Tensor BinaryLogLoss::backward(const vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const {
  const real y_pred = xs.front()->front();
  const real ty = *ptarget_y;
  real scale = 0;
  if (ty > 0.) scale -= ty / y_pred;
  if ((1 - ty) >= 0.) scale += (1 - ty) / (1 - y_pred);
  return dEdf * scale;
}

} // namespace cnn
