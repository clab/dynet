#include "cnn/nodes.h"

#include <limits>
#include <cmath>
#include <sstream>

#if HAVE_CUDA
#include "cnn/cuda.h"
#include "cnn/gpu-ops.h"
#endif

using namespace std;

// notes on implementing differentiable components
// 1) fx can be understood as a pointer to the (preallocated) location for the result
//    of forward to be stored
// 2) fx is not initialized, so after calling forward fx must point to the correct answer
// 3) fx can be repointed to an input, if forward(x) evaluates to x (e.g., in reshaping)
// 4) dEdxi MUST **ACCUMULATE** a result since multiple calls to forward may depend on
//    the same x_i. Even, e.g., Identity must be implemented as
//    dEdx1 += dEdf. THIS IS EXTREMELY IMPORTANT
// 5) scalars results of forward are placed in fx.v[0]
// 6) CNN manages its own memory, not Eigen, and it is configured with the
//    EIGEN_NO_MALLOC option. If you get an error about Eigen attempting to allocate
//    memory, it is (probably) because of an implicit creation of a temporary variable.
//    To tell Eigen this is not necessary, the noalias() method is available. If you really
//    do need a temporary variable, its capacity must be requested by Node::aux_storage_space

namespace cnn {

void Reshape::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  // just point to the input memory and change dimensions
  // dimensions are handled by forward_dim
  fx.v = xs[0]->v;
}

void Reshape::backward(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  const Tensor reshaped(from, dEdf.v);
  *dEdxi += *reshaped;
}

void SumColumns::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  auto x = **xs[0];
  *fx = x.rowwise().sum();
}

void SumColumns::backward(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  auto out = *dEdxi;
  const int c = out.cols();
  for (int j = 0; j < c; ++j)
    out.col(j) += *dEdf;
}

void KMHNGram::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
  auto x = **xs[0];
  const int new_cols = x.cols() - n + 1;
  assert(new_cols > 0);
  auto res = *fx;
  res.setZero();
  for (int j = 0; j < new_cols; ++j) {
    auto c_j = res.col(j);
    for (unsigned k = 0; k < n; ++k)
      c_j += x.col(j + k);
  }
}

void KMHNGram::backward(const vector<const Tensor*>& xs,
                        const Tensor& fx,
                        const Tensor& dEdf,
                        unsigned i,
                        Tensor& dEdxi) const {
  const int c = dEdf.d.cols();
  for (int j = 0; j < c; ++j)
    for (unsigned k = 0; k < n; ++k)
      (*dEdxi).col(j+k) += (*dEdf).col(j);
}

void InnerProduct3D_1D::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(!"not implemented");
}

void InnerProduct3D_1D::backward(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
  assert(!"not implemented");
}

void GaussianNoise::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  cerr << "FIX IMPL GaussianNoise::f\n"; abort();
#if 0
  assert(xs.size() == 1);
  const Tensor& x = *xs[0];
  return x + RandomNormal(Dim(x.rows(), x.cols()), 0, stddev);
#endif
}

void GaussianNoise::backward(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
  cerr << "FIX IMPL GaussianNoise::b\n"; abort();
#if 0
  assert(i == 0);
  return dEdf;
#endif
};

void Dropout::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  cerr << "FIX IMPL Dropout::f\n"; abort();
#if 0
  assert(xs.size() == 1);
  const Tensor& x = *xs[0];
  noise_mask = RandomBernoulli(Dim(x.rows(), x.cols()), p);
  return x.cwiseProduct(noise_mask);
#endif
}

void Dropout::backward(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
  cerr << "FIX IMPL Dropout::b\n"; abort();
#if 0
  assert(i == 0);
  return dEdf.cwiseProduct(noise_mask);
#endif
};

struct FConstantMinus {
  FConstantMinus(float c) : c(c) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float operator()(float x) const {
    return c - x;
  }
  float c;
};

void ConstantMinusX::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  auto x = **xs[0];
  *fx = x.unaryExpr(FConstantMinus(c));
}

void ConstantMinusX::backward(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
  *dEdxi -= *dEdf;
};

void Sum::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  const unsigned num_args = xs.size();
  if (num_args == 1) {
    fx.v = xs[0]->v;
    return;
  }
#if HAVE_CUDA
  for (unsigned i = 0; i < num_args; ++i)
    CUBLAS_CHECK(cublasSaxpy(cublas_handle, fx.d.size(), kSCALAR_ONE, xs[i]->v, 1, fx.v, 1));
#else
  auto res = *fx;
  const unsigned remainder = num_args % 4;
  switch (remainder) {
    case 0: res.setZero(); break;
    case 1: res = **xs[0]; break;
    case 2: res = **xs[0] + **xs[1]; break;
    case 3: res = **xs[0] + **xs[1] + **xs[2]; break;
  }
  for (unsigned i = remainder; i < num_args; i += 4)
    res += **xs[i] + **xs[i+1] + **xs[i+2] + **xs[i+3];
#endif
}

void Sum::backward(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
  *dEdxi += *dEdf;
};

struct FTanh {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float operator()(float x) const {
    return tanhf(x);
  }
};

void Tanh::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
#if HAVE_CUDA
  gpu::vtanh(fx.d.size(), xs[0]->v, fx.v);
#else
  auto x = **xs[0];
  *fx = x.unaryExpr(FTanh());
#endif
}

struct FTanhBackward {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float operator()(float t, float d) const {
    return (1.f - t * t) * d;
  }
};

void Tanh::backward(const vector<const Tensor*>& xs,
                      const Tensor& fx,
                      const Tensor& dEdf,
                      unsigned i,
                      Tensor& dEdxi) const {
  *dEdxi += (*fx).binaryExpr(*dEdf, FTanhBackward());
}

void Square::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  auto x = **xs[0];
  *fx = x.cwiseProduct(x);
}

void Square::backward(const vector<const Tensor*>& xs,
                        const Tensor& fx,
                        const Tensor& dEdf,
                        unsigned i,
                        Tensor& dEdxi) const {
  auto x = **xs[0];
  *dEdxi += (*dEdf).cwiseProduct(x) * 2;
};

void Exp::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  auto x = **xs[0];
  *fx = x.array().exp();
}

void Exp::backward(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
  *dEdxi += (*dEdf).cwiseProduct(*fx);
}

void Log::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  auto x = **xs[0];
  *fx = x.array().log();
}

void Log::backward(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
  auto x = **xs[0];
  *dEdxi += (*dEdf).cwiseQuotient(x);
}

void Concatenate::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  unsigned rows = 0;
  for (auto x : xs) rows += x->d.rows();
  // the following should use auxiliary memory
  src_row_indices.resize(xs.size());
  unsigned ind = 0;
  unsigned k = 0;
  for (auto x : xs) {
    src_row_indices[k++] = ind;
    auto xi = *x;
    assert(xi.d.cols() == 1); // this can be relaxed to the same everywhere
    const unsigned rows = xi.d.rows();
#if HAVE_CUDA
    CUDA_CHECK(cudaMemcpyAsync(&fx.v[ind], &xi.v[0], sizeof(float) * rows, cudaMemcpyDeviceToDevice));
#else
    (*fx).block(ind, 0, rows, 1) = *xi;
#endif
    ind += rows;
  }
}

void Concatenate::backward(const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  assert(i < src_row_indices.size());
  const unsigned rows = dEdxi.d.rows();
  const unsigned begin = src_row_indices[i];
  *dEdxi += (*dEdf).block(begin, 0, rows, 1);
}

void ConcatenateColumns::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  for (unsigned i = 0; i < xs.size(); ++i)
    (*fx).col(i) = **xs[i];
}

void ConcatenateColumns::backward(const vector<const Tensor*>& xs,
                                    const Tensor& fx,
                                    const Tensor& dEdf,
                                    unsigned i,
                                    Tensor& dEdxi) const {
  *dEdxi += (*dEdf).col(i);
}

void Hinge::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  cerr << "FIX IMPL3\n"; abort();
#if 0
  assert(xs.size() == 1);
  const Tensor& x = *xs.front();
  const unsigned rows = x.rows();
  if (u.rows() != rows)
    u = Tensor(rows, 1);  // local forward value
  real y = 0;
  const real mlystar = margin - x(*pelement, 0);
  for (unsigned i = 0; i < rows; ++i)
    if (*pelement != i)
      y += u(i, 0) = max(real(0), mlystar + x(i,0));
  Tensor res(1,1);
  res(0,0) = y;
  return res;
#endif
}

void Hinge::backward(const vector<const Tensor*>& xs,
                       const Tensor& fx,
                       const Tensor& dEdf,
                       unsigned i,
                       Tensor& dEdxi) const {
  cerr << "FIX IMPL4\n"; abort();
#if 0
  assert(i == 0);
  const Tensor& x = *xs.front();
  const unsigned rows = x.rows();
  Tensor dEdx = Zero(Dim(rows, 1));
  if (fx(0,0) == 0) return dEdx;
  const real diff = dEdf(0,0);
  unsigned tv = 0;
  for (unsigned i = 0; i < rows; ++i)
    if (*pelement != i && u(i, 0) > 0) { dEdx(i, 0) = diff; tv++; }
  dEdx(*pelement, 0) = -diff * tv;
  return dEdx;
#endif
}

void Identity::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.d = xs[0]->d;
  fx.v = xs[0]->v;
}

void Identity::backward(const vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const {
  *dEdxi += *dEdf;
}

void MaxPooling1D::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  cerr << "FIX IMPL5\n"; abort();
#if 0
  assert(xs.size() == 1);
  const Tensor& x = *xs.front();
  const unsigned x_rows = x.rows();
  assert(x.cols() == 1);
  const unsigned fx_rows = x_rows / width;
  ind.resize(fx_rows);
  Tensor fx = Zero(Dim(fx_rows, 1));
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
#endif
}

void MaxPooling1D::backward(const vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const {
  cerr << "FIX IMPL6\n"; abort();
#if 0
  const Tensor& x = *xs.front();
  const unsigned x_rows = x.rows();
  Tensor dEdx = Zero(Dim(x_rows, 1));
  const unsigned fx_rows = x_rows / width;
  assert(fx_rows == ind.size());
  assert(fx_rows == dEdf.rows());
  for (unsigned i = 0; i < fx_rows; ++i)
    dEdx(ind[i], 0) = dEdf(i, 0);
  return dEdx;
#endif
}

template <class T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float logsumexp(const T& x) {
  const float m = x.maxCoeff();
  float z = 0;
  for (unsigned i = 0; i < x.rows(); ++i)
    z += expf(x(i,0) - m);
  return m + logf(z);
}

struct FSoftmaxNormalize {
  explicit FSoftmaxNormalize(float logz) : logz(logz) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float operator()(float x) const {
    return expf(x - logz);
  }
  float logz;
};

void Softmax::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  if (xs[0]->d.cols() == 1) {
    auto x = **xs[0];
    *fx = x.unaryExpr(FSoftmaxNormalize(logsumexp(x)));
  } else {
    cerr << "SoftmaxForward not implemented for multiple columns\n";
    abort();
  }
}

struct FSoftmaxBackward {
  explicit FSoftmaxBackward(float off_diag_sum) : off_diag_sum(off_diag_sum) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float operator()(float t, float d) const {
    return (off_diag_sum + d) * t;
  }
  float off_diag_sum;
};

void Softmax::backward(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  float off_diag_sum = -(*fx).cwiseProduct(*dEdf).sum();
  *dEdxi += (*fx).binaryExpr(*dEdf, FSoftmaxBackward(off_diag_sum));
}

void PickNegLogSoftmax::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  if (xs[0]->d.cols() == 1) {
    auto x = **xs[0];
    logz = logsumexp(x);
    fx.v[0] = logz - x(*pval);
  } else {
    cerr << "SoftmaxForward not implemented for multiple columns\n";
    abort();
  }
}

struct FNegLogSoftmaxBackward {
  FNegLogSoftmaxBackward(float lz, float err) : logz(lz), d(err) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float operator()(float t) const {
    return expf(t - logz) * d;
  }
  float logz;
  float d;
};

void PickNegLogSoftmax::backward(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  if (xs[0]->d.cols() == 1) {
    const auto elem = *pval;
    const float err = dEdf.v[0];
    auto x = **xs[0];
    // logz is computed in the forward pass and cached
    *dEdxi += x.unaryExpr(FNegLogSoftmaxBackward(logz, err));
    (*dEdxi)(elem) -= err;
  } else {
    cerr << "PickNegLogSoftmax not implemented for multiple columns\n";
    abort();
  }
}

struct FLogSoftmaxNormalize {
  explicit FLogSoftmaxNormalize(float logz) : logz(logz) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float operator()(float x) const {
    return x - logz;
  }
  float logz;
};

void LogSoftmax::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
  if (xs[0]->d.cols() == 1) {
    auto x = **xs[0];
    *fx = x.unaryExpr(FLogSoftmaxNormalize(logsumexp(x)));
  } else {
    cerr << "LogSoftmaxForward not implemented for multiple columns\n";
    abort();
  }
}

struct FWeightedError {
  float operator()(float t, float d) const {
    return expf(t) * d / expf(t);
  }
};

struct FLogSoftmaxBackward {
  explicit FLogSoftmaxBackward(float off_diag_sum) : off_diag_sum(off_diag_sum) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float operator()(float t, float d) const {
    return off_diag_sum * expf(t) + d;
    //return (off_diag_sum + d) * t;
  }
  float off_diag_sum;
};

void LogSoftmax::backward(const vector<const Tensor*>& xs,
                          const Tensor& fx,
                          const Tensor& dEdf,
                          unsigned i,
                          Tensor& dEdxi) const {
  if (xs[0]->d.cols() == 1) {
    float off_diag_sum = -(*fx).binaryExpr(*dEdf, FWeightedError()).sum();
    *dEdxi += (*fx).binaryExpr(*dEdf, FLogSoftmaxBackward(off_diag_sum));
  } else {
    cerr << "LogSoftmaxBackward not implemented for multiple columns\n";
    abort();
  }
}

template <class T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE real logsumexp(const T& x, const vector<unsigned>& denom) {
  real m = x(denom[0],0);
  for (auto i : denom) {
    real r = x(i,0);
    if (r > m) m = r;
  }
  real z = 0;
  for (auto i : denom)
    z += expf(x(i,0) - m);
  return m + logf(z);
}

void RestrictedLogSoftmax::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  // TODO create auxiliary mask with -infty's
  // and do usual LogSoftmax stuff
  assert(xs.size() == 1);
  assert(denom.size() > 0);
  auto x = **xs[0];
  assert(x.cols() == 1);
  const real logz = logsumexp(x, denom);
  TensorTools::Constant(fx, -numeric_limits<real>::infinity());
  for (auto i : denom)
    (*fx)(i,0) = x(i,0) - logz;
  if (denom.size() == 1) (*fx)(denom.front(), 0) = 0;
}

void RestrictedLogSoftmax::backward(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  assert(i == 0);
  float z = 0;
  for (auto ind : denom)
    z += (*dEdf)(ind, 0);
  for (auto ind : denom)
    (*dEdxi)(ind, 0) += (*dEdf)(ind, 0) - expf((*fx)(ind, 0)) * z;
}

// x_1 is a vector
// y = (x_1)_{*pval}
void PickElement::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
  auto x = **xs[0];
  fx.v[0] = x(*pval);
}

// derivative is 0 in all dimensions except 1 for the selected element
void PickElement::backward(const vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i,
                    Tensor& dEdxi) const {
  assert(i == 0);
  (*dEdxi)(*pval) += dEdf.v[0];
}

// x_1 is a vector
// y = (x_1)[start:end]
// slice of vector from index start (inclusive) to index end (exclusive)
void PickRange::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
  auto x = **xs[0];
  assert(x.cols() == 1);
  assert(start >= 0);
  assert(end <= x.rows());
  assert(start < end);
  assert(int(fx.d.rows()) == int(end-start));
  (*fx) = x.block(start, 0, end-start, 1);
}

// derivative is 0 in all dimensions except the slice range
void PickRange::backward(const vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i,
                    Tensor& dEdxi) const {
  assert(i == 0);
  assert(int(dEdf.d.rows()) == int(end-start));
  assert(dEdf.d.cols() == 1);
  (*dEdxi).block(start, 0, end-start, 1) += (*dEdf);
}

void MatrixMultiply::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 2);
#if HAVE_CUDA
  auto x1 = *xs[0];
  auto x2 = *xs[1];
  CUBLAS_CHECK(cublasSgemv(cublas_handle, CUBLAS_OP_N, x1.d.rows(), x1.d.cols(),
             kSCALAR_ONE, x1.v, x1.d.rows(), x2.v, 1, kSCALAR_ZERO, fx.v, 1));
#else
  auto x1 = **xs[0];
  auto x2 = **xs[1];
  (*fx).noalias() = x1 * x2;
#endif
}

void MatrixMultiply::backward(const vector<const Tensor*>& xs,
                                const Tensor& fx,
                                const Tensor& dEdf,
                                unsigned i,
                                Tensor& dEdxi) const {
  assert(i < 2);
  if (i == 0) {
    (*dEdxi).noalias() += *dEdf * (**xs[1]).transpose();
  } else {
    (*dEdxi).noalias() += (**xs[0]).transpose() * *dEdf;
  }
}

void CwiseMultiply::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 2);
  auto x1 = **xs[0];
  auto x2 = **xs[1];
  *fx = x1.cwiseProduct(x2);
}

void CwiseMultiply::backward(const vector<const Tensor*>& xs,
                               const Tensor& fx,
                               const Tensor& dEdf,
                               unsigned i,
                               Tensor& dEdxi) const {
  assert(i < 2);
  if (i == 0) {
    auto x2 = **xs[1];
    *dEdxi += (*dEdf).cwiseProduct(x2);
  } else {
    auto x1 = **xs[0];
    *dEdxi += (*dEdf).cwiseProduct(x1);
  }
}

void AffineTransform::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() % 2 == 1);
  if (xs.size() == 1) {
    fx.v = xs[0]->v;
    return;
  } else {
    (*fx) = **xs[0];
    for (unsigned i = 1; i < xs.size(); i += 2)
      (*fx).noalias() += (**xs[i]) * (**xs[i + 1]);
  }
}

void AffineTransform::backward(const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  assert(i < xs.size());
  if (i == 0) { // bias term
    *dEdxi += *dEdf;
  } else if (i % 2 == 1) { // left argument of matrix multiply
    (*dEdxi).noalias() += *dEdf * (**xs[i+1]).transpose();
  } else {  // right argument of matrix multiply
    (*dEdxi).noalias() += (**xs[i-1]).transpose() * *dEdf;
  }
}

void Negate::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
  auto x = **xs[0];
  *fx = -x;
}

void Negate::backward(const vector<const Tensor*>& xs,
                        const Tensor& fx,
                        const Tensor& dEdf,
                        unsigned i,
                        Tensor& dEdxi) const {
  assert(i == 0);
  *dEdxi -= *dEdf;
}

struct FRectify {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float operator()(float x) const {
    return (x > 0.f) ? x : 0.f;
  }
};

void Rectify::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
  auto x = **xs[0];
  *fx = x.unaryExpr(FRectify());
}

struct FRectifyBackward {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float operator()(float t, float d) const {
    return (t) ? d : 0.f;
  }
};

void Rectify::backward(const vector<const Tensor*>& xs,
                         const Tensor& fx,
                         const Tensor& dEdf,
                         unsigned i,
                         Tensor& dEdxi) const {
  *dEdxi += (*fx).binaryExpr(*dEdf, FRectifyBackward());
}

void SquaredEuclideanDistance::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 2);
#if HAVE_CUDA
  gpu::sqeucdist(xs[0]->d.size(), xs[0]->v, xs[1]->v, fx.v);
#else
  auto x1 = **xs[0];
  auto x2 = **xs[1];
  fx.v[0] = (x1 - x2).squaredNorm();
#endif
}

void SquaredEuclideanDistance::backward(const vector<const Tensor*>& xs,
                                 const Tensor& fx,
                                 const Tensor& dEdf,
                                 unsigned i,
                                 Tensor& dEdxi) const {
  assert(i < 2);
  auto x1 = **xs[0];
  auto x2 = **xs[1];
  real scale = dEdf.v[0] * 2;
  if (i == 1) scale = -scale;
  *dEdxi += scale * (x1 - x2);
}

struct FLogisticSigmoid {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float operator()(float x) const {
    return 1.f / (1.f + expf(-x));
  }
};

void LogisticSigmoid::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
  auto x = **xs[0];
  *fx = x.unaryExpr(FLogisticSigmoid());
}

struct FLogisticSigmoidBackward {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float operator()(float t, float d) const {
    return (1.f - t) * t * d;
  }
};

void LogisticSigmoid::backward(const vector<const Tensor*>& xs,
                                 const Tensor& fx,
                                 const Tensor& dEdf,
                                 unsigned i,
                                 Tensor& dEdxi) const {
  *dEdxi += (*fx).binaryExpr(*dEdf, FLogisticSigmoidBackward());
}

// you could do this with LogisticSigmoid, Softmax or a variety of other
// functions, but this is often useful.
// x_1 must be a scalar that is a value between 0 and 1
// target_y is a value between 0 and 1
// y = ty * log(x_1) + (1 - ty) * log(x_1)
void BinaryLogLoss::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
  assert(xs.front()->d.size() == 1);
  const auto y_pred = xs[0]->v[0];
  assert(y_pred >= 0.);
  assert(y_pred <= 1.);
  const real ty = *ptarget_y;
  assert(ty >= 0.);
  assert(ty <= 1.);
  auto& res = fx.v[0];
  res = 0;
  if (ty > 0.) res -= ty * log(y_pred);
  if ((1 - ty) > 0.) res -= (1 - ty) * log1p(-y_pred);
}

void BinaryLogLoss::backward(const vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const {
  const auto y_pred = xs[0]->v[0];
  const real ty = *ptarget_y;
  real scale = 0;
  if (ty > 0.) scale -= ty / y_pred;
  if ((1 - ty) >= 0.) scale += (1 - ty) / (1 - y_pred);
  *dEdxi += *dEdf * scale;
}

} // namespace cnn
