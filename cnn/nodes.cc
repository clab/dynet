#include "cnn/nodes.h"

#include <limits>
#include <cmath>
#include <stdexcept>

#include "cnn/simd-functors.h"
#include "cnn/functors.h"
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
//
// notes on debugging problems with differentiable components
// 1) fx is uninitialized when forward is called- are you relying on it being 0?
// 2) dEdxi must accummulate (see point 4 above!)
//

namespace cnn {

#define MAX_SPARSEMAX_LOSS_ROWS 65536

size_t SparsemaxLoss::aux_storage_size() const {
  // first dim.size dimensions is the sparsemax
  const unsigned rows = MAX_SPARSEMAX_LOSS_ROWS;  // this should be xs[0]->d.rows()
  return rows * sizeof(float);
}

void SparsemaxLoss::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  if (xs[0]->d.cols() == 1) {
#if HAVE_CUDA
    throw std::runtime_error("SparsemaxLoss not implemented on GPU");
#else
    const int rows = xs[0]->d.rows();
    if (rows > MAX_SPARSEMAX_LOSS_ROWS) {
      cerr << "MAX_SPARSEMAX_LOSS_ROWS is not sufficient. Recompile with larger value.\n";
      abort();
    }
    float& y = fx.v[0];  // loss
    const unsigned qsupport_size = pq->size();
    const float qprop = 1.f / qsupport_size;

    float *zs = static_cast<float*>(aux_mem);
    std::partial_sort_copy(xs[0]->v, xs[0]->v+rows, zs, zs + rows, std::greater<float>());
    float sum = 0, maxsum = 0;
    unsigned k = 0;
    for (k = 0; k < rows; ++k) {
      sum += zs[k];
      float t = 1 + (k + 1) * zs[k];
      if (t <= sum) break;
      maxsum = sum;
    }
    float tau = (maxsum - 1) / k;
    Tensor tsm(xs[0]->d, (float*)aux_mem);
    auto x = **xs[0];
    auto sm = *tsm;
    sm = (x - Eigen::MatrixXf::Ones(rows, 1)*tau).cwiseMax(0.f);
    //cerr << "SpM: " << (sm).transpose() << " |||";
    //for (unsigned i = 0; i < pq->size(); ++i) cerr << ' ' << (*pq)[i];
    //cerr << endl;
    y = 0;
    float tau_sq = tau * tau;
    for (unsigned i = 0; i < rows; ++i) {
      if (sm(i, 0)) {
        const float x_s = x(i, 0);
        y += x_s * x_s - tau_sq;
      }
    }
    y /= 2;
    y += qprop * qprop * qsupport_size / 2;
    for (unsigned i = 0; i < qsupport_size; ++i)
      y -= qprop * x((*pq)[i], 0);
    if (y < 0) y = 0;
#endif
  } else {
    throw std::runtime_error("SparsemaxLoss not yet implemented for multiple columns");
  }
}

void SparsemaxLoss::backward_impl(const vector<const Tensor*>& xs,
                                  const Tensor& fx,
                                  const Tensor& dEdf,
                                  unsigned i,
                                  Tensor& dEdxi) const {
#if HAVE_CUDA
  throw std::runtime_error("SparsemaxLoss not implemented on GPU");
#else
  const float d = dEdf.v[0];
  float* psm = static_cast<float*>(aux_mem);
  float dqprop = d / pq->size();
  Tensor tsm(xs[0]->d, psm);
  auto sm = *tsm;  // sparsemax(z)
  *dEdxi += sm * d;
  for (unsigned i = 0; i < pq->size(); ++i)
    (*dEdxi)((*pq)[i], 0) -= dqprop;
#endif
}

size_t Sparsemax::aux_storage_size() const {
  return (1 + dim.size()) * max(sizeof(int), sizeof(float));
  // first is the size of the support
  // then comes the indices of the support
}

void Sparsemax::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  if (xs[0]->d.cols() == 1) {
#if HAVE_CUDA
    throw std::runtime_error("Sparsemax not implemented on GPU");
#else
    const unsigned rows = xs[0]->d.rows();
    float *zs = static_cast<float*>(aux_mem);
    std::partial_sort_copy(xs[0]->v, xs[0]->v+rows, zs, zs + rows, std::greater<float>());
    float sum = 0, maxsum = 0;
    unsigned k = 0;
    for (k = 0; k < rows; ++k) {
      sum += zs[k];
      float t = 1 + (k + 1) * zs[k];
      if (t <= sum) break;
      maxsum = sum;
    }
    float tau = (maxsum - 1) / k;
    auto x = **xs[0];
    auto y = *fx;
    y = (x - Eigen::MatrixXf::Ones(rows, 1)*tau).cwiseMax(0.f);
    int c = 1;
    int *cc = static_cast<int*>(aux_mem);
    for (unsigned i = 0; i < rows; ++i)
      if (y(i,0) > 0.f) cc[c++] = i;
    cc[0] = c - 1;
#endif
  } else {
    throw std::runtime_error("Sparsemax not yet implemented for multiple columns");
  }
}

void Sparsemax::backward_impl(const vector<const Tensor*>& xs,
                              const Tensor& fx,
                              const Tensor& dEdf,
                              unsigned i,
                              Tensor& dEdxi) const {
#if HAVE_CUDA
  throw std::runtime_error("Sparsemax not implemented on GPU");
#else
  const int ssize = static_cast<int*>(aux_mem)[0];
  int *support = static_cast<int*>(aux_mem) + 1;
  float dhat = 0;
  auto& d = *dEdf;
  for (int i = 0; i < ssize; ++i)
    dhat += d(support[i], 0);
  dhat /= ssize;
  for (int i = 0; i < ssize; ++i)
    (*dEdxi)(support[i], 0) += d(support[i], 0) - dhat;
#endif
}

void MatrixInverse::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
#ifdef HAVE_CUDA
  throw std::runtime_error("MatrixInverse not yet implemented for CUDA");
#else
  auto x = **xs[0];
  auto y = *fx;
  y = x.inverse();
#endif
}

void MatrixInverse::backward_impl(const vector<const Tensor*>& xs,
                                  const Tensor& fx,
                                  const Tensor& dEdf,
                                  unsigned i,
                                  Tensor& dEdxi) const {
  assert(xs.size() == 1);
#ifdef HAVE_CUDA
  throw std::runtime_error("MatrixInverse not yet implemented for CUDA");
#else
  auto d = *dEdf;
  auto y = *fx;
  (*dEdxi) -= y * d * y;
#endif
}

void AddMv::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 2);
#ifdef HAVE_CUDA
  throw std::runtime_error("AddMv not yet implemented for CUDA");
#else
  auto M = **xs[0];
  auto v = **xs[1];
  auto y = *fx;
  y = M.colwise() + v.col(0);
#endif
}

void AddMv::backward_impl(const vector<const Tensor*>& xs,
                          const Tensor& fx,
                          const Tensor& dEdf,
                          unsigned i,
                          Tensor& dEdxi) const {
  assert(xs.size() == 2);
#ifdef HAVE_CUDA
  throw std::runtime_error("AddMv not yet implemented for CUDA");
#else
  auto dEdx = *dEdxi;
  auto d = *dEdf;
  if (i == 0) { // Matrix
    dEdx += d;
  } else { // vector
    dEdx += d.rowwise().sum();
  }
#endif
}

void SelectRows::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
#ifdef HAVE_CUDA
  throw std::runtime_error("SelectRows not yet implemented for CUDA");
#else
  auto x = **xs[0];
  auto y = *fx;
  auto& rm = *prows;
  for (unsigned i = 0; i < rm.size(); ++i)
    y.row(i) = x.row(rm[i]);
#endif
}

void SelectRows::backward_impl(const vector<const Tensor*>& xs,
                               const Tensor& fx,
                               const Tensor& dEdf,
                               unsigned i,
                               Tensor& dEdxi) const {
  assert(xs.size() == 1);
#ifdef HAVE_CUDA
  throw std::runtime_error("SelectRows not yet implemented for CUDA");
#else
  auto dEdx = *dEdxi;
  auto d = *dEdf;
  auto& rm = *prows;
  for (unsigned i = 0; i < rm.size(); ++i)
    dEdx.row(rm[i]) += d.row(i);
#endif
}

void SelectCols::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
#ifdef HAVE_CUDA
  throw std::runtime_error("SelectCols not yet implemented for CUDA");
#else
  auto x = **xs[0];
  auto y = *fx;
  auto& cm = *pcols;
  for (unsigned i = 0; i < cm.size(); ++i)
    y.col(i) = x.col(cm[i]);
#endif
}

void SelectCols::backward_impl(const vector<const Tensor*>& xs,
                               const Tensor& fx,
                               const Tensor& dEdf,
                               unsigned i,
                               Tensor& dEdxi) const {
  assert(xs.size() == 1);
#ifdef HAVE_CUDA
  throw std::runtime_error("SelectCols not yet implemented for CUDA");
#else
  auto dEdx = *dEdxi;
  auto d = *dEdf;
  auto& cm = *pcols;
  for (unsigned i = 0; i < cm.size(); ++i)
    dEdx.col(cm[i]) += d.col(i);
#endif
}

void Pow::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 2);
#ifdef HAVE_CUDA
  throw std::runtime_error("Pow not yet implemented for CUDA");
#else
  auto x1 = **xs[0];
  auto x2 = xs[1]->v[0];
  (*fx).array() = x1.array().pow(x2);
#endif
}

void Pow::backward_impl(const vector<const Tensor*>& xs,
                        const Tensor& fx,
                        const Tensor& dEdf,
                        unsigned i,
                        Tensor& dEdxi) const {
  assert(xs.size() == 2);
#ifdef HAVE_CUDA
  throw std::runtime_error("Pow not yet implemented for CUDA");
#else
  auto x1 = **xs[0];
  auto x2 = xs[1]->v[0];
  if (i == 0) {
    *dEdxi += (x2 * x1.array().pow(x2 - 1).matrix()).cwiseProduct(*dEdf);
  } else {
    // y = a^x
    // dy/dx = a^x * log(a)
    (*dEdxi).noalias() += (*fx).cwiseProduct(x1.array().log().matrix()).transpose() * (*dEdf);
  }
#endif
}

size_t Min::aux_storage_size() const {
  return dim.size() * sizeof(float);
}

void Min::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Min not yet implemented for CUDA");
#else
  auto y = fx.vec();
  auto x1 = xs[0]->vec();
  auto x2 = xs[1]->vec();
  Tensor t(fx.d, static_cast<float*>(aux_mem));
  auto u = t.vec();
  u = (x1.array() < x2.array()).matrix().cast<float>();
  y = x1.cwiseMin(x2);
#endif
}

void Min::backward_impl(const vector<const Tensor*>& xs,
                   const Tensor& fx,
                   const Tensor& dEdf,
                   unsigned i,
                   Tensor& dEdxi) const {
  assert(i < 2);
#ifdef HAVE_CUDA
  throw std::runtime_error("Min not yet implemented for CUDA");
#else
  const Tensor t(dEdxi.d, static_cast<float*>(aux_mem));
  if (i == 0) {
    dEdxi.vec() += t.vec().cwiseProduct(dEdf.vec());
  } else {
    dEdxi.vec() += t.vec().binaryExpr(dEdf.vec(), FMaxBackwardInv());
  }
#endif
}

size_t Max::aux_storage_size() const {
  return dim.size() * sizeof(float);
}

void Max::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Max not yet implemented for CUDA");
#else
  auto y = fx.vec();
  auto x1 = xs[0]->vec();
  auto x2 = xs[1]->vec();
  Tensor t(fx.d, static_cast<float*>(aux_mem));
  auto u = t.vec();
  u = (x1.array() > x2.array()).matrix().cast<float>();
  y = x1.cwiseMax(x2);
#endif
}

void Max::backward_impl(const vector<const Tensor*>& xs,
                   const Tensor& fx,
                   const Tensor& dEdf,
                   unsigned i,
                   Tensor& dEdxi) const {
  assert(i < 2);
#ifdef HAVE_CUDA
  throw std::runtime_error("Max not yet implemented for CUDA");
#else
  const Tensor t(dEdxi.d, static_cast<float*>(aux_mem));
  if (i == 0) {
    dEdxi.vec() += t.vec().cwiseProduct(dEdf.vec());
  } else {
    dEdxi.vec() += t.vec().binaryExpr(dEdf.vec(), FMaxBackwardInv());
  }
#endif
}

void TraceOfProduct::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("TraceOfProduct not yet implemented for CUDA");
#else
  auto x1 = **xs[0];
  auto x2 = **xs[1];
  fx.v[0] = (x1 * x2.transpose()).trace();
#endif
}

void TraceOfProduct::backward_impl(const vector<const Tensor*>& xs,
                              const Tensor& fx,
                              const Tensor& dEdf,
                              unsigned i,
                              Tensor& dEdxi) const {
  assert(i < 2);
#ifdef HAVE_CUDA
  throw std::runtime_error("TraceOfProduct not yet implemented for CUDA");
#else
  const float d = dEdf.v[0];
  auto xother = **xs[1 - i];
  *dEdxi += d * xother;
#endif
}

void ConstScalarMultiply::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("ConstantScalarMultiply not yet implemented for CUDA");
#else
  fx.vec() = xs[0]->vec() * alpha;
#endif
}

void ConstScalarMultiply::backward_impl(const vector<const Tensor*>& xs,
                                   const Tensor& fx,
                                   const Tensor& dEdf,
                                   unsigned i,
                                   Tensor& dEdxi) const {
  assert(i == 0);
#ifdef HAVE_CUDA
  throw std::runtime_error("ScalarMultiply not yet implemented for CUDA");
#else
  dEdxi.vec() += dEdf.vec() * alpha;
#endif
}

void DotProduct::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("DotProduct not yet implemented for CUDA");
#else
  *fx = (**xs[0]).transpose() * (**xs[1]);
#endif
}

void DotProduct::backward_impl(const vector<const Tensor*>& xs,
                          const Tensor& fx,
                          const Tensor& dEdf,
                          unsigned i,
                          Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("DotProduct not yet implemented for CUDA");
#else
  (*dEdxi) += (dEdf.v[0]) * (**xs[1 - i]);
#endif
}

void Transpose::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  if (dim.rows() == 1 || dim.cols() == 1) {
    fx.v = xs[0]->v;
  } else {
#if HAVE_CUDA
    for(unsigned b = 0; b < xs[0]->d.bd; ++b)
      CUBLAS_CHECK(cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, fx.d.rows(), fx.d.cols(),
                               kSCALAR_ONE, xs[0]->batch_ptr(b), xs[0]->d.rows(), kSCALAR_ZERO, NULL, fx.d.rows(), fx.batch_ptr(b), fx.d.rows()));
#else
    for(unsigned b = 0; b < xs[0]->d.bd; ++b)
      fx.batch_matrix(b).noalias() = xs[0]->batch_matrix(b).transpose();
#endif
  }
}

void Transpose::backward_impl(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
#if HAVE_CUDA
  for(unsigned b = 0; b < xs[0]->d.bd; ++b)
    CUBLAS_CHECK(cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, dEdxi.d.rows(), dEdxi.d.cols(),
                             kSCALAR_ONE, dEdf.batch_ptr(b), dEdf.d.rows(), kSCALAR_ONE, dEdxi.batch_ptr(b), dEdxi.d.rows(), dEdxi.batch_ptr(b), dEdxi.d.rows()));
#else
  for(unsigned b = 0; b < xs[0]->d.bd; ++b)
    dEdxi.batch_matrix(b) += dEdf.batch_matrix(b).transpose();
#endif
}

void Reshape::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  // just point to the input memory and change dimensions
  // dimensions are handled by forward_dim
#ifdef HAVE_CUDA
  throw std::runtime_error("Reshape not yet implemented for CUDA");
#else
  fx.v = xs[0]->v;
#endif
}

void Reshape::backward_impl(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Reshape not yet implemented for CUDA");
#else
  const Tensor reshaped(dEdxi.d, dEdf.v);
  *dEdxi += *reshaped;
#endif
}

void SumColumns::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  auto x = **xs[0];
  auto y = *fx;
  if (xs.size() == 1) {
    y = x.rowwise().sum();
  } else {
    throw std::invalid_argument("two inputs in SumColumns::forward!");
  }
}

void SumColumns::backward_impl(const vector<const Tensor*>& xs,
                          const Tensor& fx,
                          const Tensor& dEdf,
                          unsigned i,
                          Tensor& dEdxi) const {
  auto out = *dEdxi;
  // this uses Eigen's broadcast capability
  // the following doesn't compile, so i use the next line
  //out.colwise() += *dEdf;
  out.colwise() += (*dEdf).col(0);
}

void KMHNGram::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
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

void KMHNGram::backward_impl(const vector<const Tensor*>& xs,
                        const Tensor& fx,
                        const Tensor& dEdf,
                        unsigned i,
                        Tensor& dEdxi) const {
  const int c = dEdf.d.cols();
  for (int j = 0; j < c; ++j)
    for (unsigned k = 0; k < n; ++k)
      (*dEdxi).col(j+k) += (*dEdf).col(j);
}

//   Y_ij = A_ijk * B_k (+ C_ij)
void InnerProduct3D_1D::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  auto A = xs[0]->t<3>();
  auto b = xs[1]->t<1>();
  typedef Eigen::Tensor<float, 1>::DimensionPair DimPair;
  Eigen::array<DimPair, 1> dims({{DimPair(2, 0)}});
  if (xs.size() == 2) {
    fx.t<2>() = A.contract(b, dims);
  } else {
    auto C = xs[2]->t<2>();
    fx.t<2>() = A.contract(b, dims) + C;
  }
}

void InnerProduct3D_1D::backward_impl(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
  auto tdEdf = dEdf.t<2>();  // 2 tensor
  typedef Eigen::Tensor<float, 1>::DimensionPair DimPair;
  if (i == 0) { // 3 tensor
    // tensor product
    auto b = xs[1]->t<1>();
    dEdxi.t<3>() += tdEdf.contract(b, Eigen::array<DimPair, 0>{{}});
  } else if (i == 1) {
    auto A = xs[0]->t<3>();  // A is 3 tensor
    Eigen::array<DimPair, 2> dims({{DimPair(0, 0), DimPair(1, 1)}});
    dEdxi.t<1>() += tdEdf.contract(A, dims);
  } else if (i == 2) {
    dEdxi.t<2>() += tdEdf;
  } else {
    cerr << "shouldn't happen\n"; abort();
  }
}

//   Y_ij = A_ijk * B_k * C_j (+ D_i)
void InnerProduct3D_1D_1D::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  auto A = xs[0]->t<3>();
  auto b = xs[1]->t<1>();
  auto c = xs[2]->t<1>();
  typedef Eigen::Tensor<float, 1>::DimensionPair DimPair;
  Eigen::array<DimPair, 1> dims({{DimPair(2, 0)}});
  Eigen::array<DimPair, 1> dims2({{DimPair(1, 0)}});
  if (xs.size() == 3) {
    fx.t<1>() = A.contract(b, dims).contract(c, dims2);
  } else {
    auto d = xs[3]->t<1>();
    fx.t<1>() = A.contract(b, dims).contract(c, dims2) + d;
  }
}

void InnerProduct3D_1D_1D::backward_impl(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
  auto tdEdf = dEdf.t<1>();  // vector
  typedef Eigen::Tensor<float, 1>::DimensionPair DimPair;
  if (i == 0) { // 3 tensor
    // tensor product
    auto b = xs[1]->t<1>();
    auto c = xs[2]->t<1>();
    dEdxi.t<3>() += tdEdf.contract(c, Eigen::array<DimPair, 0>{{}}).contract(b, Eigen::array<DimPair, 0>{{}});
  } else if (i == 1) { // vector 1
    // TODO these should be reorganized so the contraction is first with tdEdf and then with c or b.
    // in theory, that intermediate result could be cached (although CNN doesn't support this). the fact that it
    // this part of the product is redone when i=1 and again when i=2 is probably why this is slower
    // (or maybe it's the contract implementation?)
    Eigen::array<DimPair, 1> dims({{DimPair(1, 0)}});
    Eigen::array<DimPair, 1> dims2({{DimPair(0, 0)}});
    auto A = xs[0]->t<3>();
    auto c = xs[2]->t<1>();
    dEdxi.t<1>() += A.contract(c, dims).contract(tdEdf, dims2);
  } else if (i == 2) { // vector 2
    Eigen::array<DimPair, 1> dims({{DimPair(2, 0)}});
    Eigen::array<DimPair, 1> dims2({{DimPair(0, 0)}});
    auto A = xs[0]->t<3>();
    auto b = xs[1]->t<1>();
    dEdxi.t<1>() += A.contract(b, dims).contract(tdEdf, dims2);
  } else if (i == 3) { // vector bias
    dEdxi.t<1>() += tdEdf;
  } else {
    cerr << "shouldn't happen\n"; abort();
  }
}

size_t GaussianNoise::aux_storage_size() const {
  return dim.size() * sizeof(float);
}

void GaussianNoise::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("GaussianNoise not yet implemented for CUDA");
#else
  Tensor m(dim, (float*)aux_mem);
  TensorTools::RandomizeNormal(0, stddev, m);
  fx.vec() = xs[0]->vec() + m.vec();
#endif
}

void GaussianNoise::backward_impl(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("GaussianNoise not yet implemented for CUDA");
#else
  dEdxi.vec() += dEdf.vec();
#endif
}

size_t Dropout::aux_storage_size() const {
  return dim.size() * sizeof(float);
}

void Dropout::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  Tensor m(dim, (float*)aux_mem);
  TensorTools::RandomBernoulli(m, (1.f-p), 1.f / (1.f-p));
#ifdef HAVE_CUDA
  gpu::vcwise_product(fx.d.size(), xs[0]->v, m.v, fx.v);
#else
  fx.vec() = xs[0]->vec().cwiseProduct(m.vec());
#endif
}

void Dropout::backward_impl(const vector<const Tensor*>& xs,
                       const Tensor& fx,
                       const Tensor& dEdf,
                       unsigned i,
                       Tensor& dEdxi) const {
  Tensor m(dim, (float*)aux_mem);
#ifdef HAVE_CUDA
  gpu::vcwise_product(dEdf.d.size(), dEdf.v, m.v, fx.v);
#else
  dEdxi.vec() += dEdf.vec().cwiseProduct(m.vec());
#endif
}

size_t BlockDropout::aux_storage_size() const {
  // we just need to remember whether this entire block is turned on (1.0) or off (0.0)
  return 1 * sizeof(float);
}

void BlockDropout::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("BlockDropout not yet implemented for CUDA");
#else
  bernoulli_distribution distribution(1.0 - dropout_probability);
  float block_multiplier = distribution(*rndeng)? 1.0 : 0.0;
  block_multiplier = 
    dropout_probability == 1.0? 0.0 : block_multiplier / (1.0 - dropout_probability);
  if (dropout_probability > 1.0 || dropout_probability < 0.0) {
    assert(false && "dropout probability must be in the range [0, 1]");
  }
  *(static_cast<float*>(aux_mem)) = block_multiplier;
  (*fx) = **xs[0] * block_multiplier;
#endif
}

void BlockDropout::backward_impl(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("BlockDropout not yet implemented for CUDA");
#else
  float block_multiplier = *(static_cast<float*>(aux_mem));
  (*dEdxi) += (*dEdf) * block_multiplier;
#endif
}

void ConstantPlusX::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("ConstantPlusX not yet implemented for CUDA");
#else
  fx.vec() = xs[0]->vec().unaryExpr(const_add_op<float>(c));
#endif
}

void ConstantPlusX::backward_impl(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
  dEdxi.vec() += dEdf.vec();
}

void ConstantMinusX::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#if HAVE_CUDA
  gpu::vconstant_minusx(fx.d.size(), c, xs[0]->v, fx.v);
#else
  fx.vec() = xs[0]->vec().unaryExpr(const_minus_op<float>(c));
#endif
}

void ConstantMinusX::backward_impl(const vector<const Tensor*>& xs,
                              const Tensor& fx,
                              const Tensor& dEdf,
                              unsigned i,
                              Tensor& dEdxi) const {
#if HAVE_CUDA
  gpu::vnegate_backward(dEdxi.d.size(), dEdf.v, dEdxi.v);
#else
  dEdxi.vec() -= dEdf.vec();
#endif
}

template <class T>
EIGEN_STRONG_INLINE float logsumexp(const T& x) {
  using std::exp;
  using std::log;
  const float m = x.maxCoeff();
#if 1
  // these are equivalent, but this can use vectorized arithmetic
  float z = x.unaryExpr(const_add_op<float>(-m)).array().exp().matrix().sum();
#else
  float z = 0;
  for (unsigned i = 0; i < x.rows(); ++i)
    z += exp(x(i,0) - m);
#endif
  return m + log(z);
}

// set use_cholesky if M is symmetric - it's faster and more stable
// for dep paring it won't be
template <typename MatrixType>
inline typename MatrixType::Scalar logdet(const MatrixType& M, bool use_cholesky = false) {
  using namespace Eigen;
  using std::log;
  typedef typename MatrixType::Scalar Scalar;
  Scalar ld = 0;
  if (use_cholesky) {
    LLT<Matrix<Scalar,Dynamic,Dynamic>> chol(M);
    auto& U = chol.matrixL();
    for (unsigned i = 0; i < M.rows(); ++i)
      ld += log(U(i,i));
    ld *= 2;
  } else {
    PartialPivLU<Matrix<Scalar,Dynamic,Dynamic>> lu(M);
    auto& LU = lu.matrixLU();
    Scalar c = lu.permutationP().determinant(); // -1 or 1
    for (unsigned i = 0; i < LU.rows(); ++i) {
      const auto& lii = LU(i,i);
      if (lii < Scalar(0)) c *= -1;
      ld += log(abs(lii));
    }
    ld += log(c);
  }
  return ld;
}

// this i need to do something better, but this is a work-around
// if this is too small, just make it bigger
#define MAX_LOG_SUM_EXP 65536
size_t LogSumExp::aux_storage_size() const {
  return MAX_LOG_SUM_EXP * sizeof(float);
}

void LogSumExp::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  const unsigned num_args = xs.size();
  if (num_args == 1) {
    fx.v = xs[0]->v;
    return;
  }
  for (unsigned i = 0; i < xs.size(); ++i)
    static_cast<float*>(aux_mem)[i] = (**xs[i])(0,0);
  Dim r = {(unsigned int)xs.size()};
  Tensor v(r, static_cast<float*>(aux_mem));
  fx.v[0] = logsumexp(*v);
}

void LogDet::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.v[0] = logdet(**xs[0], false);
}

void LogSumExp::backward_impl(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
  if (xs.size() == 0) {
    *dEdxi += *dEdf;
    return;
  }
  // df/dx_i = 1/{sum_j exp(x_j)} * exp(x_i)}
  //         = 1/{exp f(x)} * exp(x_i)
  //         = exp(x_i - f(x))
  auto d = *dEdxi;
  d.array() += (**xs[i] - *fx).array().exp() * (*dEdf).array();
}

void LogDet::backward_impl(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
  auto trans = (**xs[0]).transpose();
  (*dEdxi) += (dEdf.v[0]) * trans.inverse();
}

void Sum::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  const unsigned num_args = xs.size();
  if (num_args == 1) {
    fx.v = xs[0]->v;
    return;
  }
#if HAVE_CUDA
  TensorTools::Zero(fx);
  for (unsigned i = 0; i < num_args; ++i)
    CUBLAS_CHECK(cublasSaxpy(cublas_handle, fx.d.size(), kSCALAR_ONE, xs[i]->v, 1, fx.v, 1));
#else
  auto res = fx.vec();
  const unsigned remainder = num_args % 4;
  switch (remainder) {
    case 0: res.setZero(); break;
    case 1: res = xs[0]->vec(); break;
    case 2: res = xs[0]->vec() + xs[1]->vec(); break;
    case 3: res = xs[0]->vec() + xs[1]->vec() + xs[2]->vec(); break;
  }
  for (unsigned i = remainder; i < num_args; i += 4)
    res += xs[i]->vec() + xs[i+1]->vec() + xs[i+2]->vec() + xs[i+3]->vec();
#endif
}

void Sum::backward_impl(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {

#if HAVE_CUDA
  CUBLAS_CHECK(cublasSaxpy(cublas_handle, fx.d.size(), kSCALAR_ONE, dEdf.v, 1, dEdxi.v, 1));
#else
  dEdxi.vec() += dEdf.vec();
#endif
}

void SumBatches::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
  unsigned num_args = xs[0]->d.bd;
#if HAVE_CUDA
  TensorTools::Zero(fx);
  for (unsigned i = 0; i < num_args; ++i)
    CUBLAS_CHECK(cublasSaxpy(cublas_handle, fx.d.size(), kSCALAR_ONE, xs[0]->v + i * xs[0]->d.batch_size(), 1, fx.v, 1));
#else
  auto res = *fx;
  const unsigned remainder = num_args % 4;
  switch (remainder) {
    case 0: res.setZero(); break;
    case 1: res = xs[0]->batch_matrix(0); break;
    case 2: res = xs[0]->batch_matrix(0) + xs[0]->batch_matrix(1); break;
    case 3: res = xs[0]->batch_matrix(0) + xs[0]->batch_matrix(1) + xs[0]->batch_matrix(2); break;
  }
  for (unsigned i = remainder; i < num_args; i += 4)
    res += xs[0]->batch_matrix(i) + xs[0]->batch_matrix(i+1) + xs[0]->batch_matrix(i+2) + xs[0]->batch_matrix(i+3);
#endif
}

void SumBatches::backward_impl(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
  assert(i == 0);
#if HAVE_CUDA
  for (unsigned i = 0; i < dEdxi.d.bd; ++i)
    CUBLAS_CHECK(cublasSaxpy(cublas_handle, fx.d.size(), kSCALAR_ONE, dEdf.v, 1, dEdxi.v + i * dEdxi.d.batch_size(), 1));
#else
  for (unsigned i = 0; i < dEdxi.d.bd; ++i)
    dEdxi.batch_matrix(i) += *dEdf;
#endif
}

void Average::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  const unsigned num_args = xs.size();
  if (num_args == 1) {
    fx.v = xs[0]->v;
    return;
  }
  auto res = fx.vec();
  const unsigned remainder = num_args % 4;
  switch (remainder) {
    case 0: res.setZero(); break;
    case 1: res = xs[0]->vec(); break;
    case 2: res = xs[0]->vec() + xs[1]->vec(); break;
    case 3: res = xs[0]->vec() + xs[1]->vec() + xs[2]->vec(); break;
  }
  for (unsigned i = remainder; i < num_args; i += 4)
    res += xs[i]->vec() + xs[i+1]->vec() + xs[i+2]->vec() + xs[i+3]->vec();
  res /= num_args;
}

void Average::backward_impl(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
  dEdxi.vec() += (dEdf.vec() / xs.size());
}

void Sqrt::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Sqrt not yet implemented for CUDA");
#else
  fx.vec() = xs[0]->vec().cwiseSqrt();
#endif
}

void Sqrt::backward_impl(const vector<const Tensor*>& xs,
                        const Tensor& fx,
                        const Tensor& dEdf,
                        unsigned i,
                        Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Sqrt not yet implemented for CUDA");
#else
  dEdxi.vec() += fx.vec().binaryExpr(*dEdf, FSqrtBackward());
#endif
}

void Erf::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Erf not yet implemented for CUDA");
#else
  fx.vec().array() = xs[0]->vec().array().erf();
#endif
}

void Erf::backward_impl(const vector<const Tensor*>& xs,
                        const Tensor& fx,
                        const Tensor& dEdf,
                        unsigned i,
                        Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Erf not yet implemented for CUDA");
#else
  dEdxi.vec() += xs[0]->vec().binaryExpr(dEdf.vec(), scalar_erf_backward_op<float>());
#endif
}

void Tanh::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#if HAVE_CUDA
  gpu::vtanh(fx.d.size(), xs[0]->v, fx.v);
#else
  fx.vec().array() = xs[0]->vec().array().tanh();
#endif
}

void Tanh::backward_impl(const vector<const Tensor*>& xs,
                      const Tensor& fx,
                      const Tensor& dEdf,
                      unsigned i,
                      Tensor& dEdxi) const {
#if HAVE_CUDA
  gpu::vtanh_backward(fx.d.size(), fx.v, dEdf.v, dEdxi.v);
#else
  dEdxi.vec() += fx.vec().binaryExpr(dEdf.vec(), scalar_tanh_backward_op<float>());
#endif
}

void Square::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Square not yet implemented for CUDA");
#else
  fx.vec().array() = xs[0]->vec().array().square();
#endif
}

void Square::backward_impl(const vector<const Tensor*>& xs,
                        const Tensor& fx,
                        const Tensor& dEdf,
                        unsigned i,
                        Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Square not yet implemented for CUDA");
#else
  dEdxi.vec() += dEdf.vec().cwiseProduct(xs[0]->vec()) * 2;
#endif
}

void Cube::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Square not yet implemented for CUDA");
#else
  fx.vec().array() = xs[0]->vec().array().cube();
#endif
}

void Cube::backward_impl(const vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i,
                    Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Cube not yet implemented for CUDA");
#else
//  *dEdxi += (*dEdf).cwiseProduct(x.cwiseProduct(x)) * 3;
  dEdxi.vec().array() += dEdf.vec().array() * xs[0]->vec().array().square() * 3;
#endif
}

void Exp::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Exp not yet implemented for CUDA");
#else
  fx.vec() = xs[0]->vec().array().exp();
#endif
}

void Exp::backward_impl(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Exp not yet implemented for CUDA");
#else
  dEdxi.vec() += dEdf.vec().cwiseProduct(fx.vec());
#endif
}

void LogGamma::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("LogGamma not yet implemented for CUDA");
#else
  fx.vec() = xs[0]->vec().array().lgamma();
#endif
}

void LogGamma::backward_impl(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("LogGamma not yet implemented for CUDA");
#else
  dEdxi.vec() += xs[0]->vec().binaryExpr(dEdf.vec(), FLogGammaBackward());
#endif
}

void Log::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#if HAVE_CUDA
  gpu::vlog(fx.d.size(), xs[0]->v, fx.v);
#else
  fx.vec() = xs[0]->vec().array().log();
#endif
}

void Log::backward_impl(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
#if HAVE_CUDA
  gpu::vlog_backward(fx.d.size(), xs[0]->v, dEdf.v, dEdxi.v);
#else
  dEdxi.vec() += dEdf.vec().cwiseQuotient(xs[0]->vec());
#endif
}

void Concatenate::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  unsigned rows = 0;
  for (auto x : xs) rows += x->d.rows();
  // the following should use auxiliary memory
  src_row_indices.resize(xs.size());
  unsigned ind = 0;
  unsigned k = 0;
  for (auto x : xs) {
    src_row_indices[k++] = ind;
    auto & xi = *x;
    const unsigned rows = xi.d.rows();
#if HAVE_CUDA
    assert(xi.d.cols() == 1); // this can be relaxed to the same everywhere
    CUDA_CHECK(cudaMemcpyAsync(&fx.v[ind], &xi.v[0], sizeof(float) * rows, cudaMemcpyDeviceToDevice));
#else
    (*fx).middleRows(ind, rows) = *xi;
#endif
    ind += rows;
  }
}

void Concatenate::backward_impl(const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  assert(i < src_row_indices.size());
  const unsigned rows = dEdxi.d.rows();
  const unsigned begin = src_row_indices[i];
#if HAVE_CUDA
  CUBLAS_CHECK(cublasSaxpy(cublas_handle, rows, kSCALAR_ONE, &dEdf.v[begin], 1, dEdxi.v, 1));
#else
  *dEdxi += (*dEdf).middleRows(begin, rows);
#endif
}

#define MAX_CONCAT_COLS_ARGS 512
size_t ConcatenateColumns::aux_storage_size() const {
  return MAX_CONCAT_COLS_ARGS * sizeof(unsigned);
}

void ConcatenateColumns::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  unsigned c = 0;
  assert(xs.size() < MAX_CONCAT_COLS_ARGS);
  for (unsigned i = 0; i < xs.size(); ++i) {
    static_cast<unsigned*>(aux_mem)[i] = c;
#if HAVE_CUDA
    assert(xs[i]->d.cols() == 1);
    // CUBLAS matricies are column-major, so just copy the memory
    auto & xi = *xs[i];
    const unsigned rows = xi.d.rows();
    CUDA_CHECK(cudaMemcpyAsync(&fx.v[i*rows], &xi.v[0], sizeof(float) * rows, cudaMemcpyDeviceToDevice));
#else
    auto xi = **xs[i];
    int d = xi.cols();
    (*fx).middleCols(c, d) = xi;
    c += d;
#endif
  }
}

void ConcatenateColumns::backward_impl(const vector<const Tensor*>& xs,
                                    const Tensor& fx,
                                    const Tensor& dEdf,
                                    unsigned i,
                                    Tensor& dEdxi) const {
#if HAVE_CUDA
  const unsigned rows = dEdxi.d.rows();
  const unsigned begin = i*rows;
  CUBLAS_CHECK(cublasSaxpy(cublas_handle, rows, kSCALAR_ONE, &dEdf.v[begin], 1, dEdxi.v, 1));
#else
  auto dEdx = *dEdxi;
  int d = dEdx.cols();
  int c = static_cast<unsigned*>(aux_mem)[i];
  dEdx += (*dEdf).middleCols(c, d);
#endif
}

void PairwiseRankLoss::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#if HAVE_CUDA
  gpu::vpairwise_rank_loss(fx.d.size(), margin, xs[0]->v, xs[1]->v, fx.v);
#else
  auto a = xs[0]->vec();
  auto b = xs[1]->vec();
  fx.vec() = a.binaryExpr(b, FPairwiseRankLoss(margin));
#endif
}

void PairwiseRankLoss::backward_impl(const vector<const Tensor*>& xs,
                                const Tensor& fx,
                                const Tensor& dEdf,
                                unsigned i,
                                Tensor& dEdxi) const {
#if HAVE_CUDA
  gpu::vpairwise_rank_loss_backward(dEdf.d.size(), (i == 0), fx.v, dEdf.v, dEdxi.v);
#else
  if (i == 0) {
    dEdxi.vec() -= fx.vec().binaryExpr(dEdf.vec(), FRectifyBackward());
  } else {
    dEdxi.vec() += fx.vec().binaryExpr(dEdf.vec(), FRectifyBackward());
  }
#endif
}

size_t Hinge::aux_storage_size() const {
  return dim.size() * sizeof(float);
}

void Hinge::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
#ifdef HAVE_CUDA
  throw std::runtime_error("Hinge not yet implemented for CUDA");
#else
  auto x = **xs[0];
  const unsigned rows = x.rows();
  float y = 0;
  float* eloss = static_cast<float*>(aux_mem);
  const real mlystar = margin - x(*pelement);
  for (unsigned i = 0; i < rows; ++i) {
    if (*pelement != i) {
      eloss[i] = max(0.f, mlystar + x(i));
      y += eloss[i];
    } else {
      eloss[i] = 0;
    }
  }
  fx.v[0] = y;
#endif
}

void Hinge::backward_impl(const vector<const Tensor*>& xs,
                       const Tensor& fx,
                       const Tensor& dEdf,
                       unsigned i,
                       Tensor& dEdxi) const {
  assert(i == 0);
#ifdef HAVE_CUDA
  throw std::runtime_error("Hinge not yet implemented for CUDA");
#else
  if (fx.v[0]) { // there was some loss
    const float d = dEdf.v[0];
    const unsigned rows = dEdxi.d.rows();
    const float* eloss = static_cast<const float*>(aux_mem);
    unsigned tne = 0;  // total number of errors
    for (unsigned i = 0; i < rows; ++i)
      if (eloss[i] > 0) {
        (*dEdxi)(i) += d;
        ++tne;
      }
    (*dEdxi)(*pelement) -= d * tne;
  }
#endif
}

void Identity::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.d = xs[0]->d;
  fx.v = xs[0]->v;
}

void Identity::backward_impl(const vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const {
  dEdxi.vec() += dEdf.vec();
}

void NoBackprop::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.d = xs[0]->d;
  fx.v = xs[0]->v;
}

void NoBackprop::backward_impl(const vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const {
  // no op
}

void MaxPooling1D::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
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

void MaxPooling1D::backward_impl(const vector<const Tensor*>& xs,
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

void Softmax::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  if (xs[0]->d.cols() == 1) {
#if HAVE_CUDA
    gpu::softmax(xs[0]->d.size(), xs[0]->v, fx.v);
#else
    auto x = **xs[0];
    if (x.rows() == 1) {
      fx.v[0] = 1;
    } else {
      *fx = x.unaryExpr(FSoftmaxNormalize(logsumexp(x)));
    }
#endif
  } else {
    throw std::runtime_error("Softmax not yet implemented for multiple columns");
  }
}

void Softmax::backward_impl(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
#if HAVE_CUDA
  gpu::softmax_backward(fx.d.size(), fx.v, dEdf.v, dEdxi.v);
#else
  auto y = *fx;
  if (y.rows() == 1) { return; } // no error if softmax = 0
  float off_diag_sum = -y.cwiseProduct(*dEdf).sum();
  *dEdxi += y.binaryExpr(*dEdf, FSoftmaxBackward(off_diag_sum));
#endif
}

void PickNegLogSoftmax::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  if (xs[0]->d.cols() == 1) {
    logz = (float*)fxs->allocate(sizeof(float)*fx.d.batch_elems());
#if HAVE_CUDA
    if(pval) {
      gpu::pnlsoftmax(xs[0]->d.size(), *pval, xs[0]->v, fx.v, logz);
    } else {
      // TODO: It'd be nice to have a kernel that did all batches at once
      assert(pvals);
      assert(pvals->size() == fx.d.batch_elems());
      for(unsigned b = 0; b < pvals->size(); ++b)
        gpu::pnlsoftmax(xs[0]->d.batch_size(), (*pvals)[b], xs[0]->batch_ptr(b), fx.v+b, logz+b);
    }
#else
    if(pval) {
      auto x = **xs[0];
      *logz = logsumexp(x);
      fx.v[0] = *logz - x(*pval);
    } else {
      assert(pvals);
      assert(pvals->size() == fx.d.batch_elems());
      for(unsigned b = 0; b < pvals->size(); ++b) {
        auto x = xs[0]->batch_matrix(b);
        logz[b] = logsumexp(x);
        fx.v[b] = logz[b] - x((*pvals)[b]);
      }
    }
#endif
  } else {
    throw std::runtime_error("PickNegLogSoftmax::forward not yet implemented for multiple columns");
  }
}

void PickNegLogSoftmax::backward_impl(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  if (xs[0]->d.cols() == 1) {
#if HAVE_CUDA
    if(pval) {
      const auto elem = *pval;
      gpu::pnlsoftmax_backward(dEdxi.d.size(), elem, xs[0]->v, dEdf.v, logz, dEdxi.v);
    } else {
      assert(pvals);
      assert(pvals->size() == fx.d.batch_elems()); 
      // TODO: Again, it would be nice to do this with a single kernel
      for(unsigned b = 0; b < pvals->size(); ++b) {
        const auto elem = (*pvals)[b];
        gpu::pnlsoftmax_backward(dEdxi.d.batch_size(), elem, xs[0]->batch_ptr(b), dEdf.v+b, logz+b, dEdxi.batch_ptr(b));
      }
    }
#else
    if(pval) {
      const auto elem = *pval;
      const float err = dEdf.v[0];
      auto x = **xs[0];
      // logz is computed in the forward pass and cached
      *dEdxi += x.unaryExpr(FNegLogSoftmaxBackward(*logz, err));
      //*dEdxi += x.unaryExpr(scalar_nlsoftmax_backward_op<float>(*logz, err));
      (*dEdxi)(elem) -= err;
    } else {
      assert(pvals);
      assert(pvals->size() == fx.d.batch_elems()); 
      for(unsigned b = 0; b < pvals->size(); ++b) {
        const auto elem = (*pvals)[b];
        const float err = dEdf.v[b];
        auto x = xs[0]->batch_matrix(b);
        auto dEdxi_mat = dEdxi.batch_matrix(b);
        dEdxi_mat += x.unaryExpr(FNegLogSoftmaxBackward(logz[b], err));
        //dEdxi_mat += x.unaryExpr(scalar_nlsoftmax_backward_op<float>(logz[b], err));
        dEdxi_mat(elem) -= err;
      }
    }
#endif
  } else {
    throw std::runtime_error("PickNegLogSoftmax::backward not yet implemented for multiple columns");
  }
}

void LogSoftmax::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
  if (xs[0]->d.cols() == 1) {
#if HAVE_CUDA
    throw std::runtime_error("LogSoftmax::forward not yet implemented for CUDA");
#else
    auto x = **xs[0];
    *fx = x.unaryExpr(FLogSoftmaxNormalize(logsumexp(x)));
#endif
  } else {
    throw std::runtime_error("LogSoftmax::forward not yet implemented for multiple columns");
  }
}

void LogSoftmax::backward_impl(const vector<const Tensor*>& xs,
                          const Tensor& fx,
                          const Tensor& dEdf,
                          unsigned i,
                          Tensor& dEdxi) const {
  if (xs[0]->d.cols() == 1) {
#if HAVE_CUDA
    throw std::runtime_error("LogSoftmax::backward not yet implemented for CUDA");
#else
    float off_diag_sum = -(*fx).binaryExpr(*dEdf, FWeightedError()).sum();
    *dEdxi += (*fx).binaryExpr(*dEdf, FLogSoftmaxBackward(off_diag_sum));
#endif
  } else {
    throw std::runtime_error("LogSoftmax::backward not yet implemented for multiple columns");
  }
}

template <class T>
EIGEN_STRONG_INLINE real logsumexp(const T& x, const vector<unsigned>& denom) {
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

void RestrictedLogSoftmax::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("RestrictedLogSoftmax not yet implemented for CUDA");
#else
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
#endif
}

void RestrictedLogSoftmax::backward_impl(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  assert(i == 0);
#ifdef HAVE_CUDA
  throw std::runtime_error("RestrictedLogSoftmax not yet implemented for CUDA");
#else
  float z = 0;
  for (auto ind : denom)
    z += (*dEdf)(ind, 0);
  for (auto ind : denom)
    (*dEdxi)(ind, 0) += (*dEdf)(ind, 0) - expf((*fx)(ind, 0)) * z;
#endif
}

// x_1 is a vector
// y = (x_1)_{*pval}
void PickElement::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("PickElement not yet implemented for CUDA");
#else
  if(pval) {
    if (*pval >= xs[0]->d.rows()) {
      cerr << "PickElement::forward_impl requested element " << *pval
           << "from a vector of length " << xs[0]->d.rows() << endl;
      abort();
    }
    auto x = **xs[0];
    fx.v[0] = x(*pval);
  } else {
    assert(pvals);
    assert(pvals->size() == fx.d.batch_elems());
    for(unsigned b = 0; b < pvals->size(); ++b) {
      auto x = xs[0]->batch_matrix(b);
      fx.v[b] = x((*pvals)[b]);
    }
  }
#endif
}

// derivative is 0 in all dimensions except 1 for the selected element
void PickElement::backward_impl(const vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i,
                    Tensor& dEdxi) const {
  assert(i == 0);
#ifdef HAVE_CUDA
  throw std::runtime_error("PickElement not yet implemented for CUDA");
#else
  if(pval) {
    (*dEdxi)(*pval) += dEdf.v[0];
  } else {
    assert(pvals);
    for(unsigned b = 0; b < pvals->size(); ++b)
      dEdxi.batch_matrix(b)((*pvals)[b]) += dEdf.v[b];
  }
#endif
}

// x_1 is a vector
// y = (x_1)[start:end]
// slice of vector from index start (inclusive) to index end (exclusive)
void PickRange::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
  auto x = **xs[0];
  assert(x.cols() == 1);
  assert(start >= 0);
  assert(end <= x.rows());
  assert(start < end);
  assert(int(fx.d.rows()) == int(end-start));
#if HAVE_CUDA
  CUDA_CHECK(cudaMemcpyAsync(&fx.v[0], &xs[0]->v[start], sizeof(float) * (end-start), cudaMemcpyDeviceToDevice));
#else
  (*fx) = x.block(start, 0, end-start, 1);
#endif
}

// derivative is 0 in all dimensions except the slice range
void PickRange::backward_impl(const vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i,
                    Tensor& dEdxi) const {
  assert(i == 0);
  assert(int(dEdf.d.rows()) == int(end-start));
  assert(dEdf.d.cols() == 1);
#if HAVE_CUDA
  CUBLAS_CHECK(cublasSaxpy(cublas_handle, end-start, kSCALAR_ONE, dEdf.v, 1, &dEdxi.v[start], 1));
#else
  (*dEdxi).block(start, 0, end-start, 1) += (*dEdf);
#endif
}

#if HAVE_CUDA
inline void CUDAMatrixMultiply(const Tensor& l, const Tensor& r, Tensor& y, const float* acc_scalar) {
  // if (r.d.ndims() == 1 || r.d.cols() == 1) {
  //   CUBLAS_CHECK(cublasSgemv(cublas_handle, CUBLAS_OP_N, l.d.rows(), l.d.cols(),
  //              kSCALAR_ONE, l.v, l.d.rows(), r.v, 1, acc_scalar, y.v, 1));
  // } else {
  //   CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
  //         y.d.rows(), y.d.cols(), l.d.cols(),
  //         kSCALAR_ONE,
  //         l.v, l.d.rows(),
  //         r.v, r.d.rows(),
  //         acc_scalar, y.v, y.d.rows()));
  // }
  if(l.d.bd == 1) {
    // If the left side has one batch, multiply by columns
    // [x, z, b] = [x, y] * [y, z, b]
    // -> [x, z*b] = [x, y], [y, z*b]
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
          y.d.rows(), y.d.cols() * y.d.batch_elems(), l.d.cols(),
          kSCALAR_ONE,
          l.v, l.d.rows(),
          r.v, r.d.rows(),
          acc_scalar, y.v, y.d.rows()));
  } else {
    // Otherwise, loop over the batches
    assert(r.d.bd == 1 || r.d.bd == l.d.bd);
    for(unsigned b = 0; b < l.d.bd; ++b) {
      CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            y.d.rows(), y.d.cols(), l.d.cols(),
            kSCALAR_ONE,
            l.batch_ptr(b), l.d.rows(),
            r.batch_ptr(b), r.d.rows(),
            acc_scalar, y.batch_ptr(b), y.d.rows()));
    }
  }
}
#endif

void MatrixMultiply::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 2);
#if HAVE_CUDA
  // fx = 0*fx + xs[0] * xs[1]
  CUDAMatrixMultiply(*xs[0], *xs[1], fx, kSCALAR_ZERO);
#else
  assert(fx.d.bd == max(xs[0]->d.bd, xs[1]->d.bd));
  if(xs[0]->d.bd == 1) {
    // If the left side has one batch, multiply by columns
    // [x, z, b] = [x, y] * [y, z, b]
    // -> [x, z*b] = [x, y], [y, z*b]
    fx.colbatch_matrix().noalias() = **xs[0] * xs[1]->colbatch_matrix();
  } else {
    // Otherwise, loop over the batches
    assert(xs[1]->d.bd == 1 || xs[1]->d.bd == xs[0]->d.bd);
    for(unsigned b = 0; b < xs[0]->d.bd; ++b)
      fx.batch_matrix(b).noalias() = xs[0]->batch_matrix(b) * xs[1]->batch_matrix(b);
  }
#endif
}

void MatrixMultiply::backward_impl(const vector<const Tensor*>& xs,
                                const Tensor& fx,
                                const Tensor& dEdf,
                                unsigned i,
                                Tensor& dEdxi) const {
  assert(i < 2);
  int max_b = max(xs[0]->d.bd, xs[1]->d.bd);
#if HAVE_CUDA
  if (i == 0) {
    for(int b = 0; b < max_b; ++b)
      CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
            dEdxi.d.rows(), dEdxi.d.cols(), dEdf.d.cols(),
            kSCALAR_ONE,
            dEdf.batch_ptr(b), dEdf.d.rows(),
            xs[1]->batch_ptr(b), xs[1]->d.rows(),
            kSCALAR_ONE, dEdxi.batch_ptr(b), dEdxi.d.rows()));
  } else {
    // Do a single multiply if xs[0] has one batch
    if(xs[0]->d.bd == 1) {
      // dEdxi.colbatch_matrix().noalias() += (**xs[0]).transpose() * dEdf.colbatch_matrix();
      CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
            dEdxi.d.rows(), dEdxi.d.cols()*dEdxi.d.batch_elems(), xs[0]->d.rows(),
            kSCALAR_ONE,
            xs[0]->v, xs[0]->d.rows(),
            dEdf.v, dEdf.d.rows(),
            kSCALAR_ONE, dEdxi.v, dEdxi.d.rows()));
    } else {
      for(int b = 0; b < max_b; ++b)
        CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
              dEdxi.d.rows(), dEdxi.d.cols(), xs[0]->d.rows(),
              kSCALAR_ONE,
              xs[0]->batch_ptr(b), xs[0]->d.rows(),
              dEdf.batch_ptr(b), dEdf.d.rows(),
              kSCALAR_ONE, dEdxi.batch_ptr(b), dEdxi.d.rows()));
    }
  }
#else
  if (i == 0) {
    for(int b = 0; b < max_b; ++b)
      dEdxi.batch_matrix(b).noalias() += dEdf.batch_matrix(b) * xs[1]->batch_matrix(b).transpose();
  } else {
    if(xs[0]->d.bd == 1) {
      dEdxi.colbatch_matrix().noalias() += (**xs[0]).transpose() * dEdf.colbatch_matrix();
    } else {
      for(int b = 0; b < max_b; ++b)
        dEdxi.batch_matrix(b).noalias() += xs[0]->batch_matrix(b).transpose() * dEdf.batch_matrix(b);
    }
  }
#endif
}

void CwiseQuotient::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 2);
#ifdef HAVE_CUDA
  throw std::runtime_error("CwiseQuotient::forward not yet implemented for CUDA");
#else
  auto x1 = xs[0]->vec();
  auto x2 = xs[1]->vec();
  fx.vec() = x1.cwiseQuotient(x2);
#endif
}

void CwiseQuotient::backward_impl(const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  assert(i < 2);
#ifdef HAVE_CUDA
  throw std::runtime_error("CwiseQuotient::backward not yet implemented for CUDA");
#else
  if (i == 0) {
    auto x2 = xs[1]->vec();
    dEdxi.vec() += dEdf.vec().cwiseQuotient(x2);
  } else { // i = 1
    auto x1 = xs[0]->vec();
    auto x2 = xs[1]->vec();
    dEdxi.vec() -= dEdf.vec().cwiseQuotient(x2.cwiseProduct(x2)).cwiseProduct(x1);
  }
#endif
}

void CwiseMultiply::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 2);
#if HAVE_CUDA
  gpu::vcwise_product(fx.d.size(), xs[0]->v, xs[1]->v, fx.v);
#else
  auto x1 = xs[0]->vec();
  auto x2 = xs[1]->vec();
  fx.vec() = x1.cwiseProduct(x2);
#endif
}

void CwiseMultiply::backward_impl(const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  assert(i < 2);
  if (i == 0) {
#if HAVE_CUDA
    gpu::vcwise_product_backward(fx.d.size(), dEdf.v, xs[1]->v, dEdxi.v);
#else
    auto x2 = xs[1]->vec();
    dEdxi.vec() += dEdf.vec().cwiseProduct(x2);
#endif
  } else {
#if HAVE_CUDA
    gpu::vcwise_product_backward(fx.d.size(), dEdf.v, xs[0]->v, dEdxi.v);
#else
    auto x1 = xs[0]->vec();
    dEdxi.vec() += dEdf.vec().cwiseProduct(x1);
#endif
  }
}

void AffineTransform::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() % 2 == 1);
  if (xs.size() == 1) {
    fx.v = xs[0]->v;
    return;
  } else {
#if HAVE_CUDA
    for (unsigned i = 1; i < xs.size(); i += 2)
      // fx = (acc_sclar)*fx + xs[0] * xs[1]
      CUDAMatrixMultiply(*xs[i], *xs[i + 1], fx, (i == 1) ? kSCALAR_ZERO : kSCALAR_ONE);
    if(fx.d.bd == xs[0]->d.bd) {
      CUBLAS_CHECK(cublasSaxpy(cublas_handle, fx.d.size(), kSCALAR_ONE, xs[0]->v, 1, fx.v, 1));
    } else {
      // TODO: Any better way to do broadcasting?
      for(unsigned b = 0; b < fx.d.bd; ++b)
        CUBLAS_CHECK(cublasSaxpy(cublas_handle, fx.d.batch_size(), kSCALAR_ONE, xs[0]->batch_ptr(b), 1, fx.batch_ptr(b), 1));
    }
#else
    // Add, using broadcasting or not
    if(fx.d.bd > 1 && xs[0]->d.bd == 1) {
      fx.rowcol_matrix().colwise() = xs[0]->vec();
    } else {
      for(unsigned b = 0; b < fx.d.bd; ++b)
        fx.batch_matrix(b) = xs[0]->batch_matrix(b);
    }

    // Multiply
    for (unsigned i = 1; i < xs.size(); i += 2) {
      if(xs[i]->d.bd == 1 && xs[i+1]->d.bd == fx.d.bd) {
        fx.colbatch_matrix().noalias() += **xs[i] * xs[i+1]->colbatch_matrix();
      } else {
        assert(xs[i+1]->d.bd == 1 || xs[i+1]->d.bd == xs[i]->d.bd);
        for(unsigned b = 0; b < fx.d.bd; ++b) {
          fx.batch_matrix(b).noalias() += xs[i]->batch_matrix(b) * xs[i+1]->batch_matrix(b);
        }
      }
    }

#endif
  }
}

void AffineTransform::backward_impl(const vector<const Tensor*>& xs,
                               const Tensor& fx,
                               const Tensor& dEdf,
                               unsigned i,
                               Tensor& dEdxi) const {
  assert(i < xs.size());
  if (i == 0) { // bias term
#if HAVE_CUDA
    CUBLAS_CHECK(cublasSaxpy(cublas_handle, dEdxi.d.size(), kSCALAR_ONE, dEdf.v, 1, dEdxi.v, 1));
#else
    // Add, using broadcasting or not
    if(dEdxi.d.bd == 1 && dEdf.d.bd > 1) {
      (*dEdxi) += dEdf.rowcol_matrix().rowwise().sum();
    } else {
      for(unsigned b = 0; b < dEdf.d.bd; ++b)
        dEdxi.batch_matrix(b) += dEdf.batch_matrix(b);
    }
#endif
  } else if (i % 2 == 1) { // left argument of matrix multiply
    int max_b = max(dEdf.d.bd, xs[i+1]->d.bd);
#if HAVE_CUDA
    for(int b = 0; b < max_b; ++b)
      CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
            dEdxi.d.rows(), dEdxi.d.cols(), dEdf.d.cols(),
            kSCALAR_ONE,
            dEdf.batch_ptr(b), dEdf.d.rows(),
            xs[i+1]->batch_ptr(b), xs[i+1]->d.rows(),
            kSCALAR_ONE, dEdxi.batch_ptr(b), dEdxi.d.rows()));
#else
    for(int b = 0; b < max_b; ++b)
      dEdxi.batch_matrix(b).noalias() += dEdf.batch_matrix(b) * xs[i+1]->batch_matrix(b).transpose();
#endif
  } else {  // right argument of matrix multiply
    int max_b = max(xs[i-1]->d.bd, dEdf.d.bd);
#if HAVE_CUDA
    // Do a single multiply if xs[i-1] has one batch
    if(xs[i-1]->d.bd == 1) {
      CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, 
            dEdxi.d.rows(), dEdxi.d.cols()*dEdxi.d.batch_elems(), xs[i-1]->d.rows(),
            kSCALAR_ONE,
            xs[i-1]->v, xs[i-1]->d.rows(),
            dEdf.v, dEdf.d.rows(),
            kSCALAR_ONE, dEdxi.v, dEdxi.d.rows()));
    } else {
      for(int b = 0; b < max_b; ++b)
        CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
              dEdxi.d.rows(), dEdxi.d.cols(), xs[i-1]->d.rows(),
              kSCALAR_ONE,
              xs[i-1]->batch_ptr(b), xs[i-1]->d.rows(),
              dEdf.batch_ptr(b), dEdf.d.rows(),
              kSCALAR_ONE, dEdxi.batch_ptr(b), dEdxi.d.rows()));
    }
#else
    if(xs[i-1]->d.bd == 1 && dEdxi.d.bd == dEdf.d.bd) {
      dEdxi.colbatch_matrix().noalias() += (**xs[i-1]).transpose() * dEdf.colbatch_matrix();
    } else {
      for(int b = 0; b < max_b; ++b)
        dEdxi.batch_matrix(b).noalias() += xs[i-1]->batch_matrix(b).transpose() * dEdf.batch_matrix(b);
    }
#endif
  }
}

void Negate::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
#if HAVE_CUDA
  gpu::vnegate(fx.d.size(), xs[0]->v, fx.v);
#else
  auto x = xs[0]->vec();
  fx.vec() = -x;
#endif
}

void Negate::backward_impl(const vector<const Tensor*>& xs,
                      const Tensor& fx,
                      const Tensor& dEdf,
                      unsigned i,
                      Tensor& dEdxi) const {
  assert(i == 0);
#if HAVE_CUDA
  gpu::vnegate_backward(fx.d.size(), dEdf.v, dEdxi.v);
#else
  dEdxi.vec() -= dEdf.vec();
#endif
}

void Rectify::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
#if HAVE_CUDA
  gpu::vrelu(fx.d.size(), xs[0]->v, fx.v);
#else
  auto x = xs[0]->vec();
  fx.vec() = x.cwiseMax(0.f);
#endif
}

void Rectify::backward_impl(const vector<const Tensor*>& xs,
                         const Tensor& fx,
                         const Tensor& dEdf,
                         unsigned i,
                         Tensor& dEdxi) const {
#if HAVE_CUDA
  gpu::vrelu_backward(fx.d.size(), fx.v, dEdf.v, dEdxi.v);
#else
  dEdxi.vec() += fx.vec().binaryExpr(dEdf.vec(), FRectifyBackward());
#endif
}

void HuberDistance::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 2);
#ifdef HAVE_CUDA
  throw std::runtime_error("HuberDistance not yet implemented for CUDA");
#else
  auto x = *xs[0];
  auto y = *xs[1];
  const FHuberForward fhf(d);
  const size_t s = x.d.size();
  float dist = 0;
  for (size_t i = 0; i < s; ++i)
    dist += fhf(x.v[i] - y.v[i]);
  fx.v[0] = dist;
#endif
}

void HuberDistance::backward_impl(const vector<const Tensor*>& xs,
                          const Tensor& fx,
                          const Tensor& dEdf,
                          unsigned i,
                          Tensor& dEdxi) const {
  assert(i < 2);
#ifdef HAVE_CUDA
  throw std::runtime_error("HuberDistance not yet implemented for CUDA");
#else
  auto x = **xs[i];
  auto y = **xs[1-i];
  *dEdxi += (x - y).unaryExpr(FHuberBackward(d, dEdf.v[0]));
#endif
}

void L1Distance::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 2);
#ifdef HAVE_CUDA
  throw std::runtime_error("L1Distance not yet implemented for CUDA");
#else
  auto x = xs[0]->vec();
  auto y = xs[1]->vec();
  fx.v[0] = (x - y).lpNorm<1>();
#endif
}

void L1Distance::backward_impl(const vector<const Tensor*>& xs,
                          const Tensor& fx,
                          const Tensor& dEdf,
                          unsigned i,
                          Tensor& dEdxi) const {
  assert(i < 2);
#ifdef HAVE_CUDA
  throw std::runtime_error("L1Distance not yet implemented for CUDA");
#else
  auto x = xs[i]->vec();
  auto y = xs[1-i]->vec();
  dEdxi.vec() += (x - y).unaryExpr(FL1Backward(dEdf.v[0]));
#endif
}

void PoissonRegressionLoss::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("PoissonRegressionLoss not yet implemented for CUDA");
#else
  const auto y = *pty;
  const auto z = lgamma(y + 1);
  const auto x = xs[0]->v[0];
  fx.v[0] = expf(x) + z - y * x;
#endif
}

void PoissonRegressionLoss::backward_impl(const vector<const Tensor*>& xs,
                          const Tensor& fx,
                          const Tensor& dEdf,
                          unsigned i,
                          Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("PoissonRegressionLoss not yet implemented for CUDA");
#else
  const auto x = xs[0]->v[0];
  const auto y = *pty;
  auto& dEdx = dEdxi.v[0];
  dEdx += expf(x) - y;
#endif
}

void SquaredNorm::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
#if HAVE_CUDA
  throw std::runtime_error("SquaredNorm not yet implemented for CUDA");
#else
  fx.v[0] = xs[0]->vec().squaredNorm();
#endif
}

void SquaredNorm::backward_impl(const vector<const Tensor*>& xs,
                                 const Tensor& fx,
                                 const Tensor& dEdf,
                                 unsigned i,
                                 Tensor& dEdxi) const {
  assert(i < 1);
#if HAVE_CUDA
  throw std::runtime_error("SquaredNorm not yet implemented for CUDA");
#else
  real scale = dEdf.v[0] * 2;
  dEdxi.vec().noalias() += scale * xs[0]->vec();
#endif
}

void SquaredEuclideanDistance::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 2);
#if HAVE_CUDA
  gpu::sqeucdist(xs[0]->d.size(), xs[0]->v, xs[1]->v, fx.v);
#else
  auto x1 = **xs[0];
  auto x2 = **xs[1];
  fx.v[0] = (x1 - x2).squaredNorm();
#endif
}

void SquaredEuclideanDistance::backward_impl(const vector<const Tensor*>& xs,
                                 const Tensor& fx,
                                 const Tensor& dEdf,
                                 unsigned i,
                                 Tensor& dEdxi) const {
  assert(i < 2);
#if HAVE_CUDA
  gpu::sqeucdist_backward(xs[0]->d.size(), dEdf.v, xs[0]->v, xs[1]->v, dEdxi.v, i);
#else
  auto x1 = **xs[0];
  auto x2 = **xs[1];
  real scale = dEdf.v[0] * 2;
  if (i == 1) scale = -scale;
  *dEdxi += scale * (x1 - x2);
#endif
}

void LogisticSigmoid::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
#if HAVE_CUDA
  gpu::vlogistic(fx.d.size(), xs[0]->v, fx.v);
#else
  auto x = xs[0]->vec();
  fx.vec() = x.unaryExpr(scalar_logistic_sigmoid_op<float>());
#endif
}

void LogisticSigmoid::backward_impl(const vector<const Tensor*>& xs,
                                 const Tensor& fx,
                                 const Tensor& dEdf,
                                 unsigned i,
                                 Tensor& dEdxi) const {
#if HAVE_CUDA
  gpu::vlogistic_backward(dEdf.d.size(), fx.v, dEdf.v, dEdxi.v);
#else
  dEdxi.vec() += fx.vec().binaryExpr(dEdf.vec(), scalar_logistic_sigmoid_backward_op<float>());
#endif
}

void SoftSign::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
#ifdef HAVE_CUDA
  throw std::runtime_error("SoftSign not yet implemented for CUDA");
#else
  auto x = xs[0]->vec();
  fx.vec() = x.unaryExpr(FSoftSign());
#endif
}

void SoftSign::backward_impl(const vector<const Tensor*>& xs,
                        const Tensor& fx,
                        const Tensor& dEdf,
                        unsigned i,
                        Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("SoftSign not yet implemented for CUDA");
#else
  dEdxi.vec() += fx.vec().binaryExpr(dEdf.vec(), FSoftSignBackward());
#endif
}

void BinaryLogLoss::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("BinaryLogLoss not yet implemented for CUDA");
#else
  auto x = *xs[0];
  auto y = *xs[1];
  FBinaryLogLoss bll;
  const size_t s = x.d.size();
  float dist = 0;
  for (size_t i = 0; i < s; ++i)
    dist += bll(x.v[i], y.v[i]);
  fx.v[0] = dist;
#endif
}

void BinaryLogLoss::backward_impl(const vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("BinaryLogLoss not yet implemented for CUDA");
#else
  *dEdxi += (**xs[i]).binaryExpr(**xs[1-i], FBinaryLogLossBackward(dEdf.v[0]));
#endif
}

string Zeroes::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "zeroes(" << dim << ')';
  return s.str();
}

Dim Zeroes::dim_forward(const vector<Dim>& xs) const {
  return dim;
}

void Zeroes::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
#if HAVE_CUDA
  cudaMemsetAsync(fx.v, 0, dim.size() * sizeof(float));
#else
  memset(fx.v, 0, dim.size() * sizeof(float));
#endif
}

void Zeroes::backward_impl(const vector<const Tensor*>& xs,
                           const Tensor& fx,
                           const Tensor& dEdf,
                           unsigned i,
                           Tensor& dEdxi) const {
  throw std::runtime_error("Called backward() on an arity 0 node");
}

} // namespace cnn
