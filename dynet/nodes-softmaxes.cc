#include "dynet/tensor-eigen.h"
#include "dynet/nodes-softmaxes.h"

#include "dynet/nodes-impl-macros.h"
#include "dynet/functors.h"

using namespace std;

namespace dynet {

// ************* Softmax *************

#ifndef __CUDACC__

string Softmax::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "softmax(" << arg_names[0] << ')';
  return s.str();
}

Dim Softmax::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Softmax");
  DYNET_ARG_CHECK(xs[0].nd <= 2, "Bad input dimensions in Softmax, must be 2 or fewer: " << xs);
  DYNET_ARG_CHECK(dimension < xs[0].nd, "reduction dimension must be < number of dimensions, was " << dimension);
  return xs[0];
}

int Softmax::autobatch_sig(const ComputationGraph & cg, SigMap &sm) const {
  Sig s(nt::softmax);
  s.add_dim(dim);
  return sm.get_idx(s);
}

std::vector<int> Softmax::autobatch_concat(const ComputationGraph & cg) const {
  return vector<int>(1, 1);
}

#endif

template<class MyDevice>
void Softmax::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed dimension check in Softmax::forward");
  AlignedMemoryPool* scratch_allocator = fx.device->pools[(int)DeviceMempool::SCS];
  if(dimension==0){
#ifdef __CUDACC__ // GPU impl
    Tensor z(Dim({xs[0]->d.cols()},fx.d.bd), nullptr, fx.device, DeviceMempool::FXS);
    z.v = static_cast<float*>(scratch_allocator->allocate(z.d.size() * sizeof(float)));
    Tensor m(Dim({xs[0]->d.cols()},fx.d.bd), nullptr, fx.device, DeviceMempool::FXS);
    m.v = static_cast<float*>(scratch_allocator->allocate(m.d.size() * sizeof(float)));
    Eigen::array<int, 1> red_dim = {0};
    tb<1>(m).device(*dev.edevice) = tb<2>(*xs[0]).maximum(red_dim);
    Eigen::array<int, 3> bcasts = {(int)xs[0]->d.rows(), 1, 1};
    Eigen::array<int, 3> morph = {1, (int)z.d[0], (int)z.d.bd};
    tb<2>(fx).device(*dev.edevice) = (tb<2>(*xs[0]) - tvec(m).reshape(morph).broadcast(bcasts)).exp();
    tb<1>(z).device(*dev.edevice) = tb<2>(fx).sum(red_dim);
    tb<2>(fx).device(*dev.edevice) = tb<2>(fx) / tvec(z).reshape(morph).broadcast(bcasts);
#else // CPU impl
    Tensor z(Dim({1}), nullptr, fx.device, DeviceMempool::FXS);
    z.v = static_cast<float*>(scratch_allocator->allocate(z.d.size() * sizeof(float)));
    Tensor m(Dim({1}), nullptr, fx.device, DeviceMempool::FXS);
    m.v = static_cast<float*>(scratch_allocator->allocate(m.d.size() * sizeof(float)));
    unsigned size = xs[0]->d[0], num_cols = xs[0]->d[1] * xs[0]->d.bd;
    Tensor col_x(Dim({xs[0]->d[0]}), (float*)xs[0]->v, fx.device, DeviceMempool::FXS);
    Tensor col_fx(Dim({xs[0]->d[0]}), (float*)fx.v, fx.device, DeviceMempool::FXS);
    for(size_t col = 0; col < num_cols; ++col) {
      t<0>(m) = tvec(col_x).maximum();
      tvec(col_fx) = (tvec(col_x) - m.v[0]).exp();
      t<0>(z) = tvec(col_fx).sum();
      tvec(col_fx) = tvec(col_fx) / z.v[0];
      col_x.v += size;
      col_fx.v += size;
    }
#endif
  } else {
    Tensor z(Dim({xs[0]->d.rows()},fx.d.bd), nullptr, fx.device, DeviceMempool::FXS);
    z.v = static_cast<float*>(scratch_allocator->allocate(z.d.size() * sizeof(float)));
    Tensor m(Dim({xs[0]->d.rows()},fx.d.bd), nullptr, fx.device, DeviceMempool::FXS);
    m.v = static_cast<float*>(scratch_allocator->allocate(m.d.size() * sizeof(float)));
    Eigen::array<int, 1> red_dim = {1};
    tb<1>(m).device(*dev.edevice) = tb<2>(*xs[0]).maximum(red_dim);
    Eigen::array<int, 3> bcasts = {1, (int)xs[0]->d.cols(), 1};
    Eigen::array<int, 3> morph = {(int)z.d[0], 1, (int)z.d.bd};
    tb<2>(fx).device(*dev.edevice) = (tb<2>(*xs[0]) - tvec(m).reshape(morph).broadcast(bcasts)).exp();
    tb<1>(z).device(*dev.edevice) = tb<2>(fx).sum(red_dim);
    tb<2>(fx).device(*dev.edevice) = tb<2>(fx) / tvec(z).reshape(morph).broadcast(bcasts);

  }
  scratch_allocator->free();
}

template<class MyDevice>
void Softmax::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  AlignedMemoryPool* scratch_allocator = fx.device->pools[(int)DeviceMempool::SCS];
  Tensor z(Dim({fx.d.cols()},fx.d.bd), nullptr, fx.device, DeviceMempool::FXS);
  z.v = static_cast<float*>(scratch_allocator->allocate(z.d.size() * sizeof(float)));
  Eigen::array<int, 1> red_axis = {0};
  tb<1>(z).device(*dev.edevice) = (tb<2>(fx) * tb<2>(dEdf)).sum(red_axis);
  if(dimension==0){
#ifdef __CUDACC__ // GPU impl
    Eigen::array<int, 3> bcast = {(int)xs[0]->d.rows(), 1, 1};
    Eigen::array<int, 3> morph = {1, (int)z.d[0], (int)z.d.bd};
    tb<2>(dEdxi).device(*dev.edevice) += (tb<2>(dEdf) - tvec(z).reshape(morph).broadcast(bcast)) * tb<2>(fx);
#else // CPU impl
    unsigned size = xs[0]->d[0], num_cols = xs[0]->d[1] * xs[0]->d.bd;
    Tensor col_fx(Dim({xs[0]->d[0]}), (float*)fx.v, fx.device, DeviceMempool::FXS);
    Tensor col_dEdf(Dim({xs[0]->d[0]}), (float*)dEdf.v, fx.device, DeviceMempool::FXS);
    Tensor col_dEdxi(Dim({xs[0]->d[0]}), (float*)dEdxi.v, fx.device, DeviceMempool::FXS);
    for(size_t col = 0; col < num_cols; ++col) {
      tvec(col_dEdxi) += (tvec(col_dEdf) - z.v[col]) * tvec(col_fx);
      col_fx.v += size;
      col_dEdf.v += size;
      col_dEdxi.v += size;
    }
#endif
  } else {
    Tensor z(Dim({fx.d.rows()},fx.d.bd), nullptr, fx.device, DeviceMempool::FXS);
    z.v = static_cast<float*>(scratch_allocator->allocate(z.d.size() * sizeof(float)));
    Eigen::array<int, 1> red_axis = {1};
    tb<1>(z).device(*dev.edevice) = (tb<2>(fx) * tb<2>(dEdf)).sum(red_axis);
    Eigen::array<int, 3> bcast = {1, (int)xs[0]->d.cols(), 1};
    Eigen::array<int, 3> morph = {(int)z.d[0], 1, (int)z.d.bd};
    tb<2>(dEdxi).device(*dev.edevice) += (tb<2>(dEdf) - tvec(z).reshape(morph).broadcast(bcast)) * tb<2>(fx);
  }
  scratch_allocator->free();
}
DYNET_NODE_INST_DEV_IMPL(Softmax)


// ************* LogSoftmax *************

#ifndef __CUDACC__

string LogSoftmax::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "log_softmax(" << arg_names[0] << ')';
  return s.str();
}

Dim LogSoftmax::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in LogSoftmax")
  DYNET_ARG_CHECK(xs[0].nd <= 2, "Bad input dimensions in LogSoftmax, must be 2 or fewer: " << xs);
  return xs[0];
}

size_t LogSoftmax::aux_storage_size() const {
  return 2 * dim.size() / dim.rows() * sizeof(float);
}

#endif

template<class MyDevice>
void LogSoftmax::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in LogSoftmax::forward");
  Tensor z(Dim({xs[0]->d.cols()},fx.d.bd), (float*)aux_mem, fx.device, DeviceMempool::FXS);
  Tensor m(Dim({xs[0]->d.cols()},fx.d.bd), (float*)aux_mem + z.d.size(), fx.device, DeviceMempool::FXS);
  TensorTools::logsumexp_dev(dev, *xs[0], m, z);
  if(fx.d.size() == fx.d.rows()) {
#ifdef __CUDACC__
    Eigen::array<int, 1> bcast;
    bcast[0] = xs[0]->d[0];
    t<1>(fx).device(*dev.edevice) = t<1>(*xs[0]) - t<1>(z).broadcast(bcast);
#else
    t<1>(fx).device(*dev.edevice) = t<1>(*xs[0]) - as_scalar(z);
#endif
  } else {
#ifdef __CUDACC__ // GPU impl
    Eigen::array<int, 3> bcasts = {(int)xs[0]->d.rows(), 1, 1};
    Eigen::array<int, 3> morph = {1, (int)z.d[0], (int)z.d.bd};
    tb<2>(fx).device(*dev.edevice) = tb<2>(*xs[0]) - tvec(z).reshape(morph).broadcast(bcasts);
#else // CPU impl
    unsigned size = xs[0]->d[0], num_cols = xs[0]->d[1] * xs[0]->d.bd;
    Tensor col_fx(Dim({xs[0]->d[0]}), (float*)fx.v, fx.device, DeviceMempool::FXS);
    Tensor col_x(Dim({xs[0]->d[0]}), (float*)xs[0]->v, fx.device, DeviceMempool::FXS);
    for(size_t col = 0; col < num_cols; ++col) {
      tvec(col_fx) = tvec(col_x) - z.v[col];
      col_x.v += size;
      col_fx.v += size;
    }
#endif
  }
}

template<class MyDevice>
void LogSoftmax::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  Tensor z(Dim({xs[0]->d.cols()},fx.d.bd), (float*)aux_mem, fx.device, DeviceMempool::FXS);
  Eigen::array<int, 1> red_axis; red_axis[0] = 0;
  tb<1>(z).device(*dev.edevice) = tb<2>(dEdf).sum(red_axis);
#ifdef __CUDACC__ // GPU impl
  Eigen::array<int, 3> bcast = {(int)fx.d.rows(), 1, 1};
  Eigen::array<int, 3> morph = {1, (int)z.d[0], (int)z.d.bd};
  tb<2>(dEdxi).device(*dev.edevice) += tb<2>(fx).exp() * -tvec(z).reshape(morph).broadcast(bcast) + tb<2>(dEdf);
#else // CPU impl
  unsigned size = xs[0]->d[0], num_cols = xs[0]->d[1] * xs[0]->d.bd;
  Tensor col_fx(Dim({xs[0]->d[0]}), (float*)fx.v, fx.device, DeviceMempool::FXS);
  Tensor col_dEdf(Dim({xs[0]->d[0]}), (float*)dEdf.v, fx.device, DeviceMempool::FXS);
  Tensor col_dEdxi(Dim({xs[0]->d[0]}), (float*)dEdxi.v, fx.device, DeviceMempool::FXS);
  for(size_t col = 0; col < num_cols; ++col) {
    tvec(col_dEdxi) += (tvec(col_fx).exp() * -z.v[col]) + tvec(col_dEdf);
    col_fx.v += size;
    col_dEdf.v += size;
    col_dEdxi.v += size;
  }
#endif
}
DYNET_NODE_INST_DEV_IMPL(LogSoftmax)

// ************* RestrictedLogSoftmax *************

#ifndef __CUDACC__

string RestrictedLogSoftmax::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "r_log_softmax(" << arg_names[0] << ')';
  return s.str();
}

Dim RestrictedLogSoftmax::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in RestrictedLogSoftmax")
  DYNET_ARG_CHECK(LooksLikeVector(xs[0]), "Bad input dimensions in RestrictedLogSoftmax: " << xs);
  return xs[0];
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

#endif

template<class MyDevice>
void RestrictedLogSoftmax::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in RestrictedLogSoftmax");
#ifdef __CUDACC__
  DYNET_NO_CUDA_IMPL_ERROR("RestrictedLogSoftmax forward");
#else
  // TODO create auxiliary mask with -infty's
  // and do usual LogSoftmax stuff
  if(denom.size() == 0)
    DYNET_INVALID_ARG("Number of elements in denominator of RestrictedLogSoftmax::forward must be zero");
  auto x = mat(*xs[0]);
  if(denom.size() == 0)
    DYNET_RUNTIME_ERR("RestrictedLogSoftmax currently only supports single column expressions (contributions expanding support to multiple columns welcome!)");
  const real logz = logsumexp(x, denom);
  TensorTools::constant(fx, -numeric_limits<real>::infinity());
  for (auto i : denom)
    (mat(fx))(i,0) = x(i,0) - logz;
  if (denom.size() == 1) (mat(fx))(denom.front(), 0) = 0;
#endif
}

template<class MyDevice>
void RestrictedLogSoftmax::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i == 0, "Failed dimension check in RestrictedLogSoftmax");
#ifdef __CUDACC__
  DYNET_NO_CUDA_IMPL_ERROR("RestrictedLogSoftmax backward");
#else
  float z = 0;
  for (auto ind : denom)
    z += (mat(dEdf))(ind, 0);
  for (auto ind : denom)
    (mat(dEdxi))(ind, 0) += (mat(dEdf))(ind, 0) - expf((mat(fx))(ind, 0)) * z;
#endif
}
DYNET_NODE_INST_DEV_IMPL(RestrictedLogSoftmax)

// ************* Sparsemax *************

#define MAX_SPARSEMAX_LOSS_ROWS 65536

#ifndef __CUDACC__

string Sparsemax::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "sparsemax(" << arg_names[0] << ")";
  return s.str();
}

Dim Sparsemax::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1 && LooksLikeVector(xs[0]), "Bad input dimensions in Sparsemax: " << xs);
  return xs[0];
}

size_t Sparsemax::aux_storage_size() const {
  return (dim.size() + 1) * sizeof(float);
}

#endif

template<class MyDevice>
void Sparsemax::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  if (xs[0]->d.cols() == 1) {
#ifdef __CUDACC__
    DYNET_NO_CUDA_IMPL_ERROR("Sparsemax forward");
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
    auto y = mat(fx);
    tvec(fx) = (tvec(*xs[0]) - tau).cwiseMax(0.f);
    int c = 1;
    int *cc = static_cast<int*>(aux_mem);
    for (unsigned i = 0; i < rows; ++i)
      if (y(i,0) > 0.f) cc[c++] = i;
    cc[0] = c - 1;
#endif
  } else {
    DYNET_RUNTIME_ERR("Sparsemax not yet implemented for multiple columns");
  }
}

template<class MyDevice>
void Sparsemax::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
#ifdef __CUDACC__
  DYNET_NO_CUDA_IMPL_ERROR("Sparsemax backward");
#else
  const int ssize = static_cast<int*>(aux_mem)[0];
  int *support = static_cast<int*>(aux_mem) + 1;
  float dhat = 0;
  auto& d = mat(dEdf);
  for (int i = 0; i < ssize; ++i)
    dhat += d(support[i], 0);
  dhat /= ssize;
  for (int i = 0; i < ssize; ++i)
    (mat(dEdxi))(support[i], 0) += d(support[i], 0) - dhat;
#endif
}
DYNET_NODE_INST_DEV_IMPL(Sparsemax)

// ************* SparsemaxLoss *************

#ifndef __CUDACC__

string SparsemaxLoss::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "sparsemax(" << arg_names[0] << ", q)";
  return s.str();
}

Dim SparsemaxLoss::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1 && LooksLikeVector(xs[0]), "Bad input dimensions in SparsemaxLoss: " << xs);
  return Dim({1});
}

size_t SparsemaxLoss::aux_storage_size() const {
  // first dim.size dimensions is the sparsemax
  const unsigned rows = MAX_SPARSEMAX_LOSS_ROWS;  // this should be xs[0]->d.rows()
  return rows * sizeof(float);
}

#endif

template<class MyDevice>
void SparsemaxLoss::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  if (xs[0]->d.cols() == 1) {
#ifdef __CUDACC__
    DYNET_NO_CUDA_IMPL_ERROR("SparsemaxLoss forward");
#else
    const int rows = xs[0]->d.rows();
    if (rows > MAX_SPARSEMAX_LOSS_ROWS)
      DYNET_RUNTIME_ERR("MAX_SPARSEMAX_LOSS_ROWS is not sufficient. Recompile with larger value.");
    const unsigned qsupport_size = pq->size();
    const float qprop = 1.f / qsupport_size;

    float *zs = static_cast<float*>(aux_mem);
    std::partial_sort_copy(xs[0]->v, xs[0]->v+rows, zs, zs + rows, std::greater<float>());
    float sum = 0, maxsum = 0;
    int k = 0;
    for (k = 0; k < rows; ++k) {
      sum += zs[k];
      float t = 1 + (k + 1) * zs[k];
      if (t <= sum) break;
      maxsum = sum;
    }
    float tau = (maxsum - 1) / k;
    Tensor tsm(xs[0]->d, (float*)aux_mem, xs[0]->device, DeviceMempool::FXS);
    t<1>(tsm) = (t<1>(*xs[0]) - tau).cwiseMax(0.f);
    t<0>(fx) = ( (t<1>(tsm) != 0.f).cast<float>() * (t<1>(*xs[0]).square() - (tau * tau)) ).sum();
    t<0>(fx) = ( t<0>(fx) + qprop * qprop * qsupport_size ) / 2.f;
    for (unsigned i = 0; i < qsupport_size; ++i)
      t<0>(fx) = t<0>(fx) - t<1>(*xs[0]).chip<0>((*pq)[i]) * qprop;
    t<0>(fx) = t<0>(fx).cwiseMax(0.f);
#endif
  } else {
    DYNET_RUNTIME_ERR("SparsemaxLoss not yet implemented for multiple columns");
  }
}

template<class MyDevice>
void SparsemaxLoss::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
#ifdef __CUDACC__
  DYNET_NO_CUDA_IMPL_ERROR("SparsemaxLoss backward");
#else
  const float d = dEdf.v[0];
  float* psm = static_cast<float*>(aux_mem);
  float dqprop = d / pq->size();
  Tensor tsm(xs[0]->d, psm, xs[0]->device, DeviceMempool::FXS);
  auto sm = mat(tsm);  // sparsemax(z)
  mat(dEdxi) += sm * d;
  for (unsigned i = 0; i < pq->size(); ++i)
    (mat(dEdxi))((*pq)[i], 0) -= dqprop;
#endif
}
DYNET_NODE_INST_DEV_IMPL(SparsemaxLoss)

// ************* Constrained softmax *************

#ifndef __CUDACC__

string ConstrainedSoftmax::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "constrained_softmax(" << arg_names[0] << ")";
  return s.str();
}

Dim ConstrainedSoftmax::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 2 && LooksLikeVector(xs[0]) &&
                  LooksLikeVector(xs[1]),
                  "Bad input dimensions in ConstrainedSoftmax: " << xs);
  return xs[0];
}

size_t ConstrainedSoftmax::aux_storage_size() const {
  // Need some extra allocation for the forward step.
  const unsigned rows = dim.size();
  return rows * sizeof(float) + rows * sizeof(int);
}

#endif

template<class MyDevice>
void ConstrainedSoftmax::forward_dev_impl(const MyDevice & dev,
                                          const vector<const Tensor*>& xs,
                                          Tensor& fx) const {
  // The first input contains the log-potentials z.
  // The second input contains the upper bound constraints u.
  if (xs[0]->d.cols() == 1 && xs[1]->d.cols() == 1) {
#ifdef __CUDACC__
    DYNET_NO_CUDA_IMPL_ERROR("ConstrainedSoftmax forward");
#else
    // Total allocated memory is rows*sizeof(float) + rows*sizeof(int).
    float max;
    const unsigned rows = xs[0]->d.rows();
    for (unsigned k = 0; k < rows; ++k) {
      if (k == 0 || xs[0]->v[k] > max) max =  xs[0]->v[k];
    }
    float *q = static_cast<float*>(aux_mem);
    for (unsigned k = 0; k < rows; ++k) {
      q[k] = exp(xs[0]->v[k] - max);
    }
    float *u = xs[1]->v;
    float mass = 0.0;
    int *indices = reinterpret_cast<int*>(static_cast<char*>(aux_mem) +
                                          rows*sizeof(float));
    for (unsigned k = 0; k < rows; ++k) {
      indices[k] = k;
    }
    unsigned num_active = rows;
    auto p = mat(fx);
    tvec(fx) = tvec(*xs[0]); // Initialize the vector with the right size.
    bool found = true;
    while (found) {
      float sum = 0.0;
      for (unsigned k = 0; k < num_active; ++k) {
        sum += q[indices[k]];
      }
      for (unsigned k = 0; k < num_active; ++k) {
        int i = indices[k];
        p(i, 0) = q[i] * (1.0 - mass) / sum;
      }
      found = false;
      unsigned j = 0;
      for (unsigned k = 0; k < num_active; ++k) {
        int i = indices[k];
        if (p(i, 0) > u[i]) {
          p(i, 0) = u[i];
          mass += u[i];
          found = true;
        } else {
          indices[j] = i;
          ++j;
        }
      }
      num_active = j;
    }
    // Write a 0/1 at each position to indicate if the variable is active.
    int *cc = static_cast<int*>(aux_mem);
    for (unsigned k = 0; k < rows; ++k) {
      cc[k] = 0;
    }
    for (unsigned k = 0; k < num_active; ++k) {
      cc[indices[k]] = 1;
    }
    float *m = reinterpret_cast<float*>(static_cast<char*>(aux_mem) +
                                        rows*sizeof(int));
    *m = mass;
#endif
  } else {
    DYNET_RUNTIME_ERR("ConstrainedSoftmax not yet implemented for multiple columns");
  }
}

template<class MyDevice>
void ConstrainedSoftmax::backward_dev_impl(const MyDevice & dev,
                                           const vector<const Tensor*>& xs,
                                           const Tensor& fx,
                                           const Tensor& dEdf,
                                           unsigned i,
                                           Tensor& dEdxi) const {
  assert(i < xs.size());
#ifdef __CUDACC__
  DYNET_NO_CUDA_IMPL_ERROR("ConstrainedSoftmax backward");
#else
  const unsigned rows = xs[0]->d.rows();
  int *active = static_cast<int*>(aux_mem);
  float mass =  reinterpret_cast<float*>(static_cast<char*>(aux_mem) +
                                         rows*sizeof(int))[0];
  float dhat = 0;
  auto& d = mat(dEdf);
  auto& p = mat(fx);
  for (unsigned k = 0; k < rows; ++k) {
    if (active[k]) dhat += p(k, 0) * d(k, 0);
  }
  dhat /= (1 - mass);
  for (unsigned k = 0; k < rows; ++k) {
    if (active[k]) {
      if (i == 0) {
        // Gradient wrt log-potentials z.
        mat(dEdxi)(k, 0) +=  p(k, 0) * (d(k, 0) - dhat);
      }
    } else {
      if (i == 1) {
        // Gradient wrt upper bound constraints u.
        mat(dEdxi)(k, 0) +=  d(k, 0) - dhat;
      }
    }
  }
#endif
}
DYNET_NODE_INST_DEV_IMPL(ConstrainedSoftmax)

}
