#include "dynet/nodes-softmaxes.h"

#include "dynet/nodes-macros.h"
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

size_t Softmax::aux_storage_size() const {
  return 2 * dim.size() / dim.rows() * sizeof(float);
}

#endif

template<class MyDevice>
void Softmax::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed dimension check in Softmax::forward");
  Tensor z(Dim({xs[0]->d.cols()},fx.d.bd), (float*)aux_mem, fx.device, DeviceMempool::FXS);
  Tensor m(Dim({xs[0]->d.cols()},fx.d.bd), (float*)aux_mem + z.d.size(), fx.device, DeviceMempool::FXS);
  TensorTools::logsumexp_dev(dev, *xs[0], m, z);
  // TODO? Is this broadcast efficient on CPU?
  Eigen::array<int, 3> bcasts = {(int)xs[0]->d.rows(), 1, 1};
  Eigen::array<int, 3> morph = {1, (int)z.d[0], (int)z.d.bd};
  fx.tb<2>().device(*dev.edevice) = (xs[0]->tb<2>() - z.tvec().reshape(morph).broadcast(bcasts)).exp();
}

template<class MyDevice>
void Softmax::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  Tensor z(Dim({fx.d.cols()},fx.d.bd), (float*)aux_mem, fx.device, DeviceMempool::FXS);
  // TODO? Is this broadcast efficient on CPU?
  Eigen::array<int, 1> red_axis = {0};
  z.tb<1>().device(*dev.edevice) = (fx.tb<2>() * dEdf.tb<2>()).sum(red_axis);
  Eigen::array<int, 3> bcast = {(int)xs[0]->d.rows(), 1, 1};
  Eigen::array<int, 3> morph = {1, (int)z.d[0], (int)z.d.bd};
  dEdxi.tb<2>().device(*dev.edevice) += (dEdf.tb<2>() - z.tvec().reshape(morph).broadcast(bcast)) * fx.tb<2>();
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
    fx.t<1>().device(*dev.edevice) = xs[0]->t<1>() - z.t<1>().broadcast(bcast);
#else
    fx.t<1>().device(*dev.edevice) = xs[0]->t<1>() - as_scalar(z);
#endif
  } else {
    // TODO? Is this broadcast efficient on CPU?
    Eigen::array<int, 3> bcasts = {(int)xs[0]->d.rows(), 1, 1};
    Eigen::array<int, 3> morph = {1, (int)z.d[0], (int)z.d.bd};
    fx.tb<2>().device(*dev.edevice) = xs[0]->tb<2>() - z.tvec().reshape(morph).broadcast(bcasts);
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
  // TODO? Is this broadcast efficient on CPU?
  Eigen::array<int, 1> red_axis; red_axis[0] = 0;
  z.tb<1>().device(*dev.edevice) = dEdf.tb<2>().sum(red_axis);
  Eigen::array<int, 3> bcast = {(int)fx.d.rows(), 1, 1};
  Eigen::array<int, 3> morph = {1, (int)z.d[0], (int)z.d.bd};
  dEdxi.tb<2>().device(*dev.edevice) += fx.tb<2>().exp() * -z.tvec().reshape(morph).broadcast(bcast) + dEdf.tb<2>();
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
  auto x = **xs[0];
  if(denom.size() == 0)
    DYNET_RUNTIME_ERR("RestrictedLogSoftmax currently only supports single column expressions (contributions expanding support to multiple columns welcome!)");
  const real logz = logsumexp(x, denom);
  TensorTools::constant(fx, -numeric_limits<real>::infinity());
  for (auto i : denom)
    (*fx)(i,0) = x(i,0) - logz;
  if (denom.size() == 1) (*fx)(denom.front(), 0) = 0;
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
    z += (*dEdf)(ind, 0);
  for (auto ind : denom)
    (*dEdxi)(ind, 0) += (*dEdf)(ind, 0) - expf((*fx)(ind, 0)) * z;
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
    auto y = *fx;
    fx.tvec() = (xs[0]->tvec() - tau).cwiseMax(0.f);
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
  auto& d = *dEdf;
  for (int i = 0; i < ssize; ++i)
    dhat += d(support[i], 0);
  dhat /= ssize;
  for (int i = 0; i < ssize; ++i)
    (*dEdxi)(support[i], 0) += d(support[i], 0) - dhat;
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
    tsm.t<1>() = (xs[0]->t<1>() - tau).cwiseMax(0.f);
    fx.t<0>() = ( (tsm.t<1>() != 0.f).cast<float>() * (xs[0]->t<1>().square() - (tau * tau)) ).sum();
    fx.t<0>() = ( fx.t<0>() + qprop * qprop * qsupport_size ) / 2.f;
    for (unsigned i = 0; i < qsupport_size; ++i)
      fx.t<0>() = fx.t<0>() - xs[0]->t<1>().chip<0>((*pq)[i]) * qprop;
    fx.t<0>() = fx.t<0>().cwiseMax(0.f);
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
  auto sm = *tsm;  // sparsemax(z)
  *dEdxi += sm * d;
  for (unsigned i = 0; i < pq->size(); ++i)
    (*dEdxi)((*pq)[i], 0) -= dqprop;
#endif
}
DYNET_NODE_INST_DEV_IMPL(SparsemaxLoss)

}
