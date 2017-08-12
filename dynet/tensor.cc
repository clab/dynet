#include "dynet/tensor.h"
#include "dynet/globals.h"
#include "dynet/except.h"

#include <random>
#include <vector>
#include <cstring>

#ifdef __CUDACC__
#include "dynet/gpu-ops.h"
#include "dynet/cuda.h"
#endif

using namespace std;

namespace dynet {

// ---- CPU only operations

#ifndef __CUDACC__

ostream& operator<<(ostream& os, const Tensor& t) {
  if (t.device->type == DeviceType::CPU) {
    os << (*t);
#if HAVE_CUDA
  } else if (t.device->type == DeviceType::GPU) {
    vector<real> vt = as_vector(t);
    Eigen::Map<Eigen::MatrixXf> m(&vt[0], t.d.rows(), t.d.cols());
    os << m;
#endif
  } else { throw std::runtime_error("Bad device type"); }
  return os;
}

real as_scalar(const Tensor& t) {
  if (t.d.size() != 1)
    throw std::runtime_error("Input tensor has more than one element, cannot convert to scalar.");
  real res = 0.;
  if (t.device->type == DeviceType::CPU) {
    return t.v[0];
#if HAVE_CUDA
  } else if (t.device->type == DeviceType::GPU) {
    CUDA_CHECK(cudaMemcpy(&res, t.v, sizeof(float), cudaMemcpyDeviceToHost));
    return res;
#endif
  } else { throw std::runtime_error("Bad device type"); }
  return res;
}

vector<real> as_vector(const Tensor& v) {
  vector<real> res(v.d.size());
  if (v.device->type == DeviceType::CPU) {
    memcpy(&res[0], v.v, sizeof(real) * res.size());
  } else if (v.device->type == DeviceType::GPU) {
#if HAVE_CUDA
    CUDA_CHECK(cudaMemcpy(&res[0], v.v, sizeof(real) * res.size(), cudaMemcpyDeviceToHost));
#endif
  } else { throw std::runtime_error("Bad device type"); }
  return res;
}

vector<Eigen::DenseIndex> as_vector(const IndexTensor& v) {
  vector<Eigen::DenseIndex> res(v.d.size());
  if (v.device->type == DeviceType::CPU) {
    memcpy(&res[0], v.v, sizeof(Eigen::DenseIndex) * res.size());
#if HAVE_CUDA
  } else if (v.device->type == DeviceType::GPU) {
    CUDA_CHECK(cudaMemcpy(&res[0], v.v, sizeof(Eigen::DenseIndex) * res.size(), cudaMemcpyDeviceToHost));
#endif
  } else { throw std::runtime_error("Bad device type"); }
  return res;
}

float TensorTools::access_element(const Tensor& v, int index) {
  float ret = 0.;
  if (v.device->type == DeviceType::CPU) {
    return v.v[index];
#if HAVE_CUDA
  } else if (v.device->type == DeviceType::GPU) {
    cudaMemcpy(&ret, &v.v[index], sizeof(real), cudaMemcpyDeviceToHost);
    return ret;
#endif
  } else { throw std::runtime_error("Bad device type"); }
  return ret;
}

float TensorTools::access_element(const Tensor& v, const Dim& index) {
  if (v.device->type == DeviceType::CPU) {
    return (*v)(index[0], index[1]);
#if HAVE_CUDA
  } else if (v.device->type == DeviceType::GPU) {
    DYNET_NO_CUDA_IMPL_ERROR("TensorTools::access_element(Tensor,Dim)")
#endif
  } else { throw std::runtime_error("Bad device type"); }
  return 0;
}

void TensorTools::set_element(const Tensor& v, int index, float value) {
  if (v.device->type == DeviceType::CPU) {
    v.v[index] = value;
#if HAVE_CUDA
  } else if (v.device->type == DeviceType::GPU) {
    cudaMemcpyAsync(&v.v[index], &value, sizeof(real), cudaMemcpyHostToDevice);
#endif
  } else { throw std::runtime_error("Bad device type"); }
}

void TensorTools::copy_element(const Tensor& l, int lindex, Tensor& r, int rindex) {
  if (l.device->type == DeviceType::CPU) {
    r.v[rindex] = l.v[lindex];
  } else {
    if (l.device != r.device) {
      throw std::invalid_argument("TensorTools::CopyElement doesn't support inter-device copy yet");
#if HAVE_CUDA
    } else if (l.device->type == DeviceType::GPU) {
      cudaMemcpyAsync(&r.v[rindex], &l.v[lindex], sizeof(real), cudaMemcpyDeviceToDevice);
#endif
    } else { throw std::runtime_error("Bad device type"); }
  }
}

void TensorTools::set_elements(const Tensor& v, const vector<float>& vec) {
  if (v.device->type == DeviceType::CPU) {
    memcpy(v.v, &vec[0], sizeof(real) * vec.size());
#if HAVE_CUDA
  } else if (v.device->type == DeviceType::GPU) {
    cudaMemcpyAsync(v.v, &vec[0], sizeof(real) * vec.size(), cudaMemcpyHostToDevice);
#endif
  } else { throw std::runtime_error("Bad device type"); }
}

void TensorTools::copy_elements(Tensor& v, const Tensor& v_src) {
  if (v.device->type == DeviceType::CPU) {
    if (v_src.device->type == DeviceType::CPU) {
      memcpy(v.v, v_src.v, sizeof(real) * v.d.size());
#if HAVE_CUDA
    } else if (v_src.device->type == DeviceType::GPU) {
      cudaMemcpyAsync(v.v, v_src.v, sizeof(real) * v.d.size(), cudaMemcpyDeviceToHost);
#endif
    } else { throw std::runtime_error("Bad device type"); }
#if HAVE_CUDA
  } else if (v.device->type == DeviceType::GPU) {
    if (v_src.device->type == DeviceType::CPU) {
      cudaMemcpyAsync(v.v, v_src.v, sizeof(real) * v.d.size(), cudaMemcpyHostToDevice);
    } else {
      cudaMemcpyAsync(v.v, v_src.v, sizeof(real) * v.d.size(), cudaMemcpyDeviceToDevice);
    }
#endif
  } else { throw std::runtime_error("Bad device type"); }
}

void TensorTools::zero(Tensor& d) {
  constant(d, 0);
}

void TensorTools::identity(Tensor& val) {
  if (val.d.nd != 2 || val.d[0] != val.d[1])
    throw std::runtime_error("Attempt to set a tensor that is not a square matrix to identity");
  size_t pos = 0;
  if (val.device->type == DeviceType::CPU) {
    for (size_t i = 0; i < val.d[0]; ++i)
      for (size_t j = 0; j < val.d[1]; ++j)
        val.v[pos++] = (i == j ? 1 : 0);
#if HAVE_CUDA
  } else if (val.device->type == DeviceType::GPU) {
    float* t = new float[val.d.size()];
    for (size_t i = 0; i < val.d[0]; ++i)
      for (size_t j = 0; j < val.d[1]; ++j)
        t[pos++] = (i == j ? 1 : 0);
    CUDA_CHECK(cudaMemcpy(val.v, t, sizeof(real) * val.d.size(), cudaMemcpyHostToDevice));
    delete[] t;
#endif
  } else { throw std::runtime_error("Bad device type"); }
}

void TensorTools::randomize_bernoulli(Tensor& val, real p, real scale) {
  bernoulli_distribution distribution(p);
  auto b = [&] {return distribution(*rndeng) * scale;};
  if (val.device->type == DeviceType::CPU) {
    generate(val.v, val.v + val.d.size(), b);
#if HAVE_CUDA
  } else if (val.device->type == DeviceType::GPU) {
    float* t = new float[val.d.size()];
    generate(t, t + val.d.size(), b);
    CUDA_CHECK(cudaMemcpy(val.v, t, sizeof(real) * val.d.size(), cudaMemcpyHostToDevice));
    delete[] t;
#endif
  } else { throw std::runtime_error("Bad device type"); }
}

void TensorTools::randomize_normal(Tensor& val, real mean, real stddev) {
  normal_distribution<real> distribution(mean, stddev);
  auto b = [&] {return distribution(*rndeng);};
  if (val.device->type == DeviceType::CPU) {
    generate(val.v, val.v + val.d.size(), b);
#if HAVE_CUDA
  } else if (val.device->type == DeviceType::GPU) {
    float* t = new float[val.d.size()];
    generate(t, t + val.d.size(), b);
    CUDA_CHECK(cudaMemcpy(val.v, t, sizeof(real) * val.d.size(), cudaMemcpyHostToDevice));
    delete[] t;
#endif
  } else { throw std::runtime_error("Bad device type"); }
}

void TensorTools::randomize_uniform(Tensor& val, real left, real right) {
  uniform_real_distribution<real> distribution(left, right);
  auto b = [&] {return distribution(*rndeng);};
  if (val.device->type == DeviceType::CPU) {
    generate(val.v, val.v + val.d.size(), b);
#if HAVE_CUDA
  } else if (val.device->type == DeviceType::GPU) {
    float* t = new float[val.d.size()];
    generate(t, t + val.d.size(), b);
    CUDA_CHECK(cudaMemcpy(val.v, t, sizeof(real) * val.d.size(), cudaMemcpyHostToDevice));
    delete[] t;
#endif
  } else { throw std::runtime_error("Bad device type"); }
}

void TensorTools::randomize_orthonormal(Tensor& val, real scale) {
  if (val.d.nd != 2 || val.d[0] != val.d[1])
    throw std::runtime_error("Attempt to set a tensor that is not a square matrix to an orthogonal matrix");
  if (val.device->type == DeviceType::CPU) {
    randomize_uniform(val, -1.0, 1.0);
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(*val, Eigen::ComputeFullU | Eigen::ComputeThinV);
    *val = scale * svd.matrixU();
#ifdef HAVE_CUDA
  } else if (val.device->type == DeviceType::GPU) {
    DYNET_NO_CUDA_IMPL_ERROR("Orthonormal initialization");
    // TODO: The following should work, but for some reason it isn't working
    // float* t = new float[val.d.size()];
    // Tensor tt(val);
    // tt.v = t;
    // randomize_uniform(tt, -1.0, 1.0);
    // Eigen::JacobiSVD<Eigen::MatrixXf> svd(*tt, Eigen::ComputeFullU | Eigen::ComputeThinV);
    // *tt = scale * svd.matrixU();
    // CUDA_CHECK(cudaMemcpy(val.v, tt.v, sizeof(real) * val.d.size(), cudaMemcpyHostToDevice));
    // delete[] t;
#endif
  } else { throw std::runtime_error("Bad device type"); }
}

real rand01() {
  uniform_real_distribution<real> distribution(0, 1);
  return distribution(*rndeng);
}

int rand0n(int n) {
  if (n <= 0) throw std::runtime_error("Integer upper bound is non-positive");
  int x = rand01() * n;
  while (n == x) { x = rand01() * n; }
  return x;
}

real rand_normal() {
  normal_distribution<real> distribution(0, 1);
  return distribution(*rndeng);
}

#endif

// ---- CPU/GPU operations
// TODO: would like to get rid of all the verbose code dispatching o the appropriate device
template <class MyDevice>
void TensorTools::accumulate_dev(const MyDevice & dev, Tensor& v, const Tensor& v_src) {
  DYNET_ASSERT(v.d.size() == v_src.d.size(), "TensorTools::accumulate can only be used with tensors of identical size");
  v.tvec().device(*dev.edevice) += v_src.tvec();
}
#ifdef __CUDACC__
template void TensorTools::accumulate_dev<Device_GPU>(const Device_GPU & dev, Tensor& v, const Tensor& v_src);
#else
template void TensorTools::accumulate_dev<Device_CPU>(const Device_CPU & dev, Tensor& v, const Tensor& v_src);
#ifdef HAVE_CUDA
extern template void TensorTools::accumulate_dev<Device_GPU>(const Device_GPU & dev, Tensor& v, const Tensor& v_src);
void TensorTools::accumulate(Tensor& v, const Tensor& v_src) {
  if (v.device->type == DeviceType::CPU) { return accumulate_dev(*(const Device_CPU*)v.device, v, v_src); }
  else if (v.device->type == DeviceType::GPU) { return accumulate_dev(*(const Device_GPU*)v.device, v, v_src); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
void TensorTools::accumulate(Tensor& v, const Tensor& v_src) {
  if (v.device->type == DeviceType::CPU) { return accumulate_dev(*(const Device_CPU*)v.device, v, v_src); }
  else { throw std::runtime_error("Bad device type"); }
}
#endif
#endif

template <class MyDevice>
void TensorTools::constant_dev(const MyDevice & dev, Tensor& d, float c) {
  d.tvec().device(*dev.edevice) = d.tvec().constant(c);
}
#ifdef __CUDACC__
template void TensorTools::constant_dev<Device_GPU>(const Device_GPU & dev, Tensor& d, float c);
#else
template void TensorTools::constant_dev<Device_CPU>(const Device_CPU & dev, Tensor& d, float c);
#ifdef HAVE_CUDA
extern template void TensorTools::constant_dev<Device_GPU>(const Device_GPU & dev, Tensor& d, float c);
void TensorTools::constant(Tensor& d, float c) {
  if (d.device->type == DeviceType::CPU) { return constant_dev(*(const Device_CPU*)d.device, d, c); }
  else if (d.device->type == DeviceType::GPU) { return constant_dev(*(const Device_GPU*)d.device, d, c); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
void TensorTools::constant(Tensor& d, float c) {
  if (d.device->type == DeviceType::CPU) { return constant_dev(*(const Device_CPU*)d.device, d, c); }
  else { throw std::runtime_error("Bad device type"); }
}
#endif
#endif

template <class MyDevice>
void TensorTools::clip_dev(const MyDevice & dev, Tensor& d, float left, float right) {
  d.tvec().device(*dev.edevice) = d.tvec().cwiseMax(left).cwiseMin(right);
}
#ifdef __CUDACC__
template void TensorTools::clip_dev<Device_GPU>(const Device_GPU & dev, Tensor& d, float left, float right);
#else
template void TensorTools::clip_dev<Device_CPU>(const Device_CPU & dev, Tensor& d, float left, float right);
#ifdef HAVE_CUDA
extern template void TensorTools::clip_dev<Device_GPU>(const Device_GPU & dev, Tensor& d, float left, float right);
void TensorTools::clip(Tensor& d, float left, float right) {
  if (d.device->type == DeviceType::CPU) { return clip_dev(*(const Device_CPU*)d.device, d, left, right); }
  else if (d.device->type == DeviceType::GPU) { return clip_dev(*(const Device_GPU*)d.device, d, left, right); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
void TensorTools::clip(Tensor& d, float left, float right) {
  if (d.device->type == DeviceType::CPU) { return clip_dev(*(const Device_CPU*)d.device, d, left, right); }
  else { throw std::runtime_error("Bad device type"); }
}
#endif
#endif

template <class MyDevice>
void TensorTools::logsumexp_dev(const MyDevice & dev, const Tensor& x, Tensor & m, Tensor& z) {
  if(x.d.bd == 1 && x.d[1] == 1) {
    m.t<0>().device(*dev.edevice) = x.t<1>().maximum();
#ifdef __CUDACC__
    Eigen::array<int, 1> bcast;
    bcast[0] = x.d[0];
    // This needs to be split into two lines to prevent memory allocation
    // TODO? Here and in logsoftmax: Is there a better way to subtract a scalar that is already on the GPU without using broadcasting (and without copying the scalar back to the host first)
    z.t<0>().device(*dev.edevice) = (x.t<1>() - m.t<1>().broadcast(bcast)).exp().sum();
    z.t<0>().device(*dev.edevice) = z.t<0>().log() + m.t<0>();
#else
    float mval = as_scalar(m);
    // This needs to be split into two lines to prevent memory allocation
    z.t<0>().device(*dev.edevice) = (x.t<1>() - mval).exp().sum();
    z.t<0>().device(*dev.edevice) = z.t<0>().log() + mval;
#endif
  } else {
    Eigen::array<int, 1> red_axis; red_axis[0] = 0;
    m.tb<1>().device(*dev.edevice) = x.tb<2>().maximum(red_axis);
    // TODO: Currently, the first version is slower on CPU, hence the switch
#ifdef __CUDACC__
    Eigen::array<int, 3> bcast({(int)x.d.rows(), 1, 1});
    Eigen::array<int, 3> morph({1, (int)m.d[0], (int)m.d.bd});
    // This needs to be split into two lines to prevent memory allocation
    z.tb<1>().device(*dev.edevice) = (x.tb<2>() - m.tb<2>().reshape(morph).broadcast(bcast)).exp().sum(red_axis);
    z.tb<1>().device(*dev.edevice) = z.tb<1>().log() + m.tb<1>();
#else
    auto miter = m.v;
    for(size_t b = 0; b < x.d.bd; ++b) {
      for(size_t i = 0; i < x.d[1]; ++i, ++miter) {
        z.tb<1>().chip<1>(b).chip<0>(i).device(*dev.edevice) = (x.tb<2>().chip<2>(b).chip<1>(i) - *miter).exp().sum();
        z.tb<1>().chip<1>(b).chip<0>(i).device(*dev.edevice) = z.tb<1>().chip<1>(b).chip<0>(i).log() + *miter;
      }
    }
#endif
  }
}
#ifdef __CUDACC__
template void TensorTools::logsumexp_dev<Device_GPU>(const Device_GPU & dev, const Tensor &x, Tensor &m, Tensor &z);
#else
template void TensorTools::logsumexp_dev<Device_CPU>(const Device_CPU & dev, const Tensor &x, Tensor &m, Tensor &z);
#ifdef HAVE_CUDA
extern template void TensorTools::logsumexp_dev<Device_GPU>(const Device_GPU & dev, const Tensor &x, Tensor &m, Tensor &z);
void TensorTools::logsumexp(const Tensor &x, Tensor &m, Tensor &z) {
  if (x.device->type == DeviceType::CPU) { return logsumexp_dev(*(const Device_CPU*)x.device, x, m, z); }
  else if (x.device->type == DeviceType::GPU) { return logsumexp_dev(*(const Device_GPU*)x.device, x, m, z); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
void TensorTools::logsumexp(const Tensor &x, Tensor &m, Tensor &z) {
  if (x.device->type == DeviceType::CPU) { return logsumexp_dev(*(const Device_CPU*)x.device, x, m, z); }
  else { throw std::runtime_error("Bad device type"); }
}
#endif
#endif

template <class MyDevice>
IndexTensor TensorTools::argmax_dev(const MyDevice & dev, const Tensor& v, unsigned dim, unsigned num) {
  if(num > 1)
    DYNET_RUNTIME_ERR("Currently do not support num > 1 in argmax");
  DYNET_ARG_CHECK(v.mem_pool != DeviceMempool::NONE, "Input Tensor to TensorTools::argmax must be associated with a memory pool.");
  Dim ids_dim = v.d; ids_dim.d[dim] = num;
  IndexTensor ids(ids_dim, nullptr, v.device, v.mem_pool);
  AlignedMemoryPool* pool = v.device->pools[(size_t)v.mem_pool];
  ids.v = static_cast<Eigen::DenseIndex*>(pool->allocate(ids_dim.size() * sizeof(Eigen::DenseIndex)));
  ids.tb<3>().device(*dev.edevice) = v.tb<4>().argmax(dim);
  return ids;
}
#ifdef __CUDACC__
template IndexTensor TensorTools::argmax_dev<Device_GPU>(const Device_GPU & dev, const Tensor& d, unsigned dim, unsigned num);
#else
template IndexTensor TensorTools::argmax_dev<Device_CPU>(const Device_CPU & dev, const Tensor& d, unsigned dim, unsigned num);
#ifdef HAVE_CUDA
extern template IndexTensor TensorTools::argmax_dev<Device_GPU>(const Device_GPU & dev, const Tensor& d, unsigned dim, unsigned num);
IndexTensor TensorTools::argmax(const Tensor& d, unsigned dim, unsigned num) {
  if (d.device->type == DeviceType::CPU) { return argmax_dev(*(const Device_CPU*)d.device, d, dim, num); }
  else if (d.device->type == DeviceType::GPU) { return argmax_dev(*(const Device_GPU*)d.device, d, dim, num); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
IndexTensor TensorTools::argmax(const Tensor& d, unsigned dim, unsigned num) {
  if (d.device->type == DeviceType::CPU) { return argmax_dev(*(const Device_CPU*)d.device, d, dim, num); }
  else { throw std::runtime_error("Bad device type"); }
}
#endif
#endif

template <class MyDevice>
IndexTensor TensorTools::categorical_sample_log_prob_dev(const MyDevice & dev, const Tensor& v, unsigned dim, unsigned num) {
  if(num > 1)
    DYNET_RUNTIME_ERR("Currently do not support num > 1 in categorical_sample_log_prob");
  DYNET_ARG_CHECK(v.mem_pool != DeviceMempool::NONE, "Input Tensor to TensorTools::argmax must be associated with a memory pool.");
  Dim ids_dim = v.d; ids_dim.d[dim] = num;
  IndexTensor ids(ids_dim, nullptr, v.device, v.mem_pool);
  AlignedMemoryPool* scratch_allocator = v.device->pools[(int)DeviceMempool::SCS];
  ids.v = static_cast<Eigen::DenseIndex*>(scratch_allocator->allocate(ids_dim.size() * sizeof(Eigen::DenseIndex)));
  Dim copy_dim = v.d; // TODO: make this match num to enable num
  Tensor copy(copy_dim, nullptr, v.device, v.mem_pool);
  copy.v = static_cast<float*>(scratch_allocator->allocate(v.d.size() * sizeof(float)));
  TensorTools::randomize_uniform(copy);
  ids.tb<3>().device(*dev.edevice) = (v.tb<4>() - (-copy.tb<4>().log()).log()).argmax(dim);
  scratch_allocator->free();
  return ids;
}
#ifdef __CUDACC__
template IndexTensor TensorTools::categorical_sample_log_prob_dev<Device_GPU>(const Device_GPU & dev, const Tensor& d, unsigned dim, unsigned num);
#else
template IndexTensor TensorTools::categorical_sample_log_prob_dev<Device_CPU>(const Device_CPU & dev, const Tensor& d, unsigned dim, unsigned num);
#ifdef HAVE_CUDA
extern template IndexTensor TensorTools::categorical_sample_log_prob_dev<Device_GPU>(const Device_GPU & dev, const Tensor& d, unsigned dim, unsigned num);
IndexTensor TensorTools::categorical_sample_log_prob(const Tensor& d, unsigned dim, unsigned num) {
  if (d.device->type == DeviceType::CPU) { return categorical_sample_log_prob_dev(*(const Device_CPU*)d.device, d, dim, num); }
  else if (d.device->type == DeviceType::GPU) { return categorical_sample_log_prob_dev(*(const Device_GPU*)d.device, d, dim, num); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
IndexTensor TensorTools::categorical_sample_log_prob(const Tensor& d, unsigned dim, unsigned num) {
  if (d.device->type == DeviceType::CPU) { return categorical_sample_log_prob_dev(*(const Device_CPU*)d.device, d, dim, num); }
  else { throw std::runtime_error("Bad device type"); }
}
#endif
#endif

} // namespace dynet

