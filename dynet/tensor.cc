
#include "dynet/tensor.h"
#include "dynet/tensor-eigen.h"
#include "dynet/index-tensor.h"
#include "dynet/globals.h"
#include "dynet/except.h"
#include "dynet/devices.h"

#include <random>
#include <vector>
#include <cstring>
#include <algorithm>

#ifdef __CUDACC__
#include "dynet/gpu-ops.h"
#include "dynet/cuda.h"
#endif

using namespace std;

namespace dynet {

// ---- CPU only operations

#ifndef __CUDACC__

bool Tensor::is_valid() const {
  // TODO : replace this with a custom exception
  if (device->type == DeviceType::CPU) {
    const size_t s = d.size();
    for (unsigned i = 0; i < s; ++i)
      if (std::isnan(v[i]) || std::isinf(v[i])) return false;
    return true;
  } else {
#if HAVE_CUDA
    if (device->type == DeviceType::GPU) {
      DYNET_NO_CUDA_IMPL_ERROR("is_valid()");
    }
#endif
  }
  return false;
}

ostream& operator<<(ostream& os, const Tensor& t) {
  if (t.device->type == DeviceType::CPU) {
    os << mat(t);
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
    CUDA_CHECK(cudaSetDevice(((Device_GPU*)t.device)->cuda_device_id));
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
    CUDA_CHECK(cudaSetDevice(((Device_GPU*)v.device)->cuda_device_id));
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
    CUDA_CHECK(cudaSetDevice(((Device_GPU*)v.device)->cuda_device_id));
    CUDA_CHECK(cudaMemcpy(&res[0], v.v, sizeof(Eigen::DenseIndex) * res.size(), cudaMemcpyDeviceToHost));
#endif
  } else { throw std::runtime_error("Bad device type"); }
  return res;
}

vector<real> as_scale_vector(const Tensor& v, float a) {
  vector<real> res(v.d.size());
  if (v.device->type == DeviceType::CPU) {
    memcpy(&res[0], v.v, sizeof(real) * res.size());
  } else if (v.device->type == DeviceType::GPU) {
#if HAVE_CUDA
    CUDA_CHECK(cudaSetDevice(((Device_GPU*)v.device)->cuda_device_id));
    CUDA_CHECK(cudaMemcpy(&res[0], v.v, sizeof(real) * res.size(), cudaMemcpyDeviceToHost));
#endif
  } else { throw std::runtime_error("Bad device type"); }
  if (a != 1.) std::transform(res.begin(), res.end(), res.begin(), [&](real t){ return t * a; });
  return res;
}

float TensorTools::access_element(const Tensor& v, int index) {
  float ret = 0.;
  if (v.device->type == DeviceType::CPU) {
    return v.v[index];
#if HAVE_CUDA
  } else if (v.device->type == DeviceType::GPU) {
    CUDA_CHECK(cudaSetDevice(((Device_GPU*)v.device)->cuda_device_id));
    cudaMemcpy(&ret, &v.v[index], sizeof(real), cudaMemcpyDeviceToHost);
    return ret;
#endif
  } else { throw std::runtime_error("Bad device type"); }
  return ret;
}

float TensorTools::access_element(const Tensor& v, const Dim& index) {
  if (v.device->type == DeviceType::CPU) {
    return mat(v)(index[0], index[1]);
#if HAVE_CUDA
  if (v.device->type == DeviceType::GPU) {
    float ret = 0.0f;
    CUDA_CHECK(cudaMemcpy(&ret, v.v + (v.d.rows() * index[0] + index[1]), sizeof(float), cudaMemcpyDeviceToHost));
    return ret;
  }
#endif
  } else { throw std::runtime_error("Bad device type"); }
  return 0;
}

void TensorTools::set_element(const Tensor& v, int index, float value) {
  if (v.device->type == DeviceType::CPU) {
    v.v[index] = value;
#if HAVE_CUDA
  } else if (v.device->type == DeviceType::GPU) {
    CUDA_CHECK(cudaSetDevice(((Device_GPU*)v.device)->cuda_device_id));
    cudaMemcpyAsync(&v.v[index], &value, sizeof(real), cudaMemcpyHostToDevice);
#endif
  } else { throw std::runtime_error("Bad device type"); }
}

void TensorTools::copy_element(const Tensor& l, int lindex, Tensor& r, int rindex) {
  if (l.device->type == DeviceType::CPU) {
    if (r.device->type == DeviceType::CPU) {
      r.v[rindex] = l.v[lindex];
#if HAVE_CUDA
    } else if (r.device->type == DeviceType::GPU) {
      CUDA_CHECK(cudaSetDevice(((Device_GPU*)r.device)->cuda_device_id));
      cudaMemcpyAsync(&r.v[rindex], &l.v[lindex], sizeof(real),
                      cudaMemcpyHostToDevice);
#endif
    } else { throw std::runtime_error("Bad device type"); }
#if HAVE_CUDA
  } else if (l.device->type == DeviceType::GPU) {
    if (r.device->type == DeviceType::CPU) {
      CUDA_CHECK(cudaSetDevice(((Device_GPU*)l.device)->cuda_device_id));
      cudaMemcpyAsync(&r.v[rindex], &l.v[lindex], sizeof(real),
                      cudaMemcpyDeviceToHost);
    } else if (r.device->type == DeviceType::GPU) {
      if (l.device == r.device) {
        cudaMemcpyAsync(&r.v[rindex], &l.v[lindex], sizeof(real), cudaMemcpyDeviceToDevice);
      } else {
        cudaMemcpyPeerAsync(&r.v[rindex], ((Device_GPU*)r.device)->cuda_device_id,
                            &l.v[lindex], ((Device_GPU*)l.device)->cuda_device_id,
                            sizeof(real));
      }
    } else { throw std::runtime_error("Bad device type"); }
#endif
  } else { throw std::runtime_error("Bad device type"); }
}

void TensorTools::set_elements(const Tensor& v, const vector<float>& vec) {
  if (v.device->type == DeviceType::CPU) {
    memcpy(v.v, &vec[0], sizeof(real) * vec.size());
#if HAVE_CUDA
  } else if (v.device->type == DeviceType::GPU) {
    CUDA_CHECK(cudaSetDevice(((Device_GPU*)v.device)->cuda_device_id));
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
      CUDA_CHECK(cudaSetDevice(((Device_GPU*)v_src.device)->cuda_device_id));
      cudaMemcpyAsync(v.v, v_src.v, sizeof(real) * v.d.size(), cudaMemcpyDeviceToHost);
#endif
    } else { throw std::runtime_error("Bad device type"); }
#if HAVE_CUDA
  } else if (v.device->type == DeviceType::GPU) {
    if (v_src.device->type == DeviceType::CPU) {
      CUDA_CHECK(cudaSetDevice(((Device_GPU*)v.device)->cuda_device_id));
      cudaMemcpyAsync(v.v, v_src.v, sizeof(real) * v.d.size(), cudaMemcpyHostToDevice);
    } else {
      if (v.device == v_src.device) {
        CUDA_CHECK(cudaSetDevice(((Device_GPU*)v.device)->cuda_device_id));
        cudaMemcpyAsync(v.v, v_src.v, sizeof(real) * v.d.size(), cudaMemcpyDeviceToDevice);
      } else {
        cudaMemcpyPeerAsync(v.v, ((Device_GPU*)v.device)->cuda_device_id,
                            v_src.v, ((Device_GPU*)v_src.device)->cuda_device_id,
                            sizeof(real) * v.d.size());
      }
    }
#endif
  } else { throw std::runtime_error("Bad device type"); }
}

void TensorTools::zero(Tensor& d) {
#if HAVE_CUDA
  if (d.device->type == DeviceType::GPU) {
    CUDA_CHECK(cudaSetDevice(((Device_GPU*)d.device)->cuda_device_id));
  }
#endif
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
    CUDA_CHECK(cudaSetDevice(((Device_GPU*)val.device)->cuda_device_id));
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
    CURAND_CHECK(curandGenerateUniform(curandeng, val.v, val.d.size()));
    TensorTools::uniform_to_bernoulli_dev<Device_GPU>(*(Device_GPU*)val.device, val, p);
    if(scale != 1.0)
      TensorTools::scale_dev<Device_GPU>(*(Device_GPU*)val.device, val, scale, 0);
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
    CURAND_CHECK(curandGenerateNormal(curandeng, val.v, val.d.size(), mean, stddev));
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
    CURAND_CHECK(curandGenerateUniform(curandeng, val.v, val.d.size()));
    if(left != 0 || right != 1)
      TensorTools::scale_dev<Device_GPU>(*(Device_GPU*)val.device, val, right-left, left);
#endif
  } else { throw std::runtime_error("Bad device type"); }
}

void TensorTools::randomize_orthonormal(Tensor& val, real scale) {
  if (val.d.nd != 2 || val.d[0] != val.d[1])
    throw std::runtime_error("Attempt to set a tensor that is not a square matrix to an orthogonal matrix");
  if (val.device->type == DeviceType::CPU) {
    randomize_uniform(val, -1.0, 1.0);
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(mat(val), Eigen::ComputeFullU | Eigen::ComputeThinV);
    mat(val) = scale * svd.matrixU();
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
    // CUDA_CHECK(cudaSetDevice(v.device->cuda_device_id));
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
  tvec(v).device(*dev.edevice) += tvec(v_src);
}
#ifdef __CUDACC__
template void TensorTools::accumulate_dev<Device_GPU>(const Device_GPU & dev, Tensor& v, const Tensor& v_src);
#else
template void TensorTools::accumulate_dev<Device_CPU>(const Device_CPU & dev, Tensor& v, const Tensor& v_src);
#ifdef HAVE_CUDA
extern template void TensorTools::accumulate_dev<Device_GPU>(const Device_GPU & dev, Tensor& v, const Tensor& v_src);
void TensorTools::accumulate(Tensor& v, const Tensor& v_src) {
  if (v.device->type == DeviceType::CPU) { return accumulate_dev(*(const Device_CPU*)v.device, v, v_src); }
  else if (v.device->type == DeviceType::GPU) {
    CUDA_CHECK(cudaSetDevice(((Device_GPU*)v.device)->cuda_device_id));
    return accumulate_dev(*(const Device_GPU*)v.device, v, v_src);
  } else { throw std::runtime_error("Bad device type"); }
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
  tvec(d).device(*dev.edevice) = tvec(d).constant(c);
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
  tvec(d).device(*dev.edevice) = tvec(d).cwiseMax(left).cwiseMin(right);
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
void TensorTools::scale_dev(const MyDevice & dev, Tensor& x, float a, float b) {
  tvec(x).device(*dev.edevice) = tvec(x) * a + b;
}
#ifdef __CUDACC__
template void TensorTools::scale_dev<Device_GPU>(const Device_GPU & dev, Tensor& d, float a, float b);
#else
template void TensorTools::scale_dev<Device_CPU>(const Device_CPU & dev, Tensor& d, float a, float b);
#ifdef HAVE_CUDA
extern template void TensorTools::scale_dev<Device_GPU>(const Device_GPU & dev, Tensor& d, float a, float b);
void TensorTools::scale(Tensor& d, float a, float b) {
  if (d.device->type == DeviceType::CPU) { return scale_dev(*(const Device_CPU*)d.device, d, a, b); }
  else if (d.device->type == DeviceType::GPU) { return scale_dev(*(const Device_GPU*)d.device, d, a, b); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
void TensorTools::scale(Tensor& d, float a, float b) {
  if (d.device->type == DeviceType::CPU) { return scale_dev(*(const Device_CPU*)d.device, d, a, b); }
  else { throw std::runtime_error("Bad device type"); }
}
#endif
#endif

template <class MyDevice>
void TensorTools::uniform_to_bernoulli_dev(const MyDevice & dev, Tensor& x, float p) {
  tvec(x).device(*dev.edevice) = (tvec(x) + p).floor();
}
#ifdef __CUDACC__
template void TensorTools::uniform_to_bernoulli_dev<Device_GPU>(const Device_GPU & dev, Tensor& d, float p);
#else
template void TensorTools::uniform_to_bernoulli_dev<Device_CPU>(const Device_CPU & dev, Tensor& d, float p);
#ifdef HAVE_CUDA
extern template void TensorTools::uniform_to_bernoulli_dev<Device_GPU>(const Device_GPU & dev, Tensor& d, float p);
void TensorTools::uniform_to_bernoulli(Tensor& d, float p) {
  if (d.device->type == DeviceType::CPU) { return uniform_to_bernoulli_dev(*(const Device_CPU*)d.device, d, p); }
  else if (d.device->type == DeviceType::GPU) { return uniform_to_bernoulli_dev(*(const Device_GPU*)d.device, d, p); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
void TensorTools::uniform_to_bernoulli(Tensor& d, float p) {
  if (d.device->type == DeviceType::CPU) { return uniform_to_bernoulli_dev(*(const Device_CPU*)d.device, d, p); }
  else { throw std::runtime_error("Bad device type"); }
}
#endif
#endif

template <class MyDevice>
void TensorTools::logsumexp_dev(const MyDevice & dev, const Tensor& x, Tensor & m, Tensor& z, unsigned axis) {
  DYNET_ARG_CHECK(x.d.nd <= 2, "TensorTools::logsumexp currently only supports tensors of dimension <= 2");
  unsigned other_axis = axis ^ 1;
  if(x.d.bd == 1 && x.d[other_axis] == 1) {
    t<0>(m).device(*dev.edevice) = tvec(x).maximum();
#ifdef __CUDACC__
    Eigen::array<int, 1> bcast;
    bcast[0] = x.d[0];
    // This needs to be split into two lines to prevent memory allocation
    // TODO? Here and in logsoftmax: Is there a better way to subtract a scalar that is already on the GPU without using broadcasting (and without copying the scalar back to the host first)
    t<0>(z).device(*dev.edevice) = (tvec(x) - tvec(m).broadcast(bcast)).exp().sum();
    t<0>(z).device(*dev.edevice) = tvec(z).log() + t<0>(m);
#else
    float mval = as_scalar(m);
    // This needs to be split into two lines to prevent memory allocation
    t<0>(z).device(*dev.edevice) = (tvec(x) - mval).exp().sum();
    t<0>(z).device(*dev.edevice) = t<0>(z).log() + mval;
#endif
  } else {
    Eigen::array<int, 1> red_axis; red_axis[0] = axis;
    tb<1>(m).device(*dev.edevice) = tb<2>(x).maximum(red_axis);
    // TODO: Currently, the first version is slower on CPU, hence the switch
#ifdef __CUDACC__
    Eigen::array<int, 3> bcast = {1, 1, 1}; bcast[axis] = (int)x.d[axis];
    Eigen::array<int, 3> morph = {1, 1, (int)m.d.bd}; morph[other_axis] = (int)m.d[0];
    // This needs to be split into two lines to prevent memory allocation
    tb<1>(z).device(*dev.edevice) = (tb<2>(x) - tb<2>(m).reshape(morph).broadcast(bcast)).exp().sum(red_axis);
    tb<1>(z).device(*dev.edevice) = tb<1>(z).log() + tb<1>(m);
#else
    auto miter = m.v;
    for(size_t b = 0; b < x.d.bd; ++b) {
      for(size_t i = 0; i < x.d[1]; ++i, ++miter) {
        tb<1>(z).chip<1>(b).chip<0>(i).device(*dev.edevice) = (tb<2>(x).chip<2>(b).chip(i,other_axis) - *miter).exp().sum();
        tb<1>(z).chip<1>(b).chip<0>(i).device(*dev.edevice) = tb<1>(z).chip<1>(b).chip<0>(i).log() + *miter;
      }
    }
#endif
  }
}
#ifdef __CUDACC__
template void TensorTools::logsumexp_dev<Device_GPU>(const Device_GPU & dev, const Tensor &x, Tensor &m, Tensor &z, unsigned d);
#else
template void TensorTools::logsumexp_dev<Device_CPU>(const Device_CPU & dev, const Tensor &x, Tensor &m, Tensor &z, unsigned d);
#ifdef HAVE_CUDA
extern template void TensorTools::logsumexp_dev<Device_GPU>(const Device_GPU & dev, const Tensor &x, Tensor &m, Tensor &z, unsigned d);
void TensorTools::logsumexp(const Tensor &x, Tensor &m, Tensor &z, unsigned d) {
  if (x.device->type == DeviceType::CPU) { return logsumexp_dev(*(const Device_CPU*)x.device, x, m, z, d); }
  else if (x.device->type == DeviceType::GPU) { return logsumexp_dev(*(const Device_GPU*)x.device, x, m, z, d); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
void TensorTools::logsumexp(const Tensor &x, Tensor &m, Tensor &z, unsigned d) {
  if (x.device->type == DeviceType::CPU) { return logsumexp_dev(*(const Device_CPU*)x.device, x, m, z, d); }
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
  tb<3>(ids).device(*dev.edevice) = tb<4>(v).argmax(dim);
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
  tb<3>(ids).device(*dev.edevice) = (tb<4>(v) - (-tb<4>(copy).log()).log()).argmax(dim);
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

