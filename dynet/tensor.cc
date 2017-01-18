#include "dynet/tensor.h"

#include <random>
#include <vector>
#include <cstring>

#include <boost/serialization/version.hpp>
#include <boost/serialization/array.hpp>

#if HAVE_CUDA
#include "dynet/gpu-ops.h"
#include "dynet/cuda.h"
#endif
#include "dynet/io-macros.h"

using namespace std;

namespace dynet {

ostream& operator<<(ostream& os, const Tensor& t) {
#if HAVE_CUDA
  vector<real> vt = as_vector(t);
  Eigen::Map<Eigen::MatrixXf> m(&vt[0], t.d.rows(), t.d.cols());
  os << m;
#else
  os << (*t);
#endif
  return os;
}

real as_scalar(const Tensor& t) {
  assert(t.d.size() == 1);
#if HAVE_CUDA
  float res;
  CUDA_CHECK(cudaMemcpy(&res, t.v, sizeof(float), cudaMemcpyDeviceToHost));
  return res;
#else
  return t.v[0];
#endif
}

vector<real> as_vector(const Tensor& v) {
  vector<real> res(v.d.size());
#if HAVE_CUDA
  CUDA_CHECK(cudaMemcpy(&res[0], v.v, sizeof(real) * res.size(), cudaMemcpyDeviceToHost));
#else
  memcpy(&res[0], v.v, sizeof(real) * res.size());
#endif
  return res;
}

float TensorTools::AccessElement(const Tensor& v, int index) {
#if HAVE_CUDA
  float ret;
  cudaMemcpyAsync(&ret, &v.v[index], sizeof(real), cudaMemcpyDeviceToHost);
  return ret;
#else
  return v.v[index];
#endif
}

float TensorTools::AccessElement(const Tensor& v, const Dim& index) {
#if HAVE_CUDA
  abort();
#else
  return (*v)(index[0], index[1]);
#endif
}

void TensorTools::SetElement(const Tensor& v, int index, float value) {
#if HAVE_CUDA
  cudaMemcpyAsync(&v.v[index], &value, sizeof(real), cudaMemcpyHostToDevice);
#else
  v.v[index] = value;
#endif
}

void TensorTools::CopyElement(const Tensor& l, int lindex, Tensor& r, int rindex) {
#if HAVE_CUDA
  cudaMemcpyAsync(&r.v[rindex], &l.v[lindex], sizeof(real), cudaMemcpyDeviceToDevice);
#else
  r.v[rindex] = l.v[lindex];
#endif
}

void TensorTools::SetElements(const Tensor& v, const vector<float>& vec) {
#if HAVE_CUDA
  cudaMemcpyAsync(v.v, &vec[0], sizeof(real) * vec.size(), cudaMemcpyHostToDevice);
#else
  memcpy(v.v, &vec[0], sizeof(real) * vec.size());
#endif
}

void TensorTools::CopyElements(const Tensor& v, const Tensor& v_src) {
#if HAVE_CUDA
  cudaMemcpyAsync(v.v, v_src.v, sizeof(real) * v.d.size(), cudaMemcpyDeviceToDevice);
#else
  memcpy(v.v, v_src.v, sizeof(real) * v.d.size());
#endif
}

void TensorTools::Constant(Tensor& d, float c) {
#if HAVE_CUDA
  if (!c) {
    CUDA_CHECK(cudaMemsetAsync(d.v, 0, d.d.size() * sizeof(float)));
  } else {
    dynet::gpu::const_init(d.d.size(), c, d.v);
  }
#else
  if (!c) {
    memset(d.v, c, d.d.size() * sizeof(float));
  } else {
    fill(d.v, d.v + d.d.size(), c);
  }
#endif
}

void TensorTools::Zero(Tensor& d) {
  Constant(d, 0);
}

void TensorTools::Identity(Tensor& val) {
  if(val.d.nd != 2 || val.d[0] != val.d[1])
    throw std::runtime_error("Attempt to set a tensor that is not a square matrix to identity");
  size_t pos = 0;
#if HAVE_CUDA
  float* t = new float[val.d.size()];
  for(size_t i = 0; i < val.d[0]; ++i)
    for(size_t j = 0; j < val.d[1]; ++j)
      t[pos++] = (i==j?1:0);
  CUDA_CHECK(cudaMemcpy(val.v, t, sizeof(real) * val.d.size(), cudaMemcpyHostToDevice));
  delete[] t;
#else
  for(size_t i = 0; i < val.d[0]; ++i)
    for(size_t j = 0; j < val.d[1]; ++j)
      val.v[pos++] = (i==j?1:0);
#endif
}


void TensorTools::RandomizeBernoulli(Tensor& val, real p, real scale) {
  bernoulli_distribution distribution(p);
  auto b = [&] {return distribution(*rndeng) * scale;};
#if HAVE_CUDA
  float* t = new float[val.d.size()];
  generate(t, t + val.d.size(), b);
  CUDA_CHECK(cudaMemcpy(val.v, t, sizeof(real) * val.d.size(), cudaMemcpyHostToDevice));
  delete[] t;
#else
  generate(val.v, val.v + val.d.size(), b);
#endif
}

void TensorTools::RandomizeNormal(Tensor& val, real mean, real stddev) {
  normal_distribution<real> distribution(mean, stddev);
  auto b = [&] {return distribution(*rndeng);};
#if HAVE_CUDA
  float* t = new float[val.d.size()];
  generate(t, t + val.d.size(), b);
  CUDA_CHECK(cudaMemcpy(val.v, t, sizeof(real) * val.d.size(), cudaMemcpyHostToDevice));
  delete[] t;
#else
  generate(val.v, val.v + val.d.size(), b);
#endif
}

void TensorTools::RandomizeUniform(Tensor& val, real left, real right) {
  uniform_real_distribution<real> distribution(left, right);
  auto b = [&] {return distribution(*rndeng);};
#if HAVE_CUDA
  float* t = new float[val.d.size()];
  generate(t, t + val.d.size(), b);
  CUDA_CHECK(cudaMemcpy(val.v, t, sizeof(real) * val.d.size(), cudaMemcpyHostToDevice));
  delete[] t;
#else
  generate(val.v, val.v + val.d.size(), b);
#endif
}

template<class Archive>
void Tensor::save(Archive& ar, const unsigned int ver) const {
  ar & d;
  int dev_id = ((device == default_device) ? (int)-1 : device->device_id);
  ar & dev_id;
  ar & mem_pool;
#ifdef HAVE_CUDA
  if(device->type == DeviceType::GPU) {
    float* vc = static_cast<float*>(std::malloc(d.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(vc, v, d.size() * sizeof(float), cudaMemcpyDeviceToHost));
    ar & boost::serialization::make_array(vc, d.size());
    free(vc);
  } else {
    ar & boost::serialization::make_array(v, d.size());
  }
#else
  ar & boost::serialization::make_array(v, d.size());
#endif
}
template<class Archive>
void Tensor::load(Archive& ar, const unsigned int ver) {
  ar & d;
  int dev_id = -1;
  // This default value is for backward compatibility with models that were
  // saved without information about what mempool a tensor belongs to.
  mem_pool = DeviceMempool::PS;
  if(ver > 0) {
    ar & dev_id;
    ar & mem_pool;
  }
  if(dev_id == -1) {
    device = default_device;
  } else {
    assert(dev_id > 0 && dev_id < (int)devices.size());
    device = devices[dev_id];
  }
  device->allocate_tensor(mem_pool, *this);
#ifdef HAVE_CUDA
  if(device->type == DeviceType::GPU) {
    float* vc = static_cast<float*>(std::malloc(d.size() * sizeof(float)));
    ar & boost::serialization::make_array(vc, d.size());
    CUDA_CHECK(cudaMemcpyAsync(v, vc, d.size() * sizeof(float), cudaMemcpyHostToDevice));
    free(vc);
  } else {
    ar & boost::serialization::make_array(v, d.size());
  }
#else
  ar & boost::serialization::make_array(v, d.size());
#endif
}
DYNET_SAVELOAD_IMPL(Tensor)

real rand01() {
  uniform_real_distribution<real> distribution(0, 1);
  return distribution(*rndeng);
}

int rand0n(int n) {
  assert(n > 0);
  int x = rand01() * n;
  while(n == x) { x = rand01() * n; }
  return x;
}

real rand_normal() {
  normal_distribution<real> distribution(0, 1);
  return distribution(*rndeng);
}

} // namespace dynet

