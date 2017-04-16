#include "dynet/tensor.h"
#include "dynet/globals.h"

#include <random>
#include <vector>
#include <cstring>

#include <boost/serialization/version.hpp>
#include <boost/serialization/array.hpp>

#if HAVE_CUDA
#include "dynet/gpu-ops.h"
#include "dynet/cuda.h"
#endif

using namespace std;

namespace dynet {

ostream& operator<<(ostream& os, const Tensor& t) {
  if (t.device->type == DeviceType::CPU) {
    os << (*t);
  } else {
#if HAVE_CUDA
    if (t.device->type == DeviceType::GPU) {
      vector<real> vt = as_vector(t);
      Eigen::Map<Eigen::MatrixXf> m(&vt[0], t.d.rows(), t.d.cols());
      os << m;
    }
#endif
  }
  return os;
}

real as_scalar(const Tensor& t) {
  if (t.d.size() != 1)
    throw std::runtime_error("Input tensor has more than one element, cannot convert to scalar.");
  real res = 0.;
  if (t.device->type == DeviceType::CPU) {
    return t.v[0];
  } else {
#if HAVE_CUDA
    if (t.device->type == DeviceType::GPU) {
      CUDA_CHECK(cudaMemcpy(&res, t.v, sizeof(float), cudaMemcpyDeviceToHost));
      return res;
    }
#endif
  }
  return res;
}

vector<real> as_vector(const Tensor& v) {
  vector<real> res(v.d.size());
  if (v.device->type == DeviceType::CPU) {
    memcpy(&res[0], v.v, sizeof(real) * res.size());
  } else {
#if HAVE_CUDA
    if (v.device->type == DeviceType::GPU) {
      CUDA_CHECK(cudaMemcpy(&res[0], v.v, sizeof(real) * res.size(), cudaMemcpyDeviceToHost));
    }
#endif
  }
  return res;
}

float TensorTools::access_element(const Tensor& v, int index) {
  float ret = 0.;
  if (v.device->type == DeviceType::CPU) {
    return v.v[index];
  } else {
#if HAVE_CUDA
    if (v.device->type == DeviceType::GPU) {
      cudaMemcpy(&ret, &v.v[index], sizeof(real), cudaMemcpyDeviceToHost);
      return ret;
    }
#endif
  }
  return ret;
}

float TensorTools::access_element(const Tensor& v, const Dim& index) {
#if HAVE_CUDA
  throw std::runtime_error("TensorTools::access_element(Tensor,Dim) not implemented for CUDA");
#else
  return (*v)(index[0], index[1]);
#endif
}

void TensorTools::set_element(const Tensor& v, int index, float value) {
  if (v.device->type == DeviceType::CPU) {
    v.v[index] = value;
  } else {
#if HAVE_CUDA
    if (v.device->type == DeviceType::GPU) {
      cudaMemcpyAsync(&v.v[index], &value, sizeof(real), cudaMemcpyHostToDevice);
    }
#endif
  }
}

void TensorTools::copy_element(const Tensor& l, int lindex, Tensor& r, int rindex) {
  if (l.device->type == DeviceType::CPU) {
    r.v[rindex] = l.v[lindex];
  } else {
#if HAVE_CUDA
    if (l.device != r.device)
      throw std::invalid_argument("TensorTools::CopyElement doesn't support inter-device copy yet");
    if (l.device->type == DeviceType::GPU) {
      cudaMemcpyAsync(&r.v[rindex], &l.v[lindex], sizeof(real), cudaMemcpyDeviceToDevice);
    }
#endif
  }
}

void TensorTools::set_elements(const Tensor& v, const vector<float>& vec) {
  if (v.device->type == DeviceType::CPU) {
    memcpy(v.v, &vec[0], sizeof(real) * vec.size());
  } else {
#if HAVE_CUDA
    if (v.device->type == DeviceType::GPU) {
      cudaMemcpyAsync(v.v, &vec[0], sizeof(real) * vec.size(), cudaMemcpyHostToDevice);
    }
#endif
  }
}

void TensorTools::copy_elements(const Tensor& v, const Tensor& v_src) {
  if (v.device->type == DeviceType::CPU) {
    memcpy(v.v, v_src.v, sizeof(real) * v.d.size());
  } else {
#if HAVE_CUDA
    if (v.device != v_src.device)
      throw std::invalid_argument("TensorTools::CopyElement doesn't support inter-device copy yet");
    if (v.device->type == DeviceType::GPU) {
      cudaMemcpyAsync(v.v, v_src.v, sizeof(real) * v.d.size(), cudaMemcpyDeviceToDevice);
    }
#endif
  }
}

void TensorTools::clip(Tensor& d, float left, float right){
#if HAVE_CUDA
	if (d.device->type == DeviceType::GPU){
		dynet::gpu::clip(d.d.size(), left, right, d.v);
	} else {
#endif
		for (size_t pos =0; pos < d.d.size(); ++pos)
		  d.v[pos] = min(right, max(left, d.v[pos]));
#if HAVE_CUDA
	}
#endif
}

void TensorTools::constant(Tensor& d, float c) {
  if (d.device->type == DeviceType::CPU) {
    if (!c) {
      memset(d.v, c, d.d.size() * sizeof(float));
    } else {
      fill(d.v, d.v + d.d.size(), c);
    }
  } else {
#if HAVE_CUDA
    if (d.device->type == DeviceType::GPU) {
      if (!c) {
        CUDA_CHECK(cudaMemsetAsync(d.v, 0, d.d.size() * sizeof(float)));
      } else {
        dynet::gpu::const_init(d.d.size(), c, d.v);
      }
    }
#endif
  }
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
  } else {
#if HAVE_CUDA
    if (val.device->type == DeviceType::GPU) {
      float* t = new float[val.d.size()];
      for (size_t i = 0; i < val.d[0]; ++i)
        for (size_t j = 0; j < val.d[1]; ++j)
          t[pos++] = (i == j ? 1 : 0);
      CUDA_CHECK(cudaMemcpy(val.v, t, sizeof(real) * val.d.size(), cudaMemcpyHostToDevice));
      delete[] t;
    }
#endif
  }
}

void TensorTools::randomize_bernoulli(Tensor& val, real p, real scale) {
  bernoulli_distribution distribution(p);
  auto b = [&] {return distribution(*rndeng) * scale;};
  if (val.device->type == DeviceType::CPU) {
    generate(val.v, val.v + val.d.size(), b);
  } else {
#if HAVE_CUDA
    if (val.device->type == DeviceType::GPU) {
      float* t = new float[val.d.size()];
      generate(t, t + val.d.size(), b);
      CUDA_CHECK(cudaMemcpy(val.v, t, sizeof(real) * val.d.size(), cudaMemcpyHostToDevice));
      delete[] t;
    }
#endif
  }
}

void TensorTools::randomize_normal(Tensor& val, real mean, real stddev) {
  normal_distribution<real> distribution(mean, stddev);
  auto b = [&] {return distribution(*rndeng);};
  if (val.device->type == DeviceType::CPU) {
    generate(val.v, val.v + val.d.size(), b);
  } else {
#if HAVE_CUDA
    if (val.device->type == DeviceType::GPU) {
      float* t = new float[val.d.size()];
      generate(t, t + val.d.size(), b);
      CUDA_CHECK(cudaMemcpy(val.v, t, sizeof(real) * val.d.size(), cudaMemcpyHostToDevice));
      delete[] t;
    }
#endif
  }
}

void TensorTools::randomize_uniform(Tensor& val, real left, real right) {
  uniform_real_distribution<real> distribution(left, right);
  auto b = [&] {return distribution(*rndeng);};
  if (val.device->type == DeviceType::CPU) {
    generate(val.v, val.v + val.d.size(), b);
  } else {
#if HAVE_CUDA
    if (val.device->type == DeviceType::GPU) {
      float* t = new float[val.d.size()];
      generate(t, t + val.d.size(), b);
      CUDA_CHECK(cudaMemcpy(val.v, t, sizeof(real) * val.d.size(), cudaMemcpyHostToDevice));
      delete[] t;
    }
#endif
  }
}

void TensorTools::randomize_orthonormal(Tensor& val, real scale) {
  if (val.d.nd != 2 || val.d[0] != val.d[1])
    throw std::runtime_error("Attempt to set a tensor that is not a square matrix to an orthogonal matrix");
#ifdef HAVE_CUDA
  throw std::runtime_error("Orthonormal initialization not implemented in CUDA (we welcome pull requests)");
#else
  randomize_uniform(val, -1.0, 1.0);
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(*val, Eigen::ComputeFullU | Eigen::ComputeThinV);
  *val = scale * svd.matrixU();
#endif
}

template<class Archive>
void Tensor::save(Archive& ar, const unsigned int ver) const {
  ar & d;
  int dev_id = ((device == default_device) ? (int) - 1 : device->device_id);
  ar & dev_id;
  ar & mem_pool;
  if (device->type == DeviceType::CPU) {
    ar & boost::serialization::make_array(v, d.size());
  } else {
#ifdef HAVE_CUDA
    if (device->type == DeviceType::GPU) {
      float* vc = static_cast<float*>(std::malloc(d.size() * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(vc, v, d.size() * sizeof(float), cudaMemcpyDeviceToHost));
      ar & boost::serialization::make_array(vc, d.size());
      free(vc);
    }
#endif
  }
}

template<class Archive>
void Tensor::load(Archive& ar, const unsigned int ver) {
  ar & d;
  int dev_id = -1;
  // This default value is for backward compatibility with models that were
  // saved without information about what mempool a tensor belongs to.
  mem_pool = DeviceMempool::PS;
  if (ver > 0) {
    ar & dev_id;
    ar & mem_pool;
  }
  if (dev_id == -1) {
    device = default_device;
  } else {
    DYNET_ASSERT(dev_id > 0 && dev_id < (int)devices.size(), "Bad device id " << dev_id << " in Tensor::load with " << devices.size() << " total devices");
    device = devices[dev_id];
  }
  device->allocate_tensor(mem_pool, *this);
  if (device->type == DeviceType::CPU) {
    ar & boost::serialization::make_array(v, d.size());
  } else {
#ifdef HAVE_CUDA
    if (device->type == DeviceType::GPU) {
      float* vc = static_cast<float*>(std::malloc(d.size() * sizeof(float)));
      ar & boost::serialization::make_array(vc, d.size());
      CUDA_CHECK(cudaMemcpyAsync(v, vc, d.size() * sizeof(float), cudaMemcpyHostToDevice));
      free(vc);
    }
#endif
  }
}
DYNET_SAVELOAD_IMPL(Tensor)

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

} // namespace dynet

