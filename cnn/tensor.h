#ifndef CNN_EIGEN_TENSOR_H
#define CNN_EIGEN_TENSOR_H

#include <initializer_list>
#include <vector>

#include "cnn/dim.h"
#include "cnn/globals.h"
#include "cnn/aligned-mem-pool.h"
#include "cnn/devices.h"

#if HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include "cnn/cuda.h"
#endif
#include <boost/serialization/array.hpp>
#include <boost/serialization/version.hpp>

// Following line is commented out because it causes errors with large nets (Antonis)
//#define EIGEN_NO_MALLOC

#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

namespace cnn {

#define EIGEN_BACKEND 1

typedef float real;

struct Tensor {
  Tensor() = default;
  Tensor(const Dim& d, float* v, Device* dev, DeviceMempool mem) : d(d), v(v), device(dev), mem_pool(mem) {}
  // Get the data as a matrix
  const Eigen::Map<Eigen::MatrixXf> operator*() const {
    assert(d.batch_elems() == 1);
    assert(d.ndims() < 3);
    return Eigen::Map<Eigen::MatrixXf>(v, d.rows(), d.cols());
  }
  Eigen::Map<Eigen::MatrixXf> operator*() {
    assert(d.batch_elems() == 1);
    assert(d.ndims() < 3);
    return Eigen::Map<Eigen::MatrixXf>(v, d.rows(), d.cols());
  }
  // Get the data as a vector
  // this returns the full tensor contents even if it has many dimensions
  const Eigen::Map<Eigen::VectorXf> vec() const {
    return Eigen::Map<Eigen::VectorXf>(v, d.size());
  }
  Eigen::Map<Eigen::VectorXf> vec() {
    return Eigen::Map<Eigen::VectorXf>(v, d.size());
  }
  // this returns the full tensor contents as a one dimensional Eigen tensor
  // which can be used for on-device processing where dimensions aren't important
  const Eigen::TensorMap<Eigen::Tensor<float,1>> tvec() const {
    return Eigen::TensorMap<Eigen::Tensor<float,1>>(v, d.size());
  }
  Eigen::TensorMap<Eigen::Tensor<float,1>> tvec() {
    return Eigen::TensorMap<Eigen::Tensor<float,1>>(v, d.size());
  }
  // Get view as a Tensor (see specializations below-- this is to work Eigen's and CNNs compile-type vs. run-time differences)
  template <int Order> Eigen::TensorMap<Eigen::Tensor<float,Order>> t();
  template <int Order> const Eigen::TensorMap<Eigen::Tensor<float,Order>> t() const;
  // Get view as a Tensor where the final dimension is the various batches
  template <int Order> Eigen::TensorMap<Eigen::Tensor<float,Order+1>> tb();
  template <int Order> const Eigen::TensorMap<Eigen::Tensor<float,Order+1>> tb() const;
  // Get the pointer for a particular batch, automatically broadcasting if the size is zero
  const float* batch_ptr(unsigned bid) const {
    assert(d.bd == 1 || bid < d.bd);
    return v + (bid%d.bd)*d.batch_size();
  }
  float* batch_ptr(unsigned bid) {
    assert(d.bd == 1 || bid < d.bd);
    return v + (bid%d.bd)*d.batch_size();
  }
  // Get the matrix for a particular batch, automatically broadcasting if the size is zero
  const Eigen::Map<Eigen::MatrixXf> batch_matrix(unsigned bid) const {
    return Eigen::Map<Eigen::MatrixXf>(v + (bid%d.bd)*d.batch_size(), d.rows(), d.cols());
  }
  Eigen::Map<Eigen::MatrixXf> batch_matrix(unsigned bid) {
    return Eigen::Map<Eigen::MatrixXf>(v + (bid%d.bd)*d.batch_size(), d.rows(), d.cols());
  }
  // Get the data as a matrix, where each "row" is the concatenation of rows and columns,
  // and each "column" is batches
  const Eigen::Map<Eigen::MatrixXf> rowcol_matrix() const {
    return Eigen::Map<Eigen::MatrixXf>(v, d.rows()*d.cols(), d.batch_elems());
  }
  Eigen::Map<Eigen::MatrixXf> rowcol_matrix() {
    return Eigen::Map<Eigen::MatrixXf>(v, d.rows()*d.cols(), d.batch_elems());
  }
  // Get the data as a matrix, where each "row" is the concatenation of rows,
  // and each "column" is the concatenation of columns and batches
  const Eigen::Map<Eigen::MatrixXf> colbatch_matrix() const {
    return Eigen::Map<Eigen::MatrixXf>(v, d.rows(), d.cols()*d.batch_elems());
  }
  Eigen::Map<Eigen::MatrixXf> colbatch_matrix() {
    return Eigen::Map<Eigen::MatrixXf>(v, d.rows(), d.cols()*d.batch_elems());
  }
  // this is very slow: use sparingly
  inline bool is_valid() const {
#if HAVE_CUDA
    std::cerr << "is_valid() not implemented with HAVE_CUDA\n";
    abort();
#else
    const size_t s = d.size();
    for (unsigned i = 0; i < s; ++i)
      if (std::isnan(v[i]) || std::isinf(v[i])) return false;
    return true;
#endif
  }

  // Get a tensor representing a single batch.
  // If this tensor only has a single batch, then broadcast. Otherwise, check to
  // make sure that the requested batch is smaller than the number of batches.
  // TODO: This is a bit wasteful, as it re-calculates bs.batch_size() every time.
  Tensor batch_elem(unsigned b) const {
    if(d.batch_elems() == 1) {
      return *this;
    } else {
      assert(b < d.batch_elems());
      const unsigned bsize = d.batch_size();
      Dim new_d(d); new_d.bd = 1;
      Tensor ret(new_d, v + bsize * b, device, mem_pool);
      // std::cerr << "Getting tensor for batch " << (b % d.batch_elems()) << " bsize: " << bsize << ", ptr=" << (long)ret.v << std::endl;
      return ret;
    }
  }

  // get tensors for all batches
  std::vector<Tensor> batch_elems() const {
    if(d.batch_elems() == 1) {
      return std::vector<Tensor>(1, *this); 
    } else {
      std::vector<Tensor> bs(d.batch_elems());
      unsigned bsize = d.batch_size();
      Dim new_d = d; new_d.bd = 1;
      for(unsigned b = 0; b < d.batch_elems(); ++b)
        bs[b] = Tensor(new_d, v + bsize * b, device, mem_pool);
      return bs;
    }
  }

  Dim d;  // shape of tensor
  float* v;  // pointer to memory
  std::vector<Tensor> bs;
  Device* device;
  DeviceMempool mem_pool;

 private:
  friend class boost::serialization::access;
  template<class Archive>
  void save(Archive& ar, const unsigned int ver) const {
    ar & d;
    ar & ((device == default_device) ? (int)-1 : device->device_id);
    ar & mem_pool;
#ifdef HAVE_CUDA
    if(device->type == DeviceType::GPU) {
      float* vc = static_cast<float*>(std::malloc(d.size() * sizeof(float)));
      CUDA_CHECK(cudaMemcpyAsync(vc, v, d.size() * sizeof(float), cudaMemcpyDeviceToHost));
      ar & boost::serialization::make_array(vc, mem.size());
      free(vc);
    } else {
      ar & boost::serialization::make_array(v, d.size());
    }
#else
    ar & boost::serialization::make_array(v, d.size());
#endif
  }
  template<class Archive>
  void load(Archive& ar, const unsigned int ver) {
    ar & d;
    int dev_id = -1;
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
  BOOST_SERIALIZATION_SPLIT_MEMBER()
};

template<> inline Eigen::TensorMap<Eigen::Tensor<float,0>> Tensor::t<0>() {
  assert(d.batch_elems() == 1 && d.size() == 1);
  return Eigen::TensorMap<Eigen::Tensor<float,0>>(v);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float,0>> Tensor::t<0>() const {
  assert(d.batch_elems() == 1 && d.size() == 1);
  return Eigen::TensorMap<Eigen::Tensor<float,0>>(v);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<float,1>> Tensor::t<1>() {
  assert(d.batch_elems() == 1 && (d.ndims() == 1 || d.size() == d.rows()));
  return Eigen::TensorMap<Eigen::Tensor<float,1>>(v, (int)d[0]);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float,1>> Tensor::t<1>() const {
  assert(d.batch_elems() == 1 && (d.ndims() == 1 || d.size() == d.rows()));
  return Eigen::TensorMap<Eigen::Tensor<float,1>>(v, (int)d[0]);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<float,2>> Tensor::t<2>() {
  assert(d.batch_elems() == 1 && d.ndims() <= 2);
  if(d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float,2>>(v, (int)d[0], (int)d[1]);
  else               return Eigen::TensorMap<Eigen::Tensor<float,2>>(v, (int)d[0], (int)1);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float,2>> Tensor::t<2>() const {
  assert(d.batch_elems() == 1 && d.ndims() <= 2);
  if(d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float,2>>(v, (int)d[0], (int)d[1]);
  else               return Eigen::TensorMap<Eigen::Tensor<float,2>>(v, (int)d[0], (int)1);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<float,3>> Tensor::t<3>() {
  assert(d.batch_elems() == 1 && d.ndims() <= 3);
  if(d.ndims() == 3)      return Eigen::TensorMap<Eigen::Tensor<float,3>>(v, (int)d[0], (int)d[1], (int)d[2]);
  else if(d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float,3>>(v, (int)d[0], (int)d[1], (int)1);
  else                    return Eigen::TensorMap<Eigen::Tensor<float,3>>(v, (int)d[0], (int)1, (int)1);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float,3>> Tensor::t<3>() const {
  assert(d.batch_elems() == 1 && d.ndims() <= 3);
  if(d.ndims() == 3)      return Eigen::TensorMap<Eigen::Tensor<float,3>>(v, (int)d[0], (int)d[1], (int)d[2]);
  else if(d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float,3>>(v, (int)d[0], (int)d[1], (int)1);
  else                    return Eigen::TensorMap<Eigen::Tensor<float,3>>(v, (int)d[0], (int)1, (int)1);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<float,4>> Tensor::t<4>() {
  assert(d.batch_elems() == 1 && d.ndims() <= 4);
  if(d.ndims() == 4)      return Eigen::TensorMap<Eigen::Tensor<float,4>>(v, (int)d[0], (int)d[1], (int)d[2], (int)d[3]);
  else if(d.ndims() == 3) return Eigen::TensorMap<Eigen::Tensor<float,4>>(v, (int)d[0], (int)d[1], (int)d[2], (int)1);
  else if(d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float,4>>(v, (int)d[0], (int)d[1], (int)1, (int)1);
  else                    return Eigen::TensorMap<Eigen::Tensor<float,4>>(v, (int)d[0], (int)1, (int)1, (int)1);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float,4>> Tensor::t<4>() const {
  assert(d.batch_elems() == 1 && d.ndims() <= 4);
  if(d.ndims() == 4)      return Eigen::TensorMap<Eigen::Tensor<float,4>>(v, (int)d[0], (int)d[1], (int)d[2], (int)d[3]);
  else if(d.ndims() == 3) return Eigen::TensorMap<Eigen::Tensor<float,4>>(v, (int)d[0], (int)d[1], (int)d[2], (int)1);
  else if(d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float,4>>(v, (int)d[0], (int)d[1], (int)1, (int)1);
  else                    return Eigen::TensorMap<Eigen::Tensor<float,4>>(v, (int)d[0], (int)1, (int)1, (int)1);
}
// ...

template<> inline Eigen::TensorMap<Eigen::Tensor<float,1>> Tensor::tb<0>() {
  assert(d.batch_size() == 1);
  return Eigen::TensorMap<Eigen::Tensor<float,1>>(v, d.bd);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float,1>> Tensor::tb<0>() const {
  assert(d.batch_size() == 1);
  return Eigen::TensorMap<Eigen::Tensor<float,1>>(v, d.bd);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<float,2>> Tensor::tb<1>() {
  assert(d.ndims() == 1 || d.batch_size() == d.rows());
  return Eigen::TensorMap<Eigen::Tensor<float,2>>(v, (int)d[0], d.bd);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float,2>> Tensor::tb<1>() const {
  assert(d.ndims() == 1 || d.batch_size() == d.rows());
  return Eigen::TensorMap<Eigen::Tensor<float,2>>(v, (int)d[0], d.bd);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<float,3>> Tensor::tb<2>() {
  assert(d.ndims() <= 2);
  if(d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float,3>>(v, (int)d[0], (int)d[1], d.bd);
  else               return Eigen::TensorMap<Eigen::Tensor<float,3>>(v, (int)d[0], (int)1, d.bd);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float,3>> Tensor::tb<2>() const {
  assert(d.ndims() <= 2);
  if(d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float,3>>(v, (int)d[0], (int)d[1], d.bd);
  else               return Eigen::TensorMap<Eigen::Tensor<float,3>>(v, (int)d[0], (int)1, d.bd);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<float,4>> Tensor::tb<3>() {
  assert(d.ndims() <= 3);
  if(d.ndims() == 3)      return Eigen::TensorMap<Eigen::Tensor<float,4>>(v, (int)d[0], (int)d[1], (int)d[2], d.bd);
  else if(d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float,4>>(v, (int)d[0], (int)d[1], (int)1, d.bd);
  else                    return Eigen::TensorMap<Eigen::Tensor<float,4>>(v, (int)d[0], (int)1, (int)1, d.bd);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float,4>> Tensor::tb<3>() const {
  assert(d.ndims() <= 3);
  if(d.ndims() == 3)      return Eigen::TensorMap<Eigen::Tensor<float,4>>(v, (int)d[0], (int)d[1], (int)d[2], d.bd);
  else if(d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float,4>>(v, (int)d[0], (int)d[1], (int)1, d.bd);
  else                    return Eigen::TensorMap<Eigen::Tensor<float,4>>(v, (int)d[0], (int)1, (int)1, d.bd);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<float,5>> Tensor::tb<4>() {
  assert(d.ndims() <= 4);
  if(d.ndims() == 4)      return Eigen::TensorMap<Eigen::Tensor<float,5>>(v, (int)d[0], (int)d[1], (int)d[2], (int)d[3], d.bd);
  else if(d.ndims() == 3) return Eigen::TensorMap<Eigen::Tensor<float,5>>(v, (int)d[0], (int)d[1], (int)d[2], (int)1, d.bd);
  else if(d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float,5>>(v, (int)d[0], (int)d[1], (int)1, (int)1, d.bd);
  else                    return Eigen::TensorMap<Eigen::Tensor<float,5>>(v, (int)d[0], (int)1, (int)1, (int)1, d.bd);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float,5>> Tensor::tb<4>() const {
  assert(d.ndims() <= 4);
  if(d.ndims() == 4)      return Eigen::TensorMap<Eigen::Tensor<float,5>>(v, (int)d[0], (int)d[1], (int)d[2], (int)d[3], d.bd);
  else if(d.ndims() == 3) return Eigen::TensorMap<Eigen::Tensor<float,5>>(v, (int)d[0], (int)d[1], (int)d[2], (int)1, d.bd);
  else if(d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float,5>>(v, (int)d[0], (int)d[1], (int)1, (int)1, d.bd);
  else                    return Eigen::TensorMap<Eigen::Tensor<float,5>>(v, (int)d[0], (int)1, (int)1, (int)1, d.bd);
}
// ...

std::ostream& operator<<(std::ostream& os, const Tensor& t);
real as_scalar(const Tensor& t);
std::vector<real> as_vector(const Tensor& v);

struct TensorTools {
  static void Constant(Tensor& d, float c);
  static void Zero(Tensor& d);
  static void Randomize(Tensor& val, real scale);
  static void Randomize(Tensor& d);
  // sample some bernoulli random variables and scale them by scale
  static void RandomBernoulli(Tensor& val, real p, real scale = 1.0);
  static void RandomizeNormal(real mean, real stddev, Tensor& val);
  // AccessElement and SetElement are very, very slow (potentially) - use appropriately
  static float AccessElement(const Tensor& v, int index);
  static float AccessElement(const Tensor& v, const Dim& index);
  static void SetElement(const Tensor& v, int index, float value);
  static void CopyElement(const Tensor& l, int lindex, Tensor& r, int rindex);

  static void SetElements(const Tensor& v, const std::vector<float>& vec);
  static void CopyElements(const Tensor& v, const Tensor& v_src);
};
real rand01();
int rand0n(int n);
real rand_normal();

} // namespace cnn

BOOST_CLASS_VERSION(cnn::Tensor, 1)

#endif
