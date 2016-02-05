#ifndef CNN_EIGEN_TENSOR_H
#define CNN_EIGEN_TENSOR_H

#include <initializer_list>
#include <vector>

#include "cnn/dim.h"
#include "cnn/random.h"
#include "cnn/aligned-mem-pool.h"

#if HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include "cnn/cuda.h"
#endif
#include <boost/serialization/array.hpp>

// Following line is commented out because it causes errors with large nets (Antonis)
//#define EIGEN_NO_MALLOC

#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

namespace cnn {

#define EIGEN_BACKEND 1

typedef float real;

struct Tensor {
  Tensor() = default;
  Tensor(const Dim& d, float* v) : d(d), v(v) {}
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
  // Get view as a Tensor (see specializations below-- this is to work Eigen's and CNNs compile-type vs. run-time differences)
  template <int Order> Eigen::TensorMap<Eigen::Tensor<float,Order>> t();
  template <int Order> const Eigen::TensorMap<Eigen::Tensor<float,Order>> t() const;
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
      Tensor ret(new_d, v + bsize * b);
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
      assert (d.batch_elems() >= 0);
      for(unsigned b = 0; b < d.batch_elems(); ++b)
        bs[b] = Tensor(new_d, v + bsize * b);
      return bs;
    }
  }

  Dim d;  // shape of tensor
  float* v;  // pointer to memory
  std::vector<Tensor> bs;
  // TODO start using this
  //Device* device; // which device does it live on?

 private:
  friend class boost::serialization::access;
  template<class Archive>
  void save(Archive& ar, const unsigned int) const {
    ar & d;
    // TODO(mem) save device
#if HAVE_CUDA
    float* vc = (float*)malloc(d.size() * sizeof(float));
    CUDA_CHECK(cudaMemcpy(vc, v, d.size() * sizeof(float), cudaMemcpyDeviceToHost));
    ar & boost::serialization::make_array(vc, d.size());
    free(vc);
#else
    ar & boost::serialization::make_array(v, d.size());
#endif
  }
  template<class Archive>
  void load(Archive& ar, const unsigned int) {
    ar & d;
    // TODO(mem) - load device and use it to create memory allocator
    // Devices should probably know how to load and save data to disk
#if HAVE_CUDA
    CUDA_CHECK(cudaMalloc(&v, d.size() * sizeof(float)));
    float* vc = static_cast<float*>(std::malloc(d.size() * sizeof(float)));
    ar & boost::serialization::make_array(vc, d.size());
    CUDA_CHECK(cudaMemcpyAsync(v, vc, d.size() * sizeof(float), cudaMemcpyHostToDevice));
#else
    v = static_cast<float*>(_mm_malloc(d.size() * sizeof(float), 32));
    ar & boost::serialization::make_array(v, d.size());
#endif
  }
  BOOST_SERIALIZATION_SPLIT_MEMBER()
};

template<> inline Eigen::TensorMap<Eigen::Tensor<float,1>> Tensor::t<1>() {
  assert(d.ndims() == 1);
  return Eigen::TensorMap<Eigen::Tensor<float,1>>(v, (int)d[0]);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float,1>> Tensor::t<1>() const {
  assert(d.ndims() == 1);
  return Eigen::TensorMap<Eigen::Tensor<float,1>>(v, (int)d[0]);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<float,2>> Tensor::t<2>() {
  assert(d.ndims() == 2);
  return Eigen::TensorMap<Eigen::Tensor<float,2>>(v, (int)d[0], (int)d[1]);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float,2>> Tensor::t<2>() const {
  assert(d.ndims() == 2);
  return Eigen::TensorMap<Eigen::Tensor<float,2>>(v, (int)d[0], (int)d[1]);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<float,3>> Tensor::t<3>() {
  assert(d.ndims() == 3);
  return Eigen::TensorMap<Eigen::Tensor<float,3>>(v, (int)d[0], (int)d[1], (int)d[2]);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float,3>> Tensor::t<3>() const {
  assert(d.ndims() == 3);
  return Eigen::TensorMap<Eigen::Tensor<float,3>>(v, (int)d[0], (int)d[1], (int)d[2]);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<float,4>> Tensor::t<4>() {
  assert(d.ndims() == 4);
  return Eigen::TensorMap<Eigen::Tensor<float,4>>(v, (int)d[0], (int)d[1], (int)d[2], (int)d[3]);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float,4>> Tensor::t<4>() const {
  assert(d.ndims() == 4);
  return Eigen::TensorMap<Eigen::Tensor<float,4>>(v, (int)d[0], (int)d[1], (int)d[2], (int)d[3]);
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

  static void SetElements(const Tensor& v, const std::vector<float>& vec);
  static void CopyElements(const Tensor& v, const Tensor& v_src);
};
real rand01();
int rand0n(int n);
real rand_normal();

} // namespace cnn

#endif
