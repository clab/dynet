#ifndef CNN_EIGEN_TENSOR_H
#define CNN_EIGEN_TENSOR_H

#include <initializer_list>
#include <vector>

#include "cnn/dim.h"
#include "cnn/backends/eigen/random.h"

#if HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include <boost/serialization/array.hpp>
// CNN manages its own memory. DO NOT remove the following line
#define EIGEN_NO_MALLOC
#include <Eigen/Eigen>

namespace cnn {

#define EIGEN_BACKEND 1

typedef float real;

struct Tensor {
  Tensor() = default;
  Tensor(const Dim& d, float* v) : d(d), v(v) {}
  const Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> operator*() const {
    return Eigen::Map<Eigen::MatrixXf, Eigen::Aligned>(v, d.rows(), d.cols());
  }
  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> operator*() {
    return Eigen::Map<Eigen::MatrixXf, Eigen::Aligned>(v, d.rows(), d.cols());
  }
  Dim d;
  float* v;

 private:
  friend class boost::serialization::access;
  template<class Archive>
  void save(Archive& ar, const unsigned int) const {
    ar & d;
    ar & boost::serialization::make_array(v, d.size());
  }
  template<class Archive>
  void load(Archive& ar, const unsigned int) {
    ar & d;
    v = static_cast<float*>(std::malloc(d.size() * sizeof(float)));
    ar & boost::serialization::make_array(v, d.size());
  }
  BOOST_SERIALIZATION_SPLIT_MEMBER()
};

real as_scalar(const Tensor& t);
std::vector<real> as_vector(const Tensor& v);

struct TensorTools {
  static void Constant(Tensor& d, float c);
  static void Zero(Tensor& d);
  static void Randomize(Tensor& val, real scale);
  static void Randomize(Tensor& d);
  static void RandomBernoulli(Tensor& val, real p);
  static void RandomizeNormal(real mean, real stddev, Tensor& val);
  // AccessElement is very, very slow (potentially) - use appropriately
  static float AccessElement(const Tensor& v, const Dim& index);
};
real rand01();

} // namespace cnn

#endif
