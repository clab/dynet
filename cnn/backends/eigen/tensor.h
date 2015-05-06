#ifndef CNN_EIGEN_TENSOR_H
#define CNN_EIGEN_TENSOR_H

#include <initializer_list>
#include <random>
#include <vector>

#include "cnn/backends/eigen/random.h"
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

inline real as_scalar(const Tensor& t) {
  assert(t.d.size() == 1);
  return t.v[0];
}

inline std::vector<real> as_vector(const Tensor& v) {
  std::vector<real> res(v.d.size());
  std::memcpy(&res[0], v.v, sizeof(real) * res.size());
  return res;
}

inline void Constant(Tensor& d, float c) {
  std::memset(d.v, c, d.d.size() * sizeof(float));
}
inline void Zero(Tensor& d) {
  Constant(d, 0);
}
inline void Randomize(Tensor& val, real scale) {
  std::uniform_real_distribution<real> distribution(-scale,scale);
  auto b = [&] (real) {return distribution(*rndeng);};
  *val = Eigen::MatrixXf::NullaryExpr(val.d.rows(), val.d.cols(), b);
}
inline void Randomize(Tensor& d) {
  Randomize(d, sqrt(6) / sqrt(d.d[0] + d.d[1]));
}
inline void RandomBernoulli(Tensor& val, real p) {
  std::bernoulli_distribution distribution(p);
  auto b = [&] (real) {return distribution(*rndeng);};
  *val = Eigen::MatrixXf::NullaryExpr(val.d.rows(), val.d.cols(), b);
}
inline void RandomizeNormal(real mean, real stddev, Tensor& val) {
  std::normal_distribution<real> distribution(mean, stddev);
  auto b = [&] (real) {return distribution(*rndeng);};
  *val = Eigen::MatrixXf::NullaryExpr(val.d.rows(), val.d.cols(), b);
}
inline real rand01() {
  std::uniform_real_distribution<real> distribution(0, 1);
  return distribution(*rndeng);
}

} // namespace cnn

#endif
