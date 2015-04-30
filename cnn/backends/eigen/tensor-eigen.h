../cnn/backends/eigen/edges.cc#ifndef CNN_TENSOR_EIGEN_H_
#define CNN_TENSOR_EIGEN_H_

#include <initializer_list>
#include <random>
#include <vector>

#include <Eigen/Eigen>
#include "cnn/backends/eigen/dim.h"
#include "cnn/backends/eigen/eigen-serialization.h"
#include "cnn/backends/eigen/random.h"

namespace cnn {

#define EIGEN_BACKEND 1

typedef Eigen::MatrixXf Tensor;
typedef float real;

inline real as_scalar(const Tensor& t) {
  assert(t.cols() == 1);
  assert(t.rows() == 1);
  return t(0,0);
}

inline std::vector<real> as_vector(const Tensor& v) {
  std::vector<real> res(v.rows() * v.cols());
  std::memcpy(&res[0], v.data(), sizeof(real) * res.size());
  return res;
}

// dummy function with Eigen backend
inline Tensor FromEigenMatrix(const Eigen::MatrixXf& src) { return src; }

inline Tensor FromRawData(const Dim& dim, const float* data) {
  Tensor t(dim.size(0), dim.ndims() > 1 ? dim.size(1) : 1);
  std::memcpy(t.data(), data, sizeof(float) * dim.size());
  return t;
}

inline Tensor Zero(const Dim& d) { return Eigen::MatrixXf::Zero(d.rows(), d.cols()); }
inline Tensor Ones(const Dim& d) { return Eigen::MatrixXf::Ones(d.rows(), d.cols()); }
inline Tensor Constant(const Dim& d, real c) { return Eigen::MatrixXf::Constant(d.rows(), d.cols(), c); }
inline Tensor Random(const Dim& d, real scale) {
  std::uniform_real_distribution<real> distribution(-scale,scale);
  auto b = [&] (real) {return distribution(*rndeng);};
  return Eigen::MatrixXf::NullaryExpr(d.rows(), d.cols(), b);
}
inline Tensor Random(const Dim& d) {
  return Random(d, sqrt(6) / sqrt(d.cols() + d.rows()));
}
inline Tensor RandomBernoulli(const Dim& d, real p) {
  std::bernoulli_distribution distribution(p);
  auto b = [&] (real) {return distribution(*rndeng);};
  return Eigen::MatrixXf::NullaryExpr(d.rows(), d.cols(), b);
}
inline Tensor RandomNormal(const Dim& d, real mean, real stddev) {
  std::normal_distribution<real> distribution(mean, stddev);
  auto b = [&] (real) {return distribution(*rndeng);};
  return Eigen::MatrixXf::NullaryExpr(d.rows(), d.cols(), b);
}
inline real rand01() {
  std::uniform_real_distribution<real> distribution(0, 1);
  return distribution(*rndeng);
}

inline Dim size(const Tensor& m) {
  if (m.cols() == 1) return Dim({m.rows()});
  return Dim(m.rows(), m.cols());
}

// column major constructor
inline Tensor Ccm(const Dim&d, const std::initializer_list<real>& v) {
  std::cerr << "d: " << d << std::endl;
  Tensor m = Zero(d);
  int cc = 0;
  int cr = 0;
  for (const auto& x : v) {
    m(cr, cc) = x;
    ++cr;
    if (cr == d.rows()) { cr = 0; ++cc; }
  }
  return m;
}

inline std::string str(const Tensor& T) {
  std::ostringstream os;
  os << T << std::endl;
  return os.str();
}

} // namespace cnn

#include "cnn/backends/eigen/eigen-backend.h"

#endif
