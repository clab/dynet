#include "cnn/backends/eigen/tensor.h"

#include <random>
#include <vector>
#include <cstring>

using namespace std;

namespace cnn {

real as_scalar(const Tensor& t) {
  assert(t.d.size() == 1);
  return t.v[0];
}

std::vector<real> as_vector(const Tensor& v) {
  std::vector<real> res(v.d.size());
#if HAVE_CUDA
  cudaMemcpy(&res[0], v.v, sizeof(real) * res.size(), cudaMemcpyHostToDevice);
#else
  std::memcpy(&res[0], v.v, sizeof(real) * res.size());
#endif
  return res;
}

void TensorTools::Constant(Tensor& d, float c) {
  if (!c) {
    std::memset(d.v, c, d.d.size() * sizeof(float));
  } else {
    std::fill(d.v, d.v + d.d.size(), c);
  }
}

void TensorTools::Zero(Tensor& d) {
  Constant(d, 0);
}

void TensorTools::Randomize(Tensor& val, real scale) {
  std::uniform_real_distribution<real> distribution(-scale,scale);
  auto b = [&] (real) {return distribution(*rndeng);};
  *val = Eigen::MatrixXf::NullaryExpr(val.d.rows(), val.d.cols(), b);
}

void TensorTools::Randomize(Tensor& d) {
  Randomize(d, sqrt(6) / sqrt(d.d[0] + d.d[1]));
}

void TensorTools::RandomBernoulli(Tensor& val, real p) {
  std::bernoulli_distribution distribution(p);
  auto b = [&] (real) {return distribution(*rndeng);};
  *val = Eigen::MatrixXf::NullaryExpr(val.d.rows(), val.d.cols(), b);
}

void TensorTools::RandomizeNormal(real mean, real stddev, Tensor& val) {
  std::normal_distribution<real> distribution(mean, stddev);
  auto b = [&] (real) {return distribution(*rndeng);};
  *val = Eigen::MatrixXf::NullaryExpr(val.d.rows(), val.d.cols(), b);
}

real rand01() {
  std::uniform_real_distribution<real> distribution(0, 1);
  return distribution(*rndeng);
}

} // namespace cnn
