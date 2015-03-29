#include "cnn/backends/eigen/eigen-backend.h"

#include <random>
#include <cmath>

using namespace std;

namespace cnn {

std::mt19937* rndeng = nullptr;
void Initialize(int& argc, char**& argv) {
   std::random_device rd;
   rndeng = new mt19937(rd());
}

Eigen::MatrixXf Elewise::Ln(const Eigen::MatrixXf& x) {
  return x.array().log();
}

Eigen::MatrixXf Elewise::Exp(const Eigen::MatrixXf& x) {
  return x.array().exp();
}

Eigen::MatrixXf Elewise::SigmoidForward(const Eigen::MatrixXf& x) {
  Eigen::MatrixXf fx = x;
  for (unsigned i = 0; i < fx.rows(); ++i)
    for (unsigned j = 0; j < fx.cols(); ++j)
      fx(i,j) = 1.f / (1.f + expf(-x(i,j)));
  return fx;
}

Eigen::MatrixXf Elewise::SigmoidBackward(const Eigen::MatrixXf& diff, const Eigen::MatrixXf& top, const Eigen::MatrixXf& bottom) {
  const unsigned rows = top.rows();
  const unsigned cols = top.cols();
  Eigen::MatrixXf dfdx(rows, cols);
  for (unsigned i = 0; i < rows; ++i)
    for (unsigned j = 0; j < cols; ++j)
      dfdx(i,j) = (1.f - top(i,j)) * top(i,j);
  return dfdx.cwiseProduct(diff);
}

Eigen::MatrixXf Elewise::ReluForward(const Eigen::MatrixXf& x) {
  Eigen::MatrixXf fx = x;
  for (unsigned i = 0; i < fx.rows(); ++i)
    for (unsigned j = 0; j < fx.cols(); ++j)
      if (fx(i,j) < 0) fx(i,j) = 0;
  return fx;
}

Eigen::MatrixXf Elewise::ReluBackward(const Eigen::MatrixXf& diff, const Eigen::MatrixXf& top, const Eigen::MatrixXf& bottom) {
  Eigen::MatrixXf dfdx = diff;
  const unsigned rows = diff.rows();
  const unsigned cols = diff.cols();
  for (unsigned i = 0; i < rows; ++i)
    for (unsigned j = 0; j < cols; ++j)
      if (!top(i,j)) dfdx(i,j) = 0;
  return dfdx;
}

Eigen::MatrixXf Elewise::TanhForward(const Eigen::MatrixXf& x) {
  Eigen::MatrixXf fx = x;
  for (unsigned i = 0; i < fx.rows(); ++i)
    for (unsigned j = 0; j < fx.cols(); ++j)
      fx(i,j) = tanhf(fx(i,j));
  return fx;
}

Eigen::MatrixXf Elewise::TanhBackward(const Eigen::MatrixXf& diff, const Eigen::MatrixXf& top, const Eigen::MatrixXf& bottom) {
  const unsigned rows = top.rows();
  const unsigned cols = top.cols();
  Eigen::MatrixXf dfdx(rows, cols);
  for (unsigned i = 0; i < rows; ++i)
    for (unsigned j = 0; j < cols; ++j)
      dfdx(i,j) = 1.f - top(i,j) * top(i,j);
  return dfdx.cwiseProduct(diff);
}

inline float logsumexp(const Eigen::MatrixXf& x) {
  float m = x(0,0);
  for (unsigned i = 1; i < x.rows(); ++i) {
    float r = x(i,0);
    if (r > m) m = r;
  }
  float z = 0;
  for (unsigned i = 0; i < x.rows(); ++i)
    z += expf(x(i,0) - m);
  return m + logf(z);
}

Eigen::MatrixXf Convolution::SoftmaxForward(const Eigen::MatrixXf& src, SoftmaxAlgorithm algorithm) {
  const unsigned rows = src.rows();
  assert(src.cols() == 1);
  const float logz = logsumexp(src);
  Eigen::MatrixXf fx(rows, 1);
  for (unsigned i = 0; i < rows; ++i)
    fx(i,0) = expf(src(i,0) - logz);
  return fx;
}

Eigen::MatrixXf Convolution::SoftmaxBackward(const Eigen::MatrixXf& diff, const Eigen::MatrixXf& top, SoftmaxAlgorithm algorithm) {
  // d softmax(x)_i / d x_j = softmax(x)_i * (1 - softmax(x)_i) if i == j
  // d softmax(x)_i / d x_j = -softmax(x)_i * softmax(x)_j if i != j
  const unsigned rows = top.rows();

  float off_diag_sum = 0;
  for (unsigned i = 0; i < rows; ++i)
    off_diag_sum -= top(i, 0) * diff(i, 0);

  Eigen::MatrixXf dEdx = Eigen::MatrixXf::Zero(rows, 1);
  for (unsigned i = 0; i < rows; ++i)
    dEdx(i, 0) = (off_diag_sum + diff(i, 0)) * top(i, 0);
  return dEdx;
}

} // namespace cnn

