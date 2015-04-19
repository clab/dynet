#include "cnn/backends/eigen/eigen-backend.h"

#include <iostream>
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

struct FSigmoid {
  inline float operator()(float x) const {
    return 1.f / (1.f + expf(-x));
  }
};

Eigen::MatrixXf Elewise::SigmoidForward(const Eigen::MatrixXf& x) {
  return x.unaryExpr(FSigmoid());
}

struct FSigmoidBackward {
  inline float operator()(float t, float d) const {
    return (1.f - t) * t * d;
  }
};

Eigen::MatrixXf Elewise::SigmoidBackward(const Eigen::MatrixXf& diff, const Eigen::MatrixXf& top, const Eigen::MatrixXf& bottom) {
  return top.binaryExpr(diff, FSigmoidBackward());
}

struct FRectify {
  inline float operator()(float x) const {
    return (x > 0) ? x : 0;
  }
};

Eigen::MatrixXf Elewise::ReluForward(const Eigen::MatrixXf& x) {
  return x.unaryExpr(FRectify());
}

struct FRectifyBackward {
  inline float operator()(float t, float d) const {
    return (t) ? d : 0.f;
  }
};

Eigen::MatrixXf Elewise::ReluBackward(const Eigen::MatrixXf& diff, const Eigen::MatrixXf& top, const Eigen::MatrixXf& bottom) {
  return top.binaryExpr(diff, FRectifyBackward());
}

Eigen::MatrixXf Elewise::TanhForward(const Eigen::MatrixXf& x) {
  return x.unaryExpr(ptr_fun(tanhf));
}

struct FTanhBackward {
  inline float operator()(float t, float d) const {
    return (1.f - t * t) * d;
  }
};

Eigen::MatrixXf Elewise::TanhBackward(const Eigen::MatrixXf& diff, const Eigen::MatrixXf& top, const Eigen::MatrixXf& bottom) {
  return top.binaryExpr(diff, FTanhBackward());
}

inline float logsumexp(const Eigen::MatrixXf& x) {
  const float m = x.maxCoeff();
  float z = 0;
  for (unsigned i = 0; i < x.rows(); ++i)
    z += expf(x(i,0) - m);
  return m + logf(z);
}

struct FSoftmaxNormalize {
  explicit FSoftmaxNormalize(float logz) : logz(logz) {}
  inline float operator()(float x) const {
    return expf(x - logz);
  }
  float logz;
};

Eigen::MatrixXf Convolution::SoftmaxForward(const Eigen::MatrixXf& src, SoftmaxAlgorithm algorithm) {
  if (src.cols() == 1) {
    return src.unaryExpr(FSoftmaxNormalize(logsumexp(src)));
  } else {
    cerr << "SoftmaxForward not implemented for multiple columns\n";
    abort();
  }
}

struct FSoftmaxBackward {
  explicit FSoftmaxBackward(float off_diag_sum) : off_diag_sum(off_diag_sum) {}
  inline float operator()(float t, float d) const {
    return (off_diag_sum + d) * t;
  }
  float off_diag_sum;
};

Eigen::MatrixXf Convolution::SoftmaxBackward(const Eigen::MatrixXf& diff, const Eigen::MatrixXf& top, SoftmaxAlgorithm algorithm) {
  // d softmax(x)_i / d x_j = softmax(x)_i * (1 - softmax(x)_i) if i == j
  // d softmax(x)_i / d x_j = -softmax(x)_i * softmax(x)_j if i != j
  if (top.cols() == 1) {
    float off_diag_sum = -top.cwiseProduct(diff).sum();
    return top.binaryExpr(diff, FSoftmaxBackward(off_diag_sum));
  } else {
    cerr << "SoftmaxBackward not implemented for multiple columns\n";
    abort();
  }
}

} // namespace cnn

