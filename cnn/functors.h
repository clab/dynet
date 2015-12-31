#ifndef CNN_GPU_FUNCTORS_H
#define CNN_GPU_FUNCTORS_H

#include <cstdint>
#include <limits>

#if HAVE_CUDA
#  define CNN_DEVICE_FUNC __device__
#  define CNN_DEVICE_MIN -1.175494351e-38f
#else
#  include <boost/math/special_functions/digamma.hpp>
#  define CNN_DEVICE_FUNC
#  define CNN_DEVICE_MIN std::numeric_limits<float>::min()
#endif

// these functions are used both in CPU and in GPU computation
// this file may be compiled with NVCC or a standard C++ tool.
// if you need a new elementwise (nullary, unary, binary...)
// functor, this is the place for it
//
// note: also see xfunctors.h - functors implemented there can
// use Eigen's internal support for vectorized operations which
// can give faster performance on some hardware

namespace cnn {

struct FHuberForward {
  FHuberForward(float c) : c(c) {}
  CNN_DEVICE_FUNC inline float operator()(float x) const {
    const float a = fabs(x);
    return (a < c) ? x*x : c*(2*a - c);
  }
  const float c;
};

template <typename T> int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

struct FL1Backward {
  FL1Backward(float d) : d(d) {}
  CNN_DEVICE_FUNC inline float operator()(float x) const {
    return sgn(x) * d;
  }
  const float d;
};

struct FHuberBackward {
  FHuberBackward(float c, float dEdf) : c(c), d(dEdf) {}
  CNN_DEVICE_FUNC inline float operator()(float x) const {
    const float a = fabs(x);
    return (2 * d) * ((a < c) ? x : c * sgn(x));
  }
  const float c;
  const float d;
};

struct FProduct {
  CNN_DEVICE_FUNC inline float operator()(float a, float b) const {
    return a * b;
  }
};

struct FQuotient {
  CNN_DEVICE_FUNC inline float operator()(float a, float b) const {
    return a / b;
  }
};

struct FConstantPlus {
  FConstantPlus(float c) : c(c) {}
  CNN_DEVICE_FUNC inline float operator()(float x) const {
    return c + x;
  }
  float c;
};

struct FConstantMinus {
  FConstantMinus(float c) : c(c) {}
  CNN_DEVICE_FUNC inline float operator()(float x) const {
    return c - x;
  }
  float c;
};

struct FNegate {
  CNN_DEVICE_FUNC inline float operator()(float x) const {
    return -x;
  }
};

struct FErf {
  CNN_DEVICE_FUNC inline float operator()(float x) const {
    return erff(x);
  }
};

struct FTanh {
  CNN_DEVICE_FUNC inline float operator()(float x) const {
#ifdef FAST_TANH
    float x2 = x * x;
    float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    return a / b;
#else
     return tanhf(x);
#endif
  }
};

struct FLog {
  CNN_DEVICE_FUNC inline float operator()(float x) const {
    return logf(x);
  }
};

struct FMaxBackwardInv {
  CNN_DEVICE_FUNC inline float operator()(float u, float d) const {
    return (1.f - u) * d;
  }
};

struct FSqrtBackward {
  CNN_DEVICE_FUNC inline float operator()(float t, float d) const {
    return d / (2.f * t);
  }
};

struct FErfBackward {
  CNN_DEVICE_FUNC inline float operator()(float x, float d) const {
    return 1.1283791670955125738961589f * expf(-x * x) * d;
  }
};

struct FTanhBackward {
  CNN_DEVICE_FUNC inline float operator()(float t, float d) const {
    return (1.f - t * t) * d;
  }
};

struct FLogBackward {
  CNN_DEVICE_FUNC inline float operator()(float t, float d) const {
    return (1.f / t) * d;
  }
};

struct FPairwiseRankLoss {
  FPairwiseRankLoss(float m) : margin(m) {}
  CNN_DEVICE_FUNC float operator()(float a, float b) const {
    float d = margin - a + b;
    return d > 0.f ? d : 0.f;
  }
  float margin;
};

struct FRectifyBackward {
  CNN_DEVICE_FUNC inline float operator()(float t, float d) const {
    return (t) ? d : 0.f;
  }
};

struct FRectifyNegateBackward {
  CNN_DEVICE_FUNC inline float operator()(float t, float d) const {
    return (t) ? -d : 0.f;
  }
};

struct FSoftmaxNormalize {
  explicit FSoftmaxNormalize(float logz) : logz(logz) {}
  CNN_DEVICE_FUNC inline float operator()(float x) const {
    return expf(x - logz);
  }
  float logz;
};

struct FSoftmaxBackward {
  explicit FSoftmaxBackward(float off_diag_sum) : off_diag_sum(off_diag_sum) {}
  CNN_DEVICE_FUNC inline float operator()(float t, float d) const {
    return (off_diag_sum + d) * t;
  }
  float off_diag_sum;
};

struct FLogGammaBackward {
  CNN_DEVICE_FUNC inline float operator()(float x, float d) const {
#ifndef HAVE_CUDA
    return boost::math::digamma(x) * d;
#else
    assert(false); // Not supported on GPUs?
    return 0;
#endif
  }
};

struct FNegLogSoftmaxBackward {
  FNegLogSoftmaxBackward(float lz, float err) : logz(lz), d(err) {}
  CNN_DEVICE_FUNC inline float operator()(float t) const {
    return expf(t - logz) * d;
  }
  float logz;
  float d;
};

struct FPtrNegLogSoftmaxBackward {
  FPtrNegLogSoftmaxBackward(const float* lz, const float* err) : logz(lz), d(err) {}
  CNN_DEVICE_FUNC inline float operator()(float t) const {
    return expf(t - *logz) * *d;
  }
  const float* logz;
  const float* d;
};

struct FLogSoftmaxNormalize {
  explicit FLogSoftmaxNormalize(float logz) : logz(logz) {}
  CNN_DEVICE_FUNC inline float operator()(float x) const {
    return x - logz;
  }
  float logz;
};

struct FWeightedError {
  float operator()(float t, float d) const {
    return expf(t) * d / expf(t);
  }
};

struct FLogSoftmaxBackward {
  explicit FLogSoftmaxBackward(float off_diag_sum) : off_diag_sum(off_diag_sum) {}
  CNN_DEVICE_FUNC inline float operator()(float t, float d) const {
    return off_diag_sum * expf(t) + d;
    //return (off_diag_sum + d) * t;
  }
  float off_diag_sum;
};

struct FRectify {
  CNN_DEVICE_FUNC inline float operator()(float x) const {
    return (x > 0.f) ? x : 0.f;
  }
};

struct FSoftSign {
  CNN_DEVICE_FUNC inline float operator()(float x) const {
    return x / (1.f + (x < 0.f ? -x : x));
  }
};

struct FSoftSignBackward {
  CNN_DEVICE_FUNC inline float operator()(float t, float d) const {
    float a = 1.f - (t < 0.f ? -t : t);
    return a * a * d;
  }
};

struct FLogisticSigmoid {
  CNN_DEVICE_FUNC inline float operator()(float x) const {
    return 1.f / (1.f + expf(-x));
  }
};

struct FLogisticSigmoidBackward {
  CNN_DEVICE_FUNC inline float operator()(float t, float d) const {
    return (1.f - t) * t * d;
  }
};

struct FSqDist {
  CNN_DEVICE_FUNC inline float operator()(float a, float b) const {
    float d = a - b;
    return d * d;
  }
};

struct FEuclideanBackward {
  FEuclideanBackward(int i, const float* s) : i(i), scalar(s) {}
  CNN_DEVICE_FUNC inline float operator()(float a, float b) const {
    return (i == 0 ? 2.f : -2.f) * (*scalar) * (a - b);
  }
  int i;
  const float* scalar;
};

struct FL2SGDUpdate {
  FL2SGDUpdate(float l, float s) : lambda(l), scale(-s) {}
  CNN_DEVICE_FUNC inline float operator()(float x, float g) const {
    return scale * g - x * lambda;
  }
  float lambda;
  float scale;
};

struct FBinaryLogLoss {
  CNN_DEVICE_FUNC inline float operator()(float x, float x_true) const {
    if (x_true == 1.f) {
      if (x == 0.f) x = CNN_DEVICE_MIN;
      return -1.f * x_true * log(x);
    }
    else if (x_true == 0.f) {
      if (x == 1.f) x = CNN_DEVICE_MIN;
      return (x_true - 1.f) * log1p(-x);
    }
    else {
      if (x == 0.f) x = CNN_DEVICE_MIN;
      if (x == 1.f) x = CNN_DEVICE_MIN;
      return -1.f * (x_true * log(x) + (1.f - x_true) * log1p(-x));
    }
  }
};

struct FBinaryLogLossBackward {
  explicit FBinaryLogLossBackward(float d) : d(d) {}
  CNN_DEVICE_FUNC inline float operator()(float x, float x_true) const {
    if (x == x_true) return 0;
    if (x == 0.f) x = CNN_DEVICE_MIN;
    if (x == 1.f) x = 0.9999999f;
    if (x_true == 1.f) {
      return d * -x_true / x;
    } else if (x_true == 0.f) {
      return d * (1.f - x_true) / (1.f - x);
    }
    return d * ((1.f - x_true) / (1.f - x) + (-x_true / x));
  }
  float d;
};

} // namespace cnn

#endif
