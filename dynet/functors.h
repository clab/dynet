// DEPRECATED FILE.
// In general, you don't want to add anything to this file since following this
// pattern will generate slow code on the CPU (since it will not generate SIMD
// code). The preferred DyNet style is to use Eigen expression templates. If
// these are not available, you should write SIMD-compatible functors (see
// simd-functors.h).

#ifndef DYNET_GPU_FUNCTORS_H
#define DYNET_GPU_FUNCTORS_H

#include <cstdint>
#include <cmath>
#include <limits>

#if HAVE_CUDA
#  define DYNET_DEVICE_FUNC __device__
#  define DYNET_DEVICE_MIN 1.175494351e-38f
#else
#  define DYNET_DEVICE_FUNC
#  define DYNET_DEVICE_MIN std::numeric_limits<float>::min()
#endif

// these functions are used both in CPU and in GPU computation
// this file may be compiled with NVCC or a standard C++ tool.
// if you need a new elementwise (nullary, unary, binary...)
// functor, this is the place for it
//
// note: also see xfunctors.h - functors implemented there can
// use Eigen's internal support for vectorized operations which
// can give faster performance on some hardware

namespace dynet {

struct FL1Backward {
  FL1Backward(float d) : d(d) {}
  DYNET_DEVICE_FUNC inline float operator()(float x) const {
    return ((0.f < x) - (x < 0.f)) * d;
  }
  const float d;
};

struct FMaxBackwardInv {
  DYNET_DEVICE_FUNC inline float operator()(float u, float d) const {
    return (1.f - u) * d;
  }
};

struct FPairwiseRankLoss {
  FPairwiseRankLoss(float m) : margin(m) {}
  DYNET_DEVICE_FUNC float operator()(float a, float b) const {
    float d = margin - a + b;
    return d > 0.f ? d : 0.f;
  }
  float margin;
};

struct FSoftSign {
  DYNET_DEVICE_FUNC inline float operator()(float x) const {
    return x / (1.f + (x < 0.f ? -x : x));
  }
};

struct FSoftSignBackward {
  DYNET_DEVICE_FUNC inline float operator()(float t, float d) const {
    float a = 1.f - (t < 0.f ? -t : t);
    return a * a * d;
  }
};

struct FSqDist {
  DYNET_DEVICE_FUNC inline float operator()(float a, float b) const {
    float d = a - b;
    return d * d;
  }
};

struct FEuclideanBackward {
  FEuclideanBackward(int i, const float* s) : i(i), scalar(s) {}
  DYNET_DEVICE_FUNC inline float operator()(float a, float b) const {
    return (i == 0 ? 2.f : -2.f) * (*scalar) * (a - b);
  }
  int i;
  const float* scalar;
};

struct FL2SGDUpdate {
  FL2SGDUpdate(float l, float s) : lambda(l), scale(-s) {}
  DYNET_DEVICE_FUNC inline float operator()(float x, float g) const {
    return scale * g - x * lambda;
  }
  float lambda;
  float scale;
};

struct FBinaryLogLoss {
  DYNET_DEVICE_FUNC inline float operator()(float x, float x_true) const {
    if (x_true == 1.f) {
      if (x == 0.f) return -1.f * log(DYNET_DEVICE_MIN);
      return -1.f * log(x);
    }
    else if (x_true == 0.f) {
      if (x == 1.f) return -1.f * log(DYNET_DEVICE_MIN);
      else return (x_true - 1.f) * log1pf(-x);
    }
    else {
      if (x == 0.f) return -1.f * log(DYNET_DEVICE_MIN);
      else if (x == 1.f) return -1.f * log(DYNET_DEVICE_MIN);
      else return -1.f * (x_true * log(x) + (1.f - x_true) * log1pf(-x));
    }
  }
};

struct FBinaryLogLossBackward {
  explicit FBinaryLogLossBackward(float d) : d(d) {}
  DYNET_DEVICE_FUNC inline float operator()(float x, float x_true) const {
    if (x == x_true) return 0;
    if (x == 0.f) x = DYNET_DEVICE_MIN;
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

struct FELUForward {
  explicit FELUForward(float alpha, float lambda) : alpha(alpha), lambda(lambda) {}
  DYNET_DEVICE_FUNC inline float operator()(float x) const {
    return lambda * ((x > 0.f) ? x : alpha * (expm1f(x)));
  }
  float alpha, lambda;
};

struct FELUBackward {
  explicit FELUBackward(float alpha, float lambda) : alpha(alpha), lambda(lambda) {}
  DYNET_DEVICE_FUNC inline float operator()(float x, float d) const {
    return d * ((x > 0.f) ? lambda : lambda * alpha * expf(x));
  }
  float alpha, lambda;
};

struct FSILUForward {
  explicit FSILUForward(float beta) : beta(beta) {}
  DYNET_DEVICE_FUNC inline float operator()(float x) const {
    return x * (0.5 + 0.5 * tanh(beta * x * 0.5));
  }
  float beta;
};

struct FSILUBackward {
  explicit FSILUBackward(float beta) : beta(beta) {}
  DYNET_DEVICE_FUNC inline float operator()(float x, float d) const {
    float l = (0.5 + 0.5 * tanh(beta * x * 0.5));
    return (l + x * l * (1 - l)) * d;
  }
  float beta;
};


} // namespace dynet

#endif
