#ifndef CNN_GPU_FUNCTORS_H
#define CNN_GPU_FUNCTORS_H

#include <cstdint>

#if HAVE_CUDA
#  define CNN_DEVICE_FUNC __device__
#else
#  define CNN_DEVICE_FUNC
#endif

// these functions are used both in CPU and in GPU computation
// this file may be compiled with NVCC or a standard C++ tool.
// if you need a new elementwise (nullary, unary, binary...)
// functor, this is the place for it

#define cast_uint32_t static_cast<uint32_t>

// THIS CODE IS BROKEN- sometimes it returns NaN
// it is commented out for this reason
static inline float fastpow2 (float p) {
  float offset = (p < 0) ? 1.0f : 0.0f;
  float clipp = (p < -126) ? -126.0f : p;
  int w = clipp;
  float z = clipp - w + offset;
  union { uint32_t i; float f; } v = { cast_uint32_t ( (1 << 23) * (clipp + 121.2740575f + 27.7280233f / (4.84252568f - z) - 1.49012907f * z) ) };

  return v.f;
}

#if 1
#if 0
static inline float fastexp (float p) {
  return fastpow2 (1.442695040f * p);
}
#else
static inline float fastexp (float p) {
  return exp(p);
}
#endif
#else
// Schraudolph version, but it's a bit crappy in terms of
// performance and not that much faster
#define EXPAF (8388608 / 0.6931471806f)
static inline float fastexp (float p) {
  union { float f; int32_t i; } eco;
  eco.i = (int32_t)(EXPAF * (p)) + 1065353216;
  return eco.f;
}
#endif

#if defined(__GNU_LIBRARY__) && (__GLIBC__ == 2) && (__GLIBC_MINOR__ < 14) && !defined(HAVE_CUDA)
#define USE_FASTEXP
#else
#undef USE_FASTEXP
#endif

#ifdef USE_FASTEXP
#define CNN_EXPF fastexp
#else
#define CNN_EXPF expf
#endif

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

struct FMaxBackwardInv {
  CNN_DEVICE_FUNC inline float operator()(float u, float d) const {
    return (1.f - u) * d;
  }
};

struct FTanhBackward {
  CNN_DEVICE_FUNC inline float operator()(float t, float d) const {
    return (1.f - t * t) * d;
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
    return CNN_EXPF(x - logz);
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

struct FNegLogSoftmaxBackward {
  FNegLogSoftmaxBackward(float lz, float err) : logz(lz), d(err) {}
  CNN_DEVICE_FUNC inline float operator()(float t) const {
    return CNN_EXPF(t - logz) * d;
  }
  float logz;
  float d;
};

struct FPtrNegLogSoftmaxBackward {
  FPtrNegLogSoftmaxBackward(const float* lz, const float* err) : logz(lz), d(err) {}
  CNN_DEVICE_FUNC inline float operator()(float t) const {
    return CNN_EXPF(t - *logz) * *d;
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
    return CNN_EXPF(t) * d / CNN_EXPF(t);
  }
};

struct FLogSoftmaxBackward {
  explicit FLogSoftmaxBackward(float off_diag_sum) : off_diag_sum(off_diag_sum) {}
  CNN_DEVICE_FUNC inline float operator()(float t, float d) const {
    return off_diag_sum * CNN_EXPF(t) + d;
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
    return 1.f / (1.f + CNN_EXPF(-x));
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
      return -1.f * x_true * log(x);
    }
    else if (x_true == 0.f) {
      return -1.f * (1.f - x_true) * log1p(-x);
    }
    else {
      return -1.f * (x_true * log(x) + (1.f - x_true) * log1p(-x));
    }
  }
};

struct FBinaryLogLossBackward {
  CNN_DEVICE_FUNC inline float operator()(float x, float x_true, float d) const {
    float scale = (x_true > 0.f) ? -x_true/x : (1.f-x_true)/(1.-x);
    return d * scale;
  }
};

} // namespace cnn

#endif
