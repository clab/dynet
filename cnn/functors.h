#ifndef CNN_GPU_FUNCTORS_H
#define CNN_GPU_FUNCTORS_H

#if HAVE_CUDA
#  define CNN_DEVICE_FUNC __device__
#else
#  define CNN_DEVICE_FUNC
#endif

// these functions are used both in CPU and in GPU computation
// this file may be compiled with NVCC or a standard C++ tool.
// if you need a new elementwise (nullary, unary, binary...)
// functor, this is the place for it

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

struct FHuberBackward {
  FHuberBackward(float c) : c(c) {}
  CNN_DEVICE_FUNC inline float operator()(float x, float d) const {
    const float a = fabs(x);
    return (2 * d) * ((a < c) ? x : c * sgn(x));
  }
  const float c;
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
    return x_true > 0.f ? -x_true * log(x) : (1.f - x_true) * log1p(-x);
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
