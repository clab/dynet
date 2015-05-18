#include "cnn/gpu-ops.h"
#include "cnn/cuda.h"

#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

namespace cnn {
namespace gpu {

template<typename Func>
__global__ void unaryExprKernel(int n, const float* x, float* y, Func func) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] = func(x[i]);
    i += gridDim.x * blockDim.x;
  }
}

template<typename Func>
__global__ void binaryExprAccKernel(int n, const float* x0, const float* x1, float* y, Func func) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] += func(x0[i], x1[i]);
    i += gridDim.x * blockDim.x;
  }
}

struct FTanh {
  __device__ inline float operator()(float x) const { return tanh(x); }
};

void vtanh(int n, float* x, float* y) {
  auto tb = SizeToBlockThreadPair(n);
  unaryExprKernel<<<tb.first, tb.second>>>(n, x, y, FTanh());
}

struct FTanhBackward {
  __device__ inline float operator()(float t, float d) const {
    return (1.f - t * t) * d;
  }
};

void vtanh_backward(int n, const float* fx, const float* dEdf, float* dEdx) {
  auto tb = SizeToBlockThreadPair(n);
  binaryExprAccKernel<<<tb.first, tb.second>>>(n, fx, dEdf, dEdx, FTanhBackward());
}

struct FSqDist {
  __device__ inline float operator()(float a, float b) const {
    float d = a - b;
    return d * d;
  }
};

struct FEuclideanBackward {
  FEuclideanBackward(int i, const float* s) : i(i), scalar(s) {}
  __device__ inline float operator()(float a, float b) const {
    return (i == 0 ? 2.f : -2.f) * (*scalar) * (a - b);
  }
  int i;
  const float* scalar;
};

template<typename Func>
__global__ void slowReduceKernel(int n, float* x, float* y, float* r, Func func) {
  float tr = 0;
  for (int i = 0; i < n; ++i)
    tr += func(x[i], y[i]);
  r[0] = tr;
}

void sqeucdist(int n, float* x, float *y, float* res) {
  slowReduceKernel<<<1,1>>>(n, x, y, res, FSqDist());
}

void sqeucdist_backward(int n, const float* dEdy, const float* x0, const float* x1, float* dEdx, int i) {
  auto tb = SizeToBlockThreadPair(n);
  binaryExprAccKernel<<<tb.first, tb.second>>>(n, x0, x1, dEdx, FEuclideanBackward(i, dEdy));
}

struct FL2SGDUpdate {
  FL2SGDUpdate(float l, float s) : lambda(l), scale(-s) {}
  __device__ inline float operator()(float x, float g) const {
    float r = x * lambda;
    return scale * g - r;
  }
  float lambda;
  float scale;
};

void sgd_update(int n, const float* g, float* x, float scale, float lambda) {
  auto tb = SizeToBlockThreadPair(n);
  binaryExprAccKernel<<<tb.first, tb.second>>>(n, x, g, x, FL2SGDUpdate(lambda, scale));
}

} // namespace gpu
} // namespace cnn
