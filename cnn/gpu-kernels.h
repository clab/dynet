#ifndef CNN_GPU_KERNELS_H
#define CNN_GPU_KERNELS_H

#include "cnn/cuda.h"

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
__global__ void accUnaryExprKernel(int n, const float* x, float* y, Func func) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] += func(x[i]);
    i += gridDim.x * blockDim.x;
  }
}

template<typename Func>
__global__ void binaryExprKernel(int n, const float* x0, const float* x1, float* y, Func func) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] = func(x0[i], x1[i]);
    i += gridDim.x * blockDim.x;
  }
}

template<typename Func>
__global__ void accBinaryExprKernel(int n, const float* x0, const float* x1, float* y, Func func) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] += func(x0[i], x1[i]);
    i += gridDim.x * blockDim.x;
  }
}

template<typename Func>
__global__ void slowReduceKernel(int n, float* x, float* y, float* r, Func func) {
  float tr = 0;
  for (int i = 0; i < n; ++i)
    tr += func(x[i], y[i]);
  r[0] = tr;
}

} // namespace gpu
} // namespace cnn

#endif
