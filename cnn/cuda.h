#ifndef CNN_CUDA_H
#define CNN_CUDA_H
#if HAVE_CUDA

#include <vector>
#include <cassert>
#include <utility>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cnn/except.h"

#define CUDA_CHECK(stmt) do {                              \
    cudaError_t err = stmt;                                \
    if (err != cudaSuccess) {                              \
      std::cerr << "CUDA failure in " << #stmt << std::endl\
                << cudaGetErrorString(err) << std::endl;   \
      throw cnn::cuda_exception(#stmt);                    \
    }                                                      \
  } while(0)

#define CUBLAS_CHECK(stmt) do {                            \
    cublasStatus_t stat = stmt;                            \
    if (stat != CUBLAS_STATUS_SUCCESS) {                   \
      std::cerr << "CUBLAS failure in " << #stmt           \
                << std::endl << stat << std::endl;         \
      throw cnn::cuda_exception(#stmt);                    \
    }                                                      \
  } while(0)

namespace cnn {

struct Device;

inline std::pair<int,int> SizeToBlockThreadPair(int n) {
  assert(n);
  int logn;
  asm("\tbsr %1, %0\n"
      : "=r"(logn)
      : "r" (n-1));
  logn = logn > 9 ? 9 : (logn < 4 ? 4 : logn);
  ++logn;
  int threads = 1 << logn;
  int blocks = (n + threads - 1) >> logn;
  blocks = blocks > 128 ? 128 : blocks;
  return std::make_pair(blocks, threads);
}

std::vector<Device*> Initialize_GPU(int& argc, char**& argv);
extern cublasHandle_t cublas_handle;

} // namespace cnn

#endif
#endif
