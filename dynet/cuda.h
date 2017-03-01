#ifndef DYNET_CUDA_H
#define DYNET_CUDA_H
#if HAVE_CUDA

#include <vector>
#include <cassert>
#include <utility>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "dynet/except.h"


#define MAX_GPUS 256

#define CUDA_CHECK(stmt) do {                              \
    cudaError_t err = stmt;                                \
    if (err != cudaSuccess) {                              \
      std::cerr << "CUDA failure in " << #stmt << std::endl\
                << cudaGetErrorString(err) << std::endl;   \
      throw dynet::cuda_exception(#stmt);                  \
    }                                                      \
  } while(0)

#define CUBLAS_CHECK(stmt) do {                            \
    cublasStatus_t stat = stmt;                            \
    if (stat != CUBLAS_STATUS_SUCCESS) {                   \
      std::cerr << "CUBLAS failure in " << #stmt           \
                << std::endl << stat << std::endl;         \
      throw dynet::cuda_exception(#stmt);                  \
    }                                                      \
  } while(0)



namespace dynet {

struct DynetParams;


class Device;

inline std::pair<int, int> SizeToBlockThreadPair(int n) {
  assert(n > 0);
  int logn;
#if defined(_MSC_VER)
  logn = 0;
  if (n > 2) {
    int localN = n - 1;
    while (localN >>= 1) 
      logn++;
  }
#else
  asm("\tbsr %1, %0\n"
      : "=r"(logn)
      : "r" (n-1));
#endif
  logn = logn > 9 ? 9 : (logn < 4 ? 4 : logn);
  ++logn;
  int threads = 1 << logn;
  int blocks = (n + threads - 1) >> logn;
  blocks = blocks > 65535 ? 65535 : blocks;
  return std::make_pair(blocks, threads);
}

std::vector<Device*> initialize_gpu(dynet::DynetParams& params);
std::vector<Device*> initialize_gpu(int& argc, char**& argv);

} // namespace dynet

#endif
#endif
