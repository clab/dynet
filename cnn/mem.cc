#include "cnn/mem.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/shm.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <mm_malloc.h>
#include "cnn/except.h"
#if HAVE_CUDA
#include "cnn/cuda.h"
#include <cuda.h>
#include <cuda_runtime.h>
#endif

using namespace std;

namespace cnn {

MemAllocator::~MemAllocator() {}

void* CPUAllocator::malloc(size_t n) {
  void* ptr = _mm_malloc(n, align);
  if (!ptr) {
    cerr << "CPU memory allocation failed n=" << n << " align=" << align << endl;
    throw cnn::out_of_memory("CPU memory allocation failed");
  }
  return ptr;
}

void CPUAllocator::free(void* mem) {
  _mm_free(mem);
}

void CPUAllocator::zero(void* p, size_t n) {
  memset(p, 0, n);
}

void* SharedAllocator::malloc(size_t n) {
  void* ptr = mmap(NULL, n, PROT_READ|PROT_WRITE, MAP_ANON|MAP_SHARED, -1, 0);
  if (!ptr) {
    cerr << "Shared memory allocation failed n=" << n << endl;
    throw cnn::out_of_memory("Shared memory allocation failed");
  }
  return ptr;
}

void SharedAllocator::free(void* mem) {
//  munmap(mem, n);
}

void SharedAllocator::zero(void* p, size_t n) {
  memset(p, 0, n);
}

#if HAVE_CUDA
void* GPUAllocator::malloc(size_t n) {
  void* ptr = nullptr;
  CUDA_CHECK(cudaSetDevice(devid));
  CUDA_CHECK(cudaMalloc(&ptr, n));
  if (!ptr) {
    cerr << "GPU memory allocation failed n=" << n << endl;
    throw cnn::out_of_memory("GPU memory allocation failed");
  }
  return ptr;
}

void GPUAllocator::free(void* mem) {
  CUDA_CHECK(cudaFree(mem));
}

void GPUAllocator::zero(void* p, size_t n) {
  CUDA_CHECK(cudaSetDevice(devid));
  CUDA_CHECK(cudaMemsetAsync(p, 0, n));
}

#endif

} // namespace cnn
