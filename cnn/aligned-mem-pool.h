#ifndef CNN_ALIGNED_MEM_POOL_H
#define CNN_ALIGNED_MEM_POOL_H

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

namespace cnn {

inline void* cnn_mm_malloc(size_t n, size_t align) {
  void* ptr = nullptr;
#if HAVE_CUDA
  CUDA_CHECK(cudaMalloc(&ptr, n));
#else
  ptr = _mm_malloc(n, align);
#endif
  if (!ptr) {
    std::cerr << "Memory allocation failed n=" << n << " align=" << align << std::endl;
    throw cnn::out_of_memory("Memory allocation failed in cnn_mm_malloc()");
  }
  return ptr;
}

inline void cnn_mm_free(void* mem) {
//#if HAVE_MM_MALLOC
#if HAVE_CUDA
  CUDA_CHECK(cudaFree(mem));
#else
  _mm_free(mem);
#endif

//#else
//  return std::free(n, align);
//#endif
}

// this is used to manage CPU memory for function values and gradients
template <unsigned AlignedBits>
class AlignedMemoryPool {
 public:
  explicit AlignedMemoryPool(size_t cap, bool shared = false) : shared(shared) {
    sys_alloc(cap);
    zero_all();
  }

  // returns nullptr if OOM
  void* allocate(size_t n) {
    auto rounded_n = round_up_align(n);
    if (rounded_n + used > capacity)
      return nullptr;
    void* res = static_cast<char*>(mem) + used;
    used += rounded_n;
    return res;
  }
  void free() {
    //std::cerr << "freeing " << used << " bytes\n";
    used = 0;
  }
  void free_and_grow_capacity(size_t new_cap = 0) {
    cnn_mm_free(mem);
    if (new_cap)
      sys_alloc(new_cap);
    else
      sys_alloc(capacity * 1.5);
    zero_all();
  }
  // zeros out the amount of allocations
  void zero_allocated_memory() {
    if (used == 0) return;
#if HAVE_CUDA
    CUDA_CHECK(cudaMemsetAsync(mem, 0, used));
#else
    std::memset(mem, 0, used);
#endif
  }

  bool is_shared() {
    return shared;
  }
 private:
  void sys_alloc(size_t cap) {
    capacity = round_up_align(cap);
    if (shared) {
      mem = mmap(NULL, capacity, PROT_READ|PROT_WRITE, MAP_ANON|MAP_SHARED, -1, 0);
    }
    else {
      mem = cnn_mm_malloc(capacity, 1 << AlignedBits);
    }
    used = 0;
  }
  void zero_all() {
    //std::cerr << "zeroing " << (used ? used : capacity) << " bytes\n";
#if HAVE_CUDA
    CUDA_CHECK(cudaMemsetAsync(mem, 0, capacity));
#else
    std::memset(mem, 0, capacity);
#endif
  }
  inline static size_t round_up_align(size_t n) {
    if (AlignedBits < 2) return n;
    auto c = (n & ((1 << (AlignedBits)) - 1)) > 0 ? 1 : 0;
    return ((n >> (AlignedBits)) + c) << (AlignedBits);
  }
  size_t capacity;
  size_t used;
  bool shared;
  void* mem;
};

} // namespace cnn

#endif
