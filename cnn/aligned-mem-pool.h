#ifndef CNN_ALIGNED_MEM_POOL_H
#define CNN_ALIGNED_MEM_POOL_H

#include <iostream>
#include "cnn/mem.h"

namespace cnn {

class AlignedMemoryPool {
 public:
  explicit AlignedMemoryPool(size_t cap, MemAllocator* a) : a(a) {
    sys_alloc(cap);
    zero_all();
  }

  void* allocate(size_t n) {
    auto rounded_n = a->round_up_align(n);
    if (rounded_n + used > capacity) {
      std::cerr << "cnn is out of memory, try increasing with --cnn-mem\n";
      abort();
    }
    void* res = static_cast<char*>(mem) + used;
    used += rounded_n;
    return res;
  }
  void free() {
    //std::cerr << "freeing " << used << " bytes\n";
    used = 0;
  }
  // zeros out the amount of allocations
  void zero_allocated_memory() {
    if (used == 0) return;
    a->zero(mem, used);
  }

  bool is_shared() {
    return shared;
  }
 private:
  void sys_alloc(size_t cap) {
    capacity = a->round_up_align(cap);
    //std::cerr << "Allocating " << capacity << " ...\n";
    mem = a->malloc(capacity);
    if (!mem) { std::cerr << "Failed to allocate " << capacity << std::endl; abort(); }
    used = 0;
  }
  void zero_all() {
    a->zero(mem, capacity);
  }
  size_t capacity;
  size_t used;
  bool shared;
  MemAllocator* a;
  void* mem;
};

} // namespace cnn

#endif
