#ifndef DYNET_ALIGNED_MEM_POOL_H
#define DYNET_ALIGNED_MEM_POOL_H

#include <iostream>
#include "dynet/mem.h"

namespace dynet {

class AlignedMemoryPool {
 public:
  explicit AlignedMemoryPool(size_t cap, MemAllocator* a) : a(a) {
    sys_alloc(cap);
    zero_all();
  }

  void* allocate(size_t n) {
    auto rounded_n = a->round_up_align(n);
    if (rounded_n + used > capacity) {
      std::cerr << "dynet is out of memory, try increasing with --dynet-mem (current capacity: " << capacity << ")\n";
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

  size_t used;
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
  MemAllocator* a;
  void* mem;
};

} // namespace dynet

#endif
