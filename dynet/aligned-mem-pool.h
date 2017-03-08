#ifndef DYNET_ALIGNED_MEM_POOL_H
#define DYNET_ALIGNED_MEM_POOL_H

#include <iostream>
#include "dynet/mem.h"
#include "dynet/globals.h"

namespace dynet {

class InternalMemoryPool {
 public:
  explicit InternalMemoryPool(const std::string & name, size_t cap, MemAllocator* a) : name(name), a(a) {
    sys_alloc(cap);
    zero_all();
  }

  ~InternalMemoryPool() {
      a->free(mem);
  }

  void* allocate(size_t n); 

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
  void sys_alloc(size_t cap);

  void zero_all() {
    a->zero(mem, capacity);
  }
  std::string name;
  size_t capacity;
  MemAllocator* a;
  void* mem;
};

class AlignedMemoryPool {
  public:
    explicit AlignedMemoryPool(const std::string &name, size_t cap, MemAllocator *a) : name(name), current(0), cap(cap), a(a) {
      pools.push_back(new InternalMemoryPool(name, cap, a));
    }
    ~AlignedMemoryPool() {
      for ( auto p : pools) { delete p; }
    }

    void* allocate(size_t n) {
      void *res = pools[current]->allocate(n);
      if (res == 0) {
        pools.push_back(new InternalMemoryPool(name, cap, a));
        current++;
        res = pools[current]->allocate(n);
      }
      return res;
    }

    void free() {
      if (current > 0) {
        for (auto p : pools) { delete p; }
        pools.clear();
        pools.push_back(new InternalMemoryPool(name, cap * (current+1), a));
        current = 0;
      } 
      pools[0]->free();
    }

    void zero_allocated_memory() {
      for (auto p : pools) { p->zero_allocated_memory(); }
    }

    
    size_t used() {
      if (current == 0) {
        return pools[0]->used;
      }
      size_t res = 0;
      for (auto p : pools) { res += p->used; }
      return res;
    }

    void set_used(size_t s) {
      int c = 0;
      while (s > pools[c]->used) {
        s -= pools[c]->used;
        c++;
        DYNET_ASSERT(c <= current, "attempt to set_used to a larger value than used()."); // TODO is that the way to use the assert?
      } 
      // s <= pools[c]->used
      pools[c]->used = s;
      current = c;
    }

  private:
    std::string name;
    std::vector<InternalMemoryPool *> pools;
    int current;
    size_t cap;
    MemAllocator* a;
};

} // namespace dynet

#endif
