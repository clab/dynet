#ifndef DYNET_ALIGNED_MEM_POOL_H
#define DYNET_ALIGNED_MEM_POOL_H

#include <iostream>
#include "dynet/mem.h"
#include "dynet/globals.h"
#include "dynet/except.h"

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
    explicit AlignedMemoryPool(const std::string &name, size_t initial_cap, MemAllocator *a, size_t expanding_unit = 1<<24);
    ~AlignedMemoryPool();

    void* allocate(size_t n);

    void free();

    void zero_allocated_memory();

    size_t used();
    void set_used(size_t s);
    size_t get_cap();

  private:
    std::string name;
    std::vector<InternalMemoryPool *> pools;
    size_t cap;
    int current;
    MemAllocator* a;
    size_t expanding_unit;
};

} // namespace dynet

#endif
