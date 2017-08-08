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

    void report_self(){
      size_t cur_used = used();
      size_t suggesting = (peak_used>>20);
      if(suggesting == 0) // never recorded (no clearing) before
        suggesting = (cap>>20);
      std::cerr << "[dynet-mem-test] " << name << ", Previously (in MB): " << "#pools/#last-used/#peak-used/#last-cap = " << pools.size() << "/" << (cur_used>>20) << "/" << (peak_used>>20) << "/" << (cap>>20) << "; suggesting " << suggesting << "MB." << std::endl;
    }

  private:
    std::string name;
    std::vector<InternalMemoryPool *> pools;
    int current;
    size_t cap;
    MemAllocator* a;
    size_t expanding_unit;

    // recordings
    size_t peak_used;	// maximum of used memory for all time
};

} // namespace dynet

#endif
