#include "aligned-mem-pool.h"

#include <sstream>

using namespace dynet;

void* AlignedMemoryPool::allocate(size_t n) {
  auto rounded_n = a->round_up_align(n);
  if (rounded_n + used > capacity) {
    std::ostringstream oss; oss << name << " is out of memory, try increasing with --dynet-mem (current capacity: " << capacity << "). See http://dynet.readthedocs.io/en/latest/commandline.html for details.";
    throw std::runtime_error(oss.str());
  }
  void* res = static_cast<char*>(mem) + used;
  used += rounded_n;
  return res;
}

void AlignedMemoryPool::sys_alloc(size_t cap) {
  capacity = a->round_up_align(cap);
  mem = a->malloc(capacity);
  if (!mem) {
    std::ostringstream oss; oss << name << " failed to allocate " << capacity;
    throw std::runtime_error(oss.str());
  }
  used = 0;
}
