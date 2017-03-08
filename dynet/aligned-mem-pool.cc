#include "aligned-mem-pool.h"

#include <sstream>

using namespace dynet;

void* InternalMemoryPool::allocate(size_t n) {
  auto rounded_n = a->round_up_align(n);
  if (rounded_n + used > capacity) {
    return 0;
  }
  void* res = static_cast<char*>(mem) + used;
  used += rounded_n;
  return res;
}

void InternalMemoryPool::sys_alloc(size_t cap) {
  capacity = a->round_up_align(cap);
  mem = a->malloc(capacity);
  if (!mem) {
    std::ostringstream oss; oss << name << " failed to allocate " << capacity;
    throw std::runtime_error(oss.str());
  }
  used = 0;
}
