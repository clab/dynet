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
  std::cout << "sys_alloc " << cap << std::endl;
  capacity = a->round_up_align(cap);
  mem = a->malloc(capacity);
  std::cout << "mem:" << mem << std::endl;
  if (mem == NULL) {
    std::ostringstream oss; oss << name << " failed to allocate " << capacity;
    throw std::runtime_error(oss.str());
  }
  used = 0;
  std::cout << "sys_alloc done" << std::endl;
}
