#ifndef CNN_DEVICES_H
#define CNN_DEVICES_H

#include <string>
#include "cnn/aligned-mem-pool.h"

namespace cnn {

enum class DeviceType {CPU, GPU};

static const unsigned ALIGN = 6;
class Device {
 protected:
  Device(DeviceType t, MemAllocator* m) : type(t), mem(m) {}
  virtual ~Device();
 public:
  DeviceType type;
  MemAllocator* mem;
  AlignedMemoryPool<ALIGN>* fxs;
  AlignedMemoryPool<ALIGN>* dEdfs;
  AlignedMemoryPool<ALIGN>* ps;
  float* kSCALAR_MINUSONE;
  float* kSCALAR_ONE;
  float* kSCALAR_ZERO;
  std::string name;
};

#if HAVE_CUDA
class Device_GPU : public Device {
 public:
  explicit Device_GPU(int mb, int device_id);
  ~Device_GPU();
  int cuda_device_id;
  GPUAllocator gpu_mem;
};
#endif

class Device_CPU : public Device {
 public:
  explicit Device_CPU(int mb, bool shared);
  ~Device_CPU();
  CPUAllocator cpu_mem;
  MemAllocator* shmem;
};

} // namespace cnn

#endif
