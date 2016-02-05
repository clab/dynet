#ifndef CNN_DEVICES_H
#define CNN_DEVICES_H

#include <string>
#include "cnn/aligned-mem-pool.h"
#include "cnn/cuda.h"

namespace cnn {

enum class DeviceType {CPU, GPU};

class Device {
 protected:
  Device(DeviceType t, MemAllocator* m) : type(t), mem(m) {}
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  virtual ~Device();
 public:
  DeviceType type;
  MemAllocator* mem;
  AlignedMemoryPool* fxs;
  AlignedMemoryPool* dEdfs;
  AlignedMemoryPool* ps;
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
  cublasHandle_t cublas_handle;
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
