#ifndef CNN_DEVICES_H
#define CNN_DEVICES_H

#include <string>
#include "cnn/aligned-mem-pool.h"
#include "cnn/cuda.h"

namespace Eigen {
  struct DefaultDevice;
  class CudaStreamDevice;
  struct GpuDevice;
}

namespace cnn {

enum class DeviceType {CPU, GPU};
enum class DeviceMempool {FXS = 0, DEDFS = 1, PS = 2, NONE = 3};

struct ComputationGraph; // TODO is there a nicer way to resolve this cyclic dependency?
struct Tensor;

struct DeviceMemCheckpoint {
  std::vector<size_t> used;
  // The three devices are those defined in DeviceMempool
  DeviceMemCheckpoint() : used(3) { }
};


class Device {
 protected:
  Device(int i, DeviceType t, MemAllocator* m) : device_id(i), type(t), mem(m), pools(3, nullptr) {}
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  virtual ~Device();
 public:
  int device_id;
  DeviceType type;
  MemAllocator* mem;
  float* kSCALAR_MINUSONE;
  float* kSCALAR_ONE;
  float* kSCALAR_ZERO;
  std::string name;
  virtual DeviceMemCheckpoint mark(ComputationGraph *cg);
  virtual void revert(DeviceMemCheckpoint cp);
  void allocate_tensor(DeviceMempool mem_pool, Tensor & tensor);
  std::vector<AlignedMemoryPool*> pools;
};

#if HAVE_CUDA
class Device_GPU : public Device {
 public:
  typedef Eigen::CudaStreamDevice EigenDevice;
  explicit Device_GPU(int my_id, int mb, int device_id);
  ~Device_GPU();
  int cuda_device_id;
  cublasHandle_t cublas_handle;
  Eigen::GpuDevice* edevice;
  Eigen::CudaStreamDevice* estream;
  GPUAllocator gpu_mem;
};
#endif

class Device_CPU : public Device {
 public:
  typedef Eigen::DefaultDevice EigenDevice;
  explicit Device_CPU(int my_id, int mb, bool shared);
  ~Device_CPU();
  CPUAllocator cpu_mem;
  Eigen::DefaultDevice* edevice;
  MemAllocator* shmem;
};

} // namespace cnn

#endif
