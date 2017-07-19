#ifndef DYNET_DEVICES_H
#define DYNET_DEVICES_H

#define EIGEN_USE_THREADS
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>

#include "dynet/aligned-mem-pool.h"
#include "dynet/cuda.h"

namespace dynet {

enum class DeviceType {CPU, GPU, ThreadPool};
enum class DeviceMempool {FXS = 0, DEDFS = 1, PS = 2, NONE = 3};

struct ComputationGraph; // TODO is there a nicer way to resolve this cyclic dependency?
struct Tensor;

struct DeviceMempoolSizes {
  size_t used[3];
  DeviceMempoolSizes() = default;
  DeviceMempoolSizes(size_t total_s);
  DeviceMempoolSizes(size_t fxs_s, size_t dEdfs_s, size_t ps_s);
  DeviceMempoolSizes(const std::string & descriptor);
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
  virtual DeviceMempoolSizes mark(ComputationGraph *cg);
  virtual void revert(const DeviceMempoolSizes & cp);
  void allocate_tensor(DeviceMempool mem_pool, Tensor & tensor);
  std::vector<AlignedMemoryPool*> pools;
};

#if HAVE_CUDA
class Device_GPU : public Device {
 public:
  typedef Eigen::CudaStreamDevice EigenDevice;
  explicit Device_GPU(int my_id, const DeviceMempoolSizes & mb, int device_id);
  ~Device_GPU();
  int cuda_device_id;
  cublasHandle_t cublas_handle;
#if HAVE_CUDNN
  cudnnHandle_t cudnnHandle;
#endif
  Eigen::GpuDevice* edevice;
  Eigen::CudaStreamDevice* estream;
  GPUAllocator gpu_mem;
};
#endif

class Device_CPU : public Device {
 public:
  typedef Eigen::DefaultDevice EigenDevice;
  explicit Device_CPU(int my_id, const DeviceMempoolSizes & mb, bool shared);
  ~Device_CPU();
  CPUAllocator cpu_mem;
  Eigen::DefaultDevice* edevice;
  MemAllocator* shmem;
};

class Device_ThreadPool : public Device {
 public:
  typedef Eigen::ThreadPoolDevice EigenDevice;
  explicit Device_ThreadPool(int my_id, const DeviceMempoolSizes &mb, bool shared, int num_cores);
  ~Device_ThreadPool();
  CPUAllocator cpu_mem;
  Eigen::ThreadPool* tp;
  Eigen::ThreadPoolDevice* edevice;
  MemAllocator* shmem;
};

} // namespace dynet

#endif
