#ifndef DYNET_DEVICES_H
#define DYNET_DEVICES_H

#include <unordered_map>
#include <string>
#include <exception>
#include "dynet/aligned-mem-pool.h"
#include "dynet/cuda.h"
#include "dynet/globals.h"
#include "dynet/device-structs.h"
#include <unsupported/Eigen/CXX11/Tensor>

namespace Eigen {
  struct DefaultDevice;
  class CudaStreamDevice;
  struct GpuDevice;
}

namespace dynet {

class Device {
 protected:
  Device(int i, DeviceType t, MemAllocator* m) : device_id(i), type(t), mem(m), pools(4, nullptr) {}
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

class DeviceManager final {
 public:
  DeviceManager();
  ~DeviceManager();

  void clear();

  void add(Device* d);

  Device* get(size_t i) { return devices[i]; }

  size_t num_devices() const { return devices.size(); }

  const std::vector<Device*>& get_devices() const { return devices; }

  Device* get_global_device(const std::string & name);

  // no copying allowed
  DeviceManager(const DeviceManager &) = delete;
  void operator=(const DeviceManager &) = delete;

 private:
  std::vector<Device*> devices;
  std::unordered_map<std::string, Device*> devices_map;
};

DeviceManager* get_device_manager();

inline void show_pool_mem_info() {
  DeviceManager* device_manager = get_device_manager();
  auto devs = device_manager->get_devices();
  if (devs.size() == 0) return;
  std::cerr << "\nMemory pool info for each devices:\n";
  for (Device* dev : devs) {
    std::cerr << " Device " << dev->name << " - FOR Memory " << (dev->pools[0]->get_cap() >> 20)
        << "MB, BACK Memory " << (dev->pools[1]->get_cap() >> 20)
        << "MB, PARAM Memory " << (dev->pools[2]->get_cap() >> 20)
        << "MB, SCRATCH Memory " << (dev->pools[3]->get_cap() >> 20) << "MB." << std::endl;
  }
}

} // namespace dynet

#endif
