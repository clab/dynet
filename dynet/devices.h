#ifndef DYNET_DEVICES_H
#define DYNET_DEVICES_H

#include <unordered_map>
#include <string>
#include <exception>
#include "dynet/aligned-mem-pool.h"
#include "dynet/cuda.h"
#include "dynet/globals.h"
#include <unsupported/Eigen/CXX11/Tensor>

namespace Eigen {
  struct DefaultDevice;
  class CudaStreamDevice;
  struct GpuDevice;
}

namespace dynet {

enum class DeviceType {CPU, GPU};

/*
 * FXS   -> forward pass memory
 * DEDFS -> backward pass memory
 * PS    -> parameter memory
 * SCS   -> scratch memory (for use in temporary calculations)
 * NONE  -> when a memory pool has not been assigned yet
 */
enum class DeviceMempool {FXS = 0, DEDFS = 1, PS = 2, SCS = 3, NONE = 4};

struct ComputationGraph; // TODO is there a nicer way to resolve this cyclic dependency?
struct Tensor;

struct DeviceMempoolSizes {
  size_t used[4];
  DeviceMempoolSizes() = default;
  DeviceMempoolSizes(size_t total_s);
  DeviceMempoolSizes(size_t fxs_s, size_t dEdfs_s, size_t ps_s, size_t sc_s);
  DeviceMempoolSizes(const std::string & descriptor);
};


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
static const cudaStream_t default_stream = cudaStreamDefault;

class EigenCudaStreamDevice : public Eigen::StreamInterface {
 public:
  // Use the default stream on the current device
  EigenCudaStreamDevice()
      : stream_(&default_stream), scratch_(NULL), semaphore_(NULL) {
    cudaGetDevice(&device_);
    Eigen::initializeDeviceProp();
  }
  // Use the default stream on the specified device
  explicit EigenCudaStreamDevice(int device)
      : stream_(&default_stream),
        device_(device),
        scratch_(NULL),
        semaphore_(NULL) {
    Eigen::initializeDeviceProp();
  }
  // Use the specified stream. Note that it's the
  // caller responsibility to ensure that the stream can run on
  // the specified device. If no device is specified the code
  // assumes that the stream is associated to the current gpu device.
  explicit EigenCudaStreamDevice(const cudaStream_t* stream, int device = -1)
      : stream_(stream), device_(device), scratch_(NULL), semaphore_(NULL) {
    if (device < 0) {
      cudaGetDevice(&device_);
    } else {
      int num_devices;
      cudaGetDeviceCount(&num_devices);
      device_ = device;
    }
    Eigen::initializeDeviceProp();
  }

  virtual ~EigenCudaStreamDevice() {
    if (scratch_) {
      deallocate(scratch_);
    }
  }

  const cudaStream_t& stream() const { return *stream_; }

  void set_stream(const cudaStream_t* cuda_stream) { stream_ = cuda_stream; }
  
  const cudaDeviceProp& deviceProperties() const {
    return Eigen::m_deviceProperties[device_];
  }
  virtual void* allocate(size_t num_bytes) const {
    cudaSetDevice(device_);
    void* result;
    cudaMalloc(&result, num_bytes);
    return result;
  }
  virtual void deallocate(void* buffer) const {
    cudaSetDevice(device_);
    cudaFree(buffer);
  }

  virtual void* scratchpad() const {
    if (scratch_ == NULL) {
      scratch_ = allocate(Eigen::kCudaScratchSize + sizeof(unsigned int));
    }
    return scratch_;
  }

  virtual unsigned int* semaphore() const {
    if (semaphore_ == NULL) {
      char* scratch =
          static_cast<char*>(scratchpad()) + Eigen::kCudaScratchSize;
      semaphore_ = reinterpret_cast<unsigned int*>(scratch);
      cudaMemsetAsync(semaphore_, 0, sizeof(unsigned int), *stream_);
    }
    return semaphore_;
  }

 private:
  const cudaStream_t* stream_;
  int device_;
  mutable void* scratch_;
  mutable unsigned int* semaphore_;
};

class Device_GPU : public Device {
 public:
  typedef EigenCudaStreamDevice EigenDevice;
  explicit Device_GPU(int my_id, const DeviceMempoolSizes & mb, int device_id);
  ~Device_GPU();
  int cuda_device_id;
  cublasHandle_t cublas_handle;
#if HAVE_CUDNN
  cudnnHandle_t cudnnHandle;
#endif
  Eigen::GpuDevice* edevice;
  EigenCudaStreamDevice* estream;
  cudaStream_t stream;
  GPUAllocator gpu_mem;
};
#endif

class Device_CPU : public Device {
 public:
  typedef Eigen::DefaultDevice EigenDevice;
  explicit Device_CPU(int my_id, const DeviceMempoolSizes & mb, bool shared, bool dynamic);
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

} // namespace dynet

#endif
