#include "cnn/devices.h"

#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

#include "cnn/cuda.h"
#include "cnn/cnn.h"

using namespace std;

namespace cnn {

Device::~Device() {}

DeviceMemCheckpoint Device::mark(ComputationGraph *cg) {
  cg->incremental_forward(); // needed so that we actually allocate the needed memory
                             // for all existing nodes.
  DeviceMemCheckpoint cp;
  for(size_t i = 0; i < 3; ++i)
    cp.used[i] = pools[i]->used;
  return cp;
}

void Device::revert(DeviceMemCheckpoint cp) {
  for(size_t i = 0; i < 3; ++i) {
    assert(cp.used[i] <= pools[i]->used);
    pools[i]->used = cp.used[i];
  }
}

void Device::allocate_tensor(DeviceMempool mp, Tensor & tens) {
  assert(mp != DeviceMempool::NONE);
  assert(pools[(int)mp] != nullptr);
  tens.v = (float*)pools[(int)mp]->allocate(tens.d.size() * sizeof(float));
  tens.mem_pool = mp;
}

#if HAVE_CUDA
Device_GPU::Device_GPU(int my_id, int mb, int device_id) :
    Device(my_id, DeviceType::GPU, &gpu_mem), cuda_device_id(device_id), gpu_mem(device_id) {
  CUDA_CHECK(cudaSetDevice(device_id));
  CUBLAS_CHECK(cublasCreate(&cublas_handle));
  CUBLAS_CHECK(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE));
  kSCALAR_MINUSONE = (float*)gpu_mem.malloc(sizeof(float));
  kSCALAR_ONE = (float*)gpu_mem.malloc(sizeof(float));
  kSCALAR_ZERO = (float*)gpu_mem.malloc(sizeof(float));
  float minusone = -1;
  CUDA_CHECK(cudaMemcpyAsync(kSCALAR_MINUSONE, &minusone, sizeof(float), cudaMemcpyHostToDevice));
  float one = 1;
  CUDA_CHECK(cudaMemcpyAsync(kSCALAR_ONE, &one, sizeof(float), cudaMemcpyHostToDevice));
  float zero = 0;
  CUDA_CHECK(cudaMemcpyAsync(kSCALAR_ZERO, &zero, sizeof(float), cudaMemcpyHostToDevice));

  // Initialize the Eigen device
  estream = new Eigen::CudaStreamDevice(device_id);
  edevice = new Eigen::GpuDevice(estream);

  // this is the big memory allocation. Do it in stages to make sure things are aligned.
  size_t byte_count = (size_t)((mb << 10)/3) << 10;
  for(size_t i = 0; i < 3; ++i)
    pools[i] = new AlignedMemoryPool(byte_count, mem);
}

Device_GPU::~Device_GPU() {}
#endif

// TODO we should be able to configure this carefully with a configuration
// script
// CPU -- 0 params
//     -- 50mb fxs
//     -- 50mb dEdfx
Device_CPU::Device_CPU(int my_id, int mb, bool shared) :
    Device(my_id, DeviceType::CPU, &cpu_mem), shmem(mem) {
  if (shared) shmem = new SharedAllocator();
  kSCALAR_MINUSONE = (float*) mem->malloc(sizeof(float));
  *kSCALAR_MINUSONE = -1;
  kSCALAR_ONE = (float*) mem->malloc(sizeof(float));
  *kSCALAR_ONE = 1;
  kSCALAR_ZERO = (float*) mem->malloc(sizeof(float));
  *kSCALAR_ZERO = 0;

  // Initialize the Eigen device
  edevice = new Eigen::DefaultDevice;

  // this is the big memory allocation. Do it in stages to make sure things are aligned.
  size_t byte_count = (size_t)((mb << 10)/3) << 10;
  for(size_t i = 0; i < 3; ++i)
    pools[i] = new AlignedMemoryPool(byte_count, mem);
}

Device_CPU::~Device_CPU() {}

} // namespace cnn
