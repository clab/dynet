#include "cnn/devices.h"

#include <iostream>

#include "cnn/cuda.h"

using namespace std;

namespace cnn {

Device::~Device() {}

#if HAVE_CUDA
Device_GPU::Device_GPU(int mb, int device_id) :
    Device(DeviceType::GPU, &gpu_mem), cuda_device_id(device_id), gpu_mem(device_id) {
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

  // this is the big memory allocation
  fxs = new AlignedMemoryPool(mb << 20, mem); // memory for node values
  dEdfs = new AlignedMemoryPool(mb << 20, mem); // memory for node gradients
  ps = new AlignedMemoryPool(mb << 20, mem); // memory for parameters
}

Device_GPU::~Device_GPU() {}
#endif

// TODO we should be able to configure this carefully with a configuration
// script
// CPU -- 0 params
//     -- 50mb fxs
//     -- 50mb dEdfx
Device_CPU::Device_CPU(int mb, bool shared) :
    Device(DeviceType::CPU, &cpu_mem), shmem(mem) {
  if (shared) shmem = new SharedAllocator();
  kSCALAR_MINUSONE = (float*) mem->malloc(sizeof(float));
  *kSCALAR_MINUSONE = -1;
  kSCALAR_ONE = (float*) mem->malloc(sizeof(float));
  *kSCALAR_ONE = 1;
  kSCALAR_ZERO = (float*) mem->malloc(sizeof(float));
  *kSCALAR_ZERO = 0;

  // this is the big memory allocation: the pools
  fxs = new AlignedMemoryPool(mb << 20, mem); // memory for node values
  dEdfs = new AlignedMemoryPool(mb << 20, mem); // memory for node gradients
  ps = new AlignedMemoryPool(mb << 20, shmem); // memory for parameters
}

Device_CPU::~Device_CPU() {}

} // namespace cnn
