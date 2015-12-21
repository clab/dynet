#include <iostream>

#include "cnn/cnn.h"
#include "cnn/cuda.h"

using namespace std;

namespace cnn {

cublasHandle_t cublas_handle;

void Initialize_GPU(int& argc, char**& argv) {
  int nDevices;
  CUDA_CHECK(cudaGetDeviceCount(&nDevices));
  if (nDevices < 1) {
    cerr << "[cnn] No GPUs found, recompile without DENABLE_CUDA=1\n";
    throw std::runtime_error("No GPUs found but CNN compiled with CUDA support.");
  }
  size_t free_bytes, total_bytes, max_free = 0;
  int selected = 0;
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
    cerr << "[cnn] Device Number: " << i << endl;
    cerr << "[cnn]   Device name: " << prop.name << endl;
    cerr << "[cnn]   Memory Clock Rate (KHz): " << prop.memoryClockRate << endl;
    cerr << "[cnn]   Memory Bus Width (bits): " << prop.memoryBusWidth << endl;
    cerr << "[cnn]   Peak Memory Bandwidth (GB/s): " << (2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6) << endl << endl;
    CUDA_CHECK(cudaSetDevice(i));
    CUDA_CHECK(cudaMemGetInfo( &free_bytes, &total_bytes ));
    CUDA_CHECK(cudaDeviceReset());
    cerr << "[cnn]   Memory Free (MB): " << (int)free_bytes/1.0e6 << "/" << (int)total_bytes/1.0e6 << endl << endl;
    if(free_bytes > max_free) {
        max_free = free_bytes;
        selected = i;
    }
  }
  cerr << "[cnn] **USING DEVICE: " << selected << endl;
  CUDA_CHECK(cudaSetDevice(selected));
  CUBLAS_CHECK(cublasCreate(&cublas_handle));
  CUBLAS_CHECK(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE));
  CUDA_CHECK(cudaMalloc(&kSCALAR_MINUSONE, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&kSCALAR_ONE, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&kSCALAR_ZERO, sizeof(float)));
  float minusone = -1;
  CUDA_CHECK(cudaMemcpyAsync(kSCALAR_MINUSONE, &minusone, sizeof(float), cudaMemcpyHostToDevice));
  float one = 1;
  CUDA_CHECK(cudaMemcpyAsync(kSCALAR_ONE, &one, sizeof(float), cudaMemcpyHostToDevice));
  float zero = 0;
  CUDA_CHECK(cudaMemcpyAsync(kSCALAR_ZERO, &zero, sizeof(float), cudaMemcpyHostToDevice));
}

} // namespace cnn
