#include "cnn/backends/eigen/init.h"
#include "cnn/aligned-mem-pool.h"

#include <iostream>
#include <random>
#include <cmath>

#if HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

using namespace std;

namespace cnn {

AlignedMemoryPool<5>* fxs;
AlignedMemoryPool<5>* dEdfs;

#if HAVE_CUDA
void Initialize_GPU(int& argc, char**& argv) {
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  if (nDevices < 1) {
    cerr << "No GPUs found, recompile without DENABLE_CUDA=1\n";
    abort();
  }
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    cerr << "Device Number: " << i << endl;
    cerr << "  Device name: " << prop.name << endl;
    cerr << "  Memory Clock Rate (KHz): " << prop.memoryClockRate << endl;
    cerr << "  Memory Bus Width (bits): " << prop.memoryBusWidth << endl;
    cerr << "  Peak Memory Bandwidth (GB/s): " << (2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6) << endl << endl;
  }
  int selected = 0;
  cerr << "**USING DEVICE: " << selected << endl;
  cudaSetDevice(selected);
}
#endif

std::mt19937* rndeng = nullptr;
void Initialize(int& argc, char**& argv) {
  cerr << "Initializing...\n";
#if HAVE_CUDA
  Initialize_GPU(argc, argv);
#endif
  std::random_device rd;
  rndeng = new mt19937(rd());
  cerr << "Allocating memory...\n";
  fxs = new AlignedMemoryPool<5>(100000000);
  dEdfs = new AlignedMemoryPool<5>(100000000);
  cerr << "Done.\n";
//  rndeng = new mt19937(1);
}

} // namespace cnn

