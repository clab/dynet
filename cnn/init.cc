#include "cnn/init.h"
#include "cnn/aligned-mem-pool.h"
#include "cnn/cnn.h"

#include <iostream>
#include <random>
#include <cmath>

#if HAVE_CUDA
#include "cnn/cuda.h"
#include <device_launch_parameters.h>
#endif

using namespace std;

namespace cnn {

const unsigned ALIGN = 6;
const unsigned node_value_memory_size = 512UL * (1UL << 20);
const unsigned parameter_memory_size = 512UL * (1UL << 20);
AlignedMemoryPool<ALIGN>* fxs = nullptr;
AlignedMemoryPool<ALIGN>* dEdfs = nullptr;
AlignedMemoryPool<ALIGN>* ps = nullptr;
mt19937* rndeng = nullptr;

void Initialize(int& argc, char**& argv, unsigned random_seed, bool shared_parameters) {
  cerr << "Initializing...\n";
#if HAVE_CUDA
  Initialize_GPU(argc, argv);
#else
  kSCALAR_MINUSONE = (float*) cnn_mm_malloc(sizeof(float), 256);
  *kSCALAR_MINUSONE = -1;
  kSCALAR_ONE = (float*) cnn_mm_malloc(sizeof(float), 256);
  *kSCALAR_ONE = 1;
  kSCALAR_ZERO = (float*) cnn_mm_malloc(sizeof(float), 256);
  *kSCALAR_ZERO = 0;
#endif
  if (random_seed == 0) {
    random_device rd;
    random_seed = rd();
  }
  rndeng = new mt19937(random_seed);
  cerr << "Allocating memory...\n";
  fxs = new AlignedMemoryPool<ALIGN>(node_value_memory_size); // values
  dEdfs = new AlignedMemoryPool<ALIGN>(node_value_memory_size); // gradients
  ps = new AlignedMemoryPool<ALIGN>(parameter_memory_size, shared_parameters); // parameters
  cerr << "Done.\n";
}

void Cleanup() {
  delete rndeng;
  delete fxs;
  delete dEdfs;
  delete ps;
}

} // namespace cnn

