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

#define ALIGN 5
AlignedMemoryPool<ALIGN>* fxs = nullptr;
AlignedMemoryPool<ALIGN>* dEdfs = nullptr;
mt19937* rndeng = nullptr;

void Initialize(int& argc, char**& argv) {
  cerr << "Initializing...\n";
#if HAVE_CUDA
  Initialize_GPU(argc, argv);
#else
  kSCALAR_MINUSONE = new float;
  *kSCALAR_MINUSONE = -1;
  kSCALAR_ONE = new float;
  *kSCALAR_ONE = 1;
  kSCALAR_ZERO = new float;
  *kSCALAR_ZERO = 0;
#endif
  random_device rd;
//  rndeng = new mt19937(1);
  rndeng = new mt19937(rd());
  cerr << "Allocating memory...\n";
  fxs = new AlignedMemoryPool<ALIGN>(512*(1<<20));
  dEdfs = new AlignedMemoryPool<ALIGN>(512*(1<<20));
  cerr << "Done.\n";
}

} // namespace cnn

