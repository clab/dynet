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

#define ALIGN 6
AlignedMemoryPool<ALIGN>* fxs = nullptr;
AlignedMemoryPool<ALIGN>* dEdfs = nullptr;
mt19937* rndeng = nullptr;

char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

void Initialize(int& argc, char**& argv) {
  cerr << "Initializing...\n";
#if HAVE_CUDA
  Initialize_GPU(argc, argv);
#else
  kSCALAR_MINUSONE = (float*) cnn_mm_malloc(1, 256);
  *kSCALAR_MINUSONE = -1;
  kSCALAR_ONE = (float*) cnn_mm_malloc(1, 256);
  *kSCALAR_ONE = 1;
  kSCALAR_ZERO = (float*) cnn_mm_malloc(1, 256);
  *kSCALAR_ZERO = 0;
#endif

  if (cmdOptionExists(argv, argv + argc, "--seed"))
  {
      string seed = getCmdOption(argv, argv + argc, "--seed");
      int rseed;
      stringstream(seed) >> rseed; 
      rndeng = new mt19937(rseed);
  }
  else 
  {
      /// do nothing
      if (rndeng == nullptr)
      {
          random_device rd;
          //  rndeng = new mt19937(1);
          rndeng = new mt19937(rd());
      }
  }

  cerr << "Allocating memory...\n";
  fxs = new AlignedMemoryPool<ALIGN>(512*(1<<20));
  dEdfs = new AlignedMemoryPool<ALIGN>(512*(1<<20));
  cerr << "Done.\n";
}

} // namespace cnn

