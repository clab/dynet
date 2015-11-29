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
AlignedMemoryPool<ALIGN>* fxs = nullptr;
AlignedMemoryPool<ALIGN>* dEdfs = nullptr;
AlignedMemoryPool<ALIGN>* ps = nullptr;
mt19937* rndeng = nullptr;

static void RemoveArgs(int& argc, char**& argv, int& argi, int n) {
  for (int i = argi + n; i < argc; ++i)
    argv[i - n] = argv[i];
  argc -= n;
  assert(argc >= 0);
}

void Initialize(int& argc, char**& argv, unsigned random_seed, bool shared_parameters) {
#if HAVE_CUDA
  cerr << "[cnn] using GPU\n";
  Initialize_GPU(argc, argv);
#else
  cerr << "[cnn] using CPU\n";
  kSCALAR_MINUSONE = (float*) cnn_mm_malloc(sizeof(float), 256);
  *kSCALAR_MINUSONE = -1;
  kSCALAR_ONE = (float*) cnn_mm_malloc(sizeof(float), 256);
  *kSCALAR_ONE = 1;
  kSCALAR_ZERO = (float*) cnn_mm_malloc(sizeof(float), 256);
  *kSCALAR_ZERO = 0;
#endif
  unsigned long num_mb = 512UL;
  int argi = 1;
  while(argi < argc) {
    string arg = argv[argi];
    if (arg == "--cnn-mem") {
      if ((argi + 1) > argc) {
        cerr << "[cnn] --cnn-mem expects an argument (the memory, in megabytes, to reserve)\n";
        abort();
      } else {
        string a2 = argv[argi+1];
        istringstream c(a2); c >> num_mb;
        RemoveArgs(argc, argv, argi, 2);
      }
    } else if (arg == "--cnn-seed") {
      if ((argi + 1) > argc) {
        cerr << "[cnn] --cnn-seed expects an argument (the random number seed)\n";
        abort();
      } else {
        string a2 = argv[argi+1];
        istringstream c(a2); c >> random_seed;
        RemoveArgs(argc, argv, argi, 2);
      }
    } else if (arg.find("--cnn") == 0) {
      cerr << "[cnn] Bad command line argument: " << arg << endl;
      abort();
    } else { break; }
  }
  if (random_seed == 0) {
    random_device rd;
    random_seed = rd();
  }
  cerr << "[cnn] random seed: " << random_seed << endl;
  rndeng = new mt19937(random_seed);

  cerr << "[cnn] allocating memory: " << num_mb << "MB\n";
  fxs = new AlignedMemoryPool<ALIGN>(num_mb << 20); // node values
  dEdfs = new AlignedMemoryPool<ALIGN>(num_mb << 20); // node gradients
  ps = new AlignedMemoryPool<ALIGN>(num_mb << 20, shared_parameters); // parameters
  cerr << "[cnn] memory allocation done.\n";
}

void Cleanup() {
  delete rndeng;
  delete fxs;
  delete dEdfs;
  delete ps;
}

} // namespace cnn

