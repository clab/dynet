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

// these should maybe live in a file called globals.cc or something
AlignedMemoryPool* fxs = nullptr;
AlignedMemoryPool* dEdfs = nullptr;
AlignedMemoryPool* ps = nullptr;
mt19937* rndeng = nullptr;
std::vector<Device*> devices;
Device* default_device = nullptr;

static void RemoveArgs(int& argc, char**& argv, int& argi, int n) {
  for (int i = argi + n; i < argc; ++i)
    argv[i - n] = argv[i];
  argc -= n;
  assert(argc >= 0);
}

void Initialize(int& argc, char**& argv, unsigned random_seed, bool shared_parameters) {
  vector<Device*> gpudevices;
#if HAVE_CUDA
  cerr << "[cnn] initializing CUDA\n";
  gpudevices = Initialize_GPU(argc, argv);
#endif
  unsigned long num_mb = 512UL;
  int argi = 1;
  while(argi < argc) {
    string arg = argv[argi];
    if (arg == "--cnn-mem" || arg == "--cnn_mem") {
      if ((argi + 1) > argc) {
        cerr << "[cnn] --cnn-mem expects an argument (the memory, in megabytes, to reserve)\n";
        abort();
      } else {
        string a2 = argv[argi+1];
        istringstream c(a2); c >> num_mb;
        RemoveArgs(argc, argv, argi, 2);
      }
    } else if (arg == "--cnn-seed" || arg == "--cnn_seed") {
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
  devices.push_back(new Device_CPU(num_mb, shared_parameters));
  int default_index = 0;
  if (gpudevices.size() > 0) {
    for (auto gpu : gpudevices)
      devices.push_back(gpu);
    default_index++;
  }
  default_device = devices[default_index];

  // TODO these should be accessed through the relevant device and removed here
  fxs = default_device->fxs;
  dEdfs = default_device->dEdfs;
  ps = default_device->ps;
  kSCALAR_MINUSONE = default_device->kSCALAR_MINUSONE;
  kSCALAR_ONE = default_device->kSCALAR_ONE;
  kSCALAR_ZERO = default_device->kSCALAR_ZERO;
  cerr << "[cnn] memory allocation done.\n";
}

void Cleanup() {
  delete rndeng;
  delete fxs;
  delete dEdfs;
  delete ps;
}

} // namespace cnn

