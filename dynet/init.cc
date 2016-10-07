#include "dynet/init.h"
#include "dynet/aligned-mem-pool.h"
#include "dynet/dynet.h"
#include "dynet/weight-decay.h"

#include <iostream>
#include <random>
#include <cmath>

#if HAVE_CUDA
#include "dynet/cuda.h"
#include <device_launch_parameters.h>
#endif

using namespace std;

namespace dynet {

static void remove_args(int& argc, char**& argv, int& argi, int n) {
  for (int i = argi + n; i < argc; ++i)
    argv[i - n] = argv[i];
  argc -= n;
  assert(argc >= 0);
}

void initialize(int& argc, char**& argv, bool shared_parameters) {
  if(default_device != nullptr) {
    cerr << "WARNING: Attempting to initialize dynet twice. Ignoring duplicate initialization." << endl;
    return;
  }
  vector<Device*> gpudevices;
#if HAVE_CUDA
  cerr << "[dynet] initializing CUDA\n";
  gpudevices = initialize_gpu(argc, argv);
#endif
  unsigned random_seed = 0;
  int argi = 1;
  string mem_descriptor = "512";
  while(argi < argc) {
    string arg = argv[argi];
    if (arg == "--dynet-mem" || arg == "--dynet_mem") {
      if ((argi + 1) > argc) {
        cerr << "[dynet] --dynet-mem expects an argument (the memory, in megabytes, to reserve)\n";
        abort();
      } else {
        mem_descriptor = argv[argi+1];
        remove_args(argc, argv, argi, 2);
      }
    } else if (arg == "--dynet-l2" || arg == "--dynet_l2") {
      if ((argi + 1) > argc) {
        cerr << "[dynet] --dynet-l2 requires an argument (the weight decay per update)\n";
        abort();
      } else {
        string a2 = argv[argi+1];
        float decay = 0;
        istringstream d(a2); d >> decay;
        remove_args(argc, argv, argi, 2);
        if (decay < 0 || decay >= 1) {
          cerr << "[dynet] weight decay parameter must be between 0 and 1 (probably very small like 1e-6)\n";
          abort();
        }
        weight_decay_lambda = decay;
      }
    } else if (arg == "--dynet-seed" || arg == "--dynet_seed") {
      if ((argi + 1) > argc) {
        cerr << "[dynet] --dynet-seed expects an argument (the random number seed)\n";
        abort();
      } else {
        string a2 = argv[argi+1];
        istringstream c(a2); c >> random_seed;
        remove_args(argc, argv, argi, 2);
      }
    } else if (arg.find("--dynet") == 0) {
      cerr << "[dynet] Bad command line argument: " << arg << endl;
      abort();
    } else { break; }
  }
  if (random_seed == 0) {
    random_device rd;
    random_seed = rd();
  }
  cerr << "[dynet] random seed: " << random_seed << endl;
  rndeng = new mt19937(random_seed);

  cerr << "[dynet] allocating memory: " << mem_descriptor << "MB\n";
  devices.push_back(new Device_CPU(devices.size(), mem_descriptor, shared_parameters));
  int default_index = 0;
  if (gpudevices.size() > 0) {
    for (auto gpu : gpudevices)
      devices.push_back(gpu);
    default_index++;
  }
  default_device = devices[default_index];

  // TODO these should be accessed through the relevant device and removed here
  kSCALAR_MINUSONE = default_device->kSCALAR_MINUSONE;
  kSCALAR_ONE = default_device->kSCALAR_ONE;
  kSCALAR_ZERO = default_device->kSCALAR_ZERO;
  cerr << "[dynet] memory allocation done.\n";
}

void cleanup() {
  delete rndeng;
  // TODO: Devices cannot be deleted at the moment
  // for(Device* device : devices) delete device;
  devices.clear();
  default_device = nullptr;
}

} // namespace dynet

