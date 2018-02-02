#include "dynet/init.h"

#include "dynet/aligned-mem-pool.h"
#include "dynet/dynet.h"
#include "dynet/weight-decay.h"
#include "dynet/globals.h"
#include "dynet/str-util.h"
#include "dynet/devices.h"

#include <iostream>
#include <random>
#include <cmath>

#if HAVE_CUDA
#include "dynet/cuda.h"
#include <device_launch_parameters.h>
#endif

using namespace std;

namespace dynet {

DynetParams::DynetParams() : random_seed(0), mem_descriptor("512"), weight_decay(0), autobatch(0), profiling(0),
  shared_parameters(false), ngpus_requested(false), ids_requested(false), cpu_requested(false), requested_gpus(-1)
{
#if HAVE_CUDA
  gpu_mask = std::vector<int>(MAX_GPUS, 0);
#endif
}

DynetParams::~DynetParams()
{
}

static bool has_arg(int argi, int argc, char** argv) {
  const std::string arg(argv[argi]);
  auto pos = arg.find('=');
  if (pos == std::string::npos) {
    if ((argi + 1) < argc) {
      const std::string argn(argv[argi + 1]);
      if (argn.size() < 2 || !(argn[0] == '-' && argn[1] == '-'))
        return true;
    }
  } else {
    // check that there is actually a string present
    if ((pos + 1) < arg.size()) {
      return true;
    }
  }
  return false;
}

static void remove_args(int& argc, char**& argv, int& argi, int n) {
  if (n == 2) {
    // when we want a single argument, check to see if it was specified with
    // an '=', if so decrement the argument count.
    const std::string arg(argv[argi]);
    if (arg.find('=') != std::string::npos) {
      --n;
    }
  }
  for (int i = argi + n; i < argc; ++i)
    argv[i - n] = argv[i];
  argc -= n;
  DYNET_ASSERT(argc >= 0, "remove_args less than 0");
}

static std::string get_arg(int argi, char** argv) {
  const std::string arg(argv[argi]);
  auto pos = arg.find('=');
  if (pos != std::string::npos) {
    return arg.substr(pos+1);
  }
  return argv[argi + 1];
}

DynetParams extract_dynet_params(int& argc,
                                 char**& argv, bool shared_parameters) {
  DynetParams params;
  params.shared_parameters = shared_parameters;

  int argi = 1;

#if HAVE_CUDA
  params.gpu_mask = std::vector<int>(MAX_GPUS, 0);
#endif


  while (argi < argc) {
    string arg = argv[argi];

    // Memory
    if (startswith(arg, "--dynet-mem") || startswith(arg, "--dynet_mem")) {
      if (!has_arg(argi, argc, argv)) {
        throw std::invalid_argument("[dynet] --dynet-mem expects an argument (the memory, in megabytes, to reserve)");
      } else {
        params.mem_descriptor = get_arg(argi, argv);
        remove_args(argc, argv, argi, 2);
      }
    }

    // Weight decay
    else if (startswith(arg, "--dynet-weight-decay") ||
             startswith(arg, "--dynet_weight_decay")) {
      if (!has_arg(argi, argc, argv)) {
        throw std::invalid_argument("[dynet] --dynet-weight-decay requires an argument (the weight decay per update)");
      } else {
        string a2 = get_arg(argi, argv);
        istringstream d(a2); d >> params.weight_decay;
        remove_args(argc, argv, argi, 2);
      }
    }

    // Random seed
    else if (startswith(arg, "--dynet-seed") ||
             startswith(arg, "--dynet_seed")) {
      if (!has_arg(argi, argc, argv)) {
        throw std::invalid_argument("[dynet] --dynet-seed expects an argument (the random number seed)");
      } else {
        string a2 = get_arg(argi, argv);
        istringstream c(a2); c >> params.random_seed;
        remove_args(argc, argv, argi, 2);
      }
    }

    // Autobatching
    else if (startswith(arg, "--dynet-autobatch") ||
             startswith(arg, "--dynet_autobatch")) {
      if (!has_arg(argi, argc, argv)) {
        throw std::invalid_argument("[dynet] --dynet-autobatch expects an argument (0 for none 1 for on)");
      } else {
        string a2 = get_arg(argi, argv);
        istringstream c(a2); c >> params.autobatch;
        remove_args(argc, argv, argi, 2);
      }
    }

    // Profiling
    else if (startswith(arg, "--dynet-profiling") ||
             startswith(arg, "--dynet_profiling")) {
      if (!has_arg(argi, argc, argv)) {
        throw std::invalid_argument("[dynet] --dynet-profiling expects an argument (0 for none 1 for on)");
      } else {
        string a2 = get_arg(argi, argv);
        istringstream c(a2); c >> params.profiling;
        remove_args(argc, argv, argi, 2);
      }
    }

#if HAVE_CUDA
    else if (startswith(arg, "--dynet-gpus") ||
             startswith(arg, "--dynet_gpus")) {
      if (!has_arg(argi, argc, argv)) {
        throw std::invalid_argument("[dynet] --dynet-gpus expects an argument (number of GPUs to use)");
      } else {
        if (params.ngpus_requested)
          throw std::invalid_argument("Multiple instances of --dynet-gpus");
        params.ngpus_requested = true;
        string a2 = get_arg(argi, argv);
        istringstream c(a2); c >> params.requested_gpus;
        remove_args(argc, argv, argi, 2);
      }
    }
#endif

    // Devices
    else if (startswith(arg, "--dynet-devices") ||
             startswith(arg, "--dynet_devices")) {
      if (!has_arg(argi, argc, argv)) {
        throw std::invalid_argument("[dynet] --dynet-devices expects an argument (comma separated list of CPU and physical GPU ids to use)");
      } else {
        string devices_str = get_arg(argi, argv);
        if (params.ids_requested)
           throw std::invalid_argument("Multiple instances of --dynet-devices");
        params.ids_requested = true;
        auto devices_info_lst = str_split(devices_str, ',');
        for (auto & devices_info : devices_info_lst) {
          if (startswith(devices_info, "CPU:")) {
            throw std::invalid_argument("DyNet doesn't support specifying CPU id");
          } else if (startswith(devices_info, "CPU")) {
            if (params.cpu_requested)
              throw std::invalid_argument("Bad argument to --dynet-devices");
            params.cpu_requested = true;
          } else if (startswith(devices_info, "GPU:")) {
            int gpu_id = std::stoi(devices_info.substr(4, devices_info.size() - 4));
            if (gpu_id >= 256) // MAX_GPUS
              throw std::runtime_error("DyNet hard limit on maximum number of GPUs (MAX_GPUS) exceeded. If you need more, modify the code to raise this hard limit.");
            params.gpu_mask[gpu_id] ++;
            params.requested_gpus++;
            if (params.gpu_mask[gpu_id] != 1) {
              ostringstream oss; oss << "Bad argument to --dynet-devices: " << devices_info;
              throw std::invalid_argument(oss.str());
            }
          } else {
            throw std::invalid_argument("Bad argument to --dynet-devices");
          }
        }
        params.cpu_requested = true;
        remove_args(argc, argv, argi, 2);
      }
    }

    // Go to next argument
    else {
      argi++;
    }
  }

#if HAVE_CUDA
  // Check for conflict between the two ways of requesting GPUs
  if (params.ids_requested && params.ngpus_requested)
    throw std::invalid_argument("Use only --dynet_gpus or --dynet_gpu_ids, not both\n");
#endif

  return params;
}

void initialize(DynetParams& params) {
  if (default_device != nullptr) {
    cerr << "WARNING: Attempting to initialize dynet twice. Ignoring duplicate initialization." << endl;
    return;
  }

  DeviceManager* device_manager = get_device_manager();

  // initialize CUDA
  vector<Device*> gpudevices;
#if HAVE_CUDA
  if (!(params.cpu_requested && (params.requested_gpus == -1))) {
    cerr << "[dynet] initializing CUDA\n";
    gpudevices = initialize_gpu(params);
    for (auto gpu : gpudevices)
      device_manager->add(gpu);
  }
#endif

  // Set random seed
  if (params.random_seed == 0) {
    random_device rd;
    params.random_seed = rd();
  }
  cerr << "[dynet] random seed: " << params.random_seed << endl;
  rndeng = new mt19937(params.random_seed);
#if HAVE_CUDA
  CURAND_CHECK(curandCreateGenerator(&curandeng, 
                                     CURAND_RNG_PSEUDO_PHILOX4_32_10));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curandeng, 
                                                  params.random_seed+1));
#endif

  // Set weight decay rate
  if (params.weight_decay < 0 || params.weight_decay >= 1)
    throw std::invalid_argument("[dynet] weight decay parameter must be between 0 and 1 (probably very small like 1e-6)\n");
  weight_decay_lambda = params.weight_decay;

  // Set autobatch
  if(params.autobatch)
    cerr << "[dynet] using autobatching" << endl;
  autobatch_flag = params.autobatch;
  
  if(params.profiling)
    cerr << "[dynet] using profiling level " << params.profiling << endl;
  profiling_flag = params.profiling;

  // Allocate memory
  cerr << "[dynet] allocating memory: " << params.mem_descriptor << "MB\n";
  int default_index = 0;

  Device *d;
  if (gpudevices.size()) {
    d = new Device_CPU(device_manager->num_devices(), std::string("128"), params.shared_parameters);
  } else {
    d = new Device_CPU(device_manager->num_devices(), params.mem_descriptor, params.shared_parameters);
  }
  device_manager->add(d);
  default_device = device_manager->get(default_index);
#if HAVE_CUDA
  if (default_device->type == DeviceType::GPU) {
    auto default_gpu_device = static_cast<Device_GPU *>(default_device);
    CUDA_CHECK(cudaSetDevice(default_gpu_device->cuda_device_id));
  }
#endif

  // TODO these should be accessed through the relevant device and removed here
  kSCALAR_MINUSONE = default_device->kSCALAR_MINUSONE;
  kSCALAR_ONE = default_device->kSCALAR_ONE;
  kSCALAR_ZERO = default_device->kSCALAR_ZERO;
  cerr << "[dynet] memory allocation done.\n";

}

void initialize(int& argc, char**& argv, bool shared_parameters) {
  DynetParams params = extract_dynet_params(argc, argv, shared_parameters);
  initialize(params);
}

void cleanup() {
  delete rndeng;
  get_device_manager()->clear();
  default_device = nullptr;
}

} // namespace dynet
