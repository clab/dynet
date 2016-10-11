#include <iostream>
#include <vector>
#include <algorithm>

#include "dynet/dynet.h"
#include "dynet/cuda.h"

using namespace std;

namespace dynet {

static void remove_args(int& argc, char**& argv, int& argi, int n) {
  for (int i = argi + n; i < argc; ++i)
    argv[i - n] = argv[i];
  argc -= n;
  assert(argc >= 0);
}

#define MAX_GPUS 256

vector<Device*> initialize_gpu(int& argc, char**& argv) {
  int nDevices;
  CUDA_CHECK(cudaGetDeviceCount(&nDevices));
  if (nDevices < 1) {
    cerr << "[dynet] No GPUs found, recompile without DENABLE_CUDA=1\n";
    throw std::runtime_error("No GPUs found but DYNET compiled with CUDA support.");
  }
  // logic: no flags, you get 1 GPU
  // or you request a certain number of GPUs explicitly
  // or you request the device ids
  int requested_gpus = -1;
  vector<int> gpu_mask(MAX_GPUS,0);
  int argi = 1;
  string mem_descriptor = "512";
  bool ngpus_requested = false;
  bool ids_requested = false;
  for( ;argi < argc; ++argi) {
    string arg = argv[argi];
    if (arg == "--dynet-mem" || arg == "--dynet_mem") {
      if ((argi + 1) > argc) {
        cerr << "[dynet] --dynet-mem expects an argument (the memory, in megabytes, to reserve)\n";
        abort();
      } else {
        mem_descriptor = argv[argi+1];
        remove_args(argc, argv, argi, 2);
      }
    } else if (arg == "--dynet_gpus" || arg == "--dynet-gpus") {
      if ((argi + 1) > argc) {
        cerr << "[dynet] --dynet-gpus expects an argument (number of GPUs to use)\n";
        abort();
      } else {
        if (ngpus_requested) {
          cerr << "Multiple instances of --dynet-gpus" << endl; abort();
        }
        ngpus_requested = true;
        string a2 = argv[argi+1];
        istringstream c(a2); c >> requested_gpus;
        remove_args(argc, argv, argi, 2);
      }
    } else if (arg == "--dynet_gpu_ids" || arg == "--dynet-gpu-ids") {
      if ((argi + 1) > argc) {
        cerr << "[dynet] --dynet-gpu-ids expects an argument (comma separated list of physical GPU ids to use)\n";
        abort();
      } else {
        string a2 = argv[argi+1];
        if (ids_requested) {
          cerr << "Multiple instances of --dynet-gpu-ids" << endl; abort();
        }
        ids_requested = true;
        if (a2.size() % 2 != 1) {
          cerr << "Bad argument to --dynet-gpu-ids: " << a2 << endl; abort();
        }
        for (unsigned i = 0; i < a2.size(); ++i) {
          if ((i % 2 == 0 && (a2[i] < '0' || a2[i] > '9')) ||
              (i % 2 == 1 && a2[i] != ',')) {
            cerr << "Bad argument to --dynet-gpu-ids: " << a2 << endl; abort();
          }
          if (i % 2 == 0) {
            int gpu_id = a2[i] - '0';
            if (gpu_id >= nDevices) {
              cerr << "You requested GPU id " << gpu_id << " but system only reports up to " << nDevices << endl;
              abort();
            }
            if (gpu_id >= MAX_GPUS) { cerr << "Raise MAX_GPUS\n"; abort(); }
            gpu_mask[gpu_id]++;
            requested_gpus++;
            if (gpu_mask[gpu_id] != 1) {
              cerr << "Bad argument to --dynet-gpu-ids: " << a2 << endl; abort();
            }
          }
        }
        remove_args(argc, argv, argi, 2);
      }
    }
  }
  if (ids_requested && ngpus_requested) {
    cerr << "Use only --dynet_gpus or --dynet_gpu_ids, not both\n";
    abort();
  }
  if (ngpus_requested || requested_gpus == -1) {
    if (requested_gpus == -1) requested_gpus = 1;
    cerr << "Request for " << requested_gpus << " GPU" << (requested_gpus == 1 ? "" : "s") << " ...\n";
    for (int i = 0; i < MAX_GPUS; ++i) gpu_mask[i] = 1;
  } else if (ids_requested) {
    requested_gpus++;
    cerr << "[dynet] Request for " << requested_gpus << " specific GPU" << (requested_gpus == 1 ? "" : "s") << " ...\n";
  }

  vector<Device*> gpudevices;
  if (requested_gpus == 0) return gpudevices;
  if (requested_gpus > nDevices) {
    cerr << "You requested " << requested_gpus << " GPUs but system only reports " << nDevices << endl;
    abort();
  }

  // after all that, requested_gpus is the number of GPUs to reserve
  // we now pick the ones that are both requested by the user or have
  // the most memory free

  vector<size_t> gpu_free_mem(MAX_GPUS, 0);
  vector<int> gpus(MAX_GPUS, 0);
  for (int i = 0; i < MAX_GPUS; ++i) gpus[i] = i;
  size_t free_bytes, total_bytes;
  for (int i = 0; i < nDevices; i++) {
    if (!gpu_mask[i]) continue;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
    cerr << "[dynet] Device Number: " << i << endl;
    cerr << "[dynet]   Device name: " << prop.name << endl;
    cerr << "[dynet]   Memory Clock Rate (KHz): " << prop.memoryClockRate << endl;
    cerr << "[dynet]   Memory Bus Width (bits): " << prop.memoryBusWidth << endl;
    cerr << "[dynet]   Peak Memory Bandwidth (GB/s): " << (2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6) << endl;
    if (!prop.unifiedAddressing) {
      cerr << "[dynet] GPU does not support unified addressing.\n";
      abort();
    }
    CUDA_CHECK(cudaSetDevice(i));
    try {
      CUDA_CHECK(cudaMemGetInfo( &free_bytes, &total_bytes ));
      cerr << "[dynet]   Memory Free (GB): " << free_bytes/1.0e9 << "/" << total_bytes/1.0e9 << endl;
      cerr << "[dynet]" << endl;
      gpu_free_mem[i] = free_bytes;
    } catch(dynet::cuda_exception e) {
      cerr << "[dynet]   FAILED to get free memory" << endl;
      gpu_free_mem[i] = 0;
      cudaGetLastError();
    }
    CUDA_CHECK(cudaDeviceReset());
  }
  stable_sort(gpus.begin(), gpus.end(), [&](int a, int b) -> bool { return gpu_free_mem[a] > gpu_free_mem[b]; });
  gpus.resize(requested_gpus);
  cerr << "[dynet] Device(s) selected:";
  for (int i = 0; i < requested_gpus; ++i) {
    cerr << ' ' << gpus[i];
    Device* d = new Device_GPU(gpudevices.size(), mem_descriptor, gpus[i]);
    gpudevices.push_back(d);
  }
  cerr << endl;

  return gpudevices;
}

} // namespace dynet
