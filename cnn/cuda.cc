#include <iostream>
#include <vector>
#include <algorithm>

#include "cnn/cnn.h"
#include "cnn/cuda.h"

using namespace std;

namespace cnn {

cublasHandle_t cublas_handle;

static void RemoveArgs(int& argc, char**& argv, int& argi, int n) {
  for (int i = argi + n; i < argc; ++i)
    argv[i - n] = argv[i];
  argc -= n;
  assert(argc >= 0);
}

#define MAX_GPUS 256

vector<Device*> Initialize_GPU(int& argc, char**& argv) {
  int nDevices;
  CUDA_CHECK(cudaGetDeviceCount(&nDevices));
  if (nDevices < 1) {
    cerr << "[cnn] No GPUs found, recompile without DENABLE_CUDA=1\n";
    throw std::runtime_error("No GPUs found but CNN compiled with CUDA support.");
  }
  // logic: no flags, you get 1 GPU
  // or you request a certain number of GPUs explicitly
  // or you request the device ids
  int requested_gpus = -1;
  vector<int> gpu_mask(MAX_GPUS);
  int argi = 1;
  bool ngpus_requested = false;
  bool ids_requested = false;
  for( ;argi < argc; ++argi) {
    string arg = argv[argi];
    if (arg == "--cnn_gpus" || arg == "--cnn-gpus") {
      if ((argi + 1) > argc) {
        cerr << "[cnn] --cnn-gpus expects an argument (number of GPUs to use)\n";
        abort();
      } else {
        if (ngpus_requested) {
          cerr << "Multiple instances of --cnn-gpus" << endl; abort();
        }
        ngpus_requested = true;
        string a2 = argv[argi+1];
        istringstream c(a2); c >> requested_gpus;
        RemoveArgs(argc, argv, argi, 2);
      }
    } else if (arg == "--cnn_gpu_ids" || arg == "--cnn-gpu-ids") {
      if ((argi + 1) > argc) {
        cerr << "[cnn] --cnn-gpu-ids expects an argument (comma separated list of physical GPU ids to use)\n";
        abort();
      } else {
        string a2 = argv[argi+1];
        if (ids_requested) {
          cerr << "Multiple instances of --cnn-gpu-ids" << endl; abort();
        }
        ids_requested = true;
        if (a2.size() % 2 != 1) {
          cerr << "Bad argument to --cnn-gpu-ids: " << a2 << endl; abort();
        }
        for (unsigned i = 0; i < a2.size(); ++i) {
          if ((i % 2 == 0 && (a2[i] < '0' || a2[i] > '9')) ||
              (i % 2 == 1 && a2[i] != ',')) {
            cerr << "Bad argument to --cnn-gpu-ids: " << a2 << endl; abort();
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
              cerr << "Bad argument to --cnn-gpu-ids: " << a2 << endl; abort();
            }
          }
        }
        RemoveArgs(argc, argv, argi, 2);
      }
    }
  }
  if (ids_requested && ngpus_requested) {
    cerr << "Use only --cnn_gpus or --cnn_gpu_ids, not both\n";
    abort();
  }
  if (ngpus_requested || requested_gpus == -1) {
    if (requested_gpus == -1) requested_gpus = 1;
    cerr << "Request for " << requested_gpus << " GPU" << (requested_gpus == 1 ? "" : "s") << " ...\n";
    for (int i = 0; i < MAX_GPUS; ++i) gpu_mask[i] = 1;
  } else if (ids_requested) {
    requested_gpus++;
    cerr << "[cnn] Request for " << requested_gpus << " specific GPU" << (requested_gpus == 1 ? "" : "s") << " ...\n";
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
  size_t free_bytes, total_bytes, max_free = 0;
  int selected = 0;
  for (int i = 0; i < nDevices; i++) {
    if (!gpu_mask[i]) continue;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
    cerr << "[cnn] Device Number: " << i << endl;
    cerr << "[cnn]   Device name: " << prop.name << endl;
    cerr << "[cnn]   Memory Clock Rate (KHz): " << prop.memoryClockRate << endl;
    cerr << "[cnn]   Memory Bus Width (bits): " << prop.memoryBusWidth << endl;
    cerr << "[cnn]   Peak Memory Bandwidth (GB/s): " << (2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6) << endl;
    if (!prop.unifiedAddressing) {
      cerr << "[cnn] GPU does not support unified addressing.\n";
      abort();
    }
    CUDA_CHECK(cudaSetDevice(i));
    CUDA_CHECK(cudaMemGetInfo( &free_bytes, &total_bytes ));
    CUDA_CHECK(cudaDeviceReset());
    cerr << "[cnn]   Memory Free (GB): " << free_bytes/1.0e9 << "/" << total_bytes/1.0e9 << endl;
    cerr << "[cnn]" << endl;
    gpu_free_mem[i] = free_bytes;
  }
  stable_sort(gpus.begin(), gpus.end(), [&](int a, int b) -> bool { return gpu_free_mem[a] > gpu_free_mem[b]; });
  gpus.resize(requested_gpus);
  cerr << "[cnn] Device(s) selected:";
  for (int i = 0; i < requested_gpus; ++i) {
    cerr << ' ' << gpus[i];
    int mb = 512;
    Device* d = new Device_GPU(mb, gpus[i]);
    gpudevices.push_back(d);
  }
  cerr << endl;

  // eventually kill the global handle
  CUDA_CHECK(cudaSetDevice(gpus[0]));
  CUBLAS_CHECK(cublasCreate(&cublas_handle));
  CUBLAS_CHECK(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE));
  return gpudevices;
}

} // namespace cnn
