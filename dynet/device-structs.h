#ifndef DYNET_DEVICE_STRUCTS_H
#define DYNET_DEVICE_STRUCTS_H

namespace dynet {

enum class DeviceType {CPU, GPU};

/*
 * FXS   -> forward pass memory
 * DEDFS -> backward pass memory
 * PS    -> parameter memory
 * SCS   -> scratch memory (for use in temporary calculations)
 * NONE  -> when a memory pool has not been assigned yet
 */
enum class DeviceMempool {FXS = 0, DEDFS = 1, PS = 2, SCS = 3, NONE = 4};

struct ComputationGraph; // TODO is there a nicer way to resolve this cyclic dependency?
struct Tensor;

struct DeviceMempoolSizes {
  size_t used[4];
  DeviceMempoolSizes() = default;
  DeviceMempoolSizes(size_t total_s);
  DeviceMempoolSizes(size_t fxs_s, size_t dEdfs_s, size_t ps_s, size_t sc_s);
  DeviceMempoolSizes(const std::string & descriptor);
};

}

#endif
