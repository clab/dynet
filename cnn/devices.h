#ifndef CNN_DEVICES_H
#define CNN_DEVICES_H

#include <string>
#include "cnn/aligned-mem-pool.h"

namespace cnn {

enum class DeviceType {CPU, GPU};

class Device {
 protected:
  Device(DeviceType t) : type(t) {}
  virtual ~Device();
 public:
  DeviceType type;
  AlignedMemoryPool<6>* fxs;
  AlignedMemoryPool<6>* dEdfs;
  AlignedMemoryPool<6>* ps;
  float* kSCALAR_MINUSONE;
  float* kSCALAR_ONE;
  float* kSCALAR_ZERO;
  std::string name;
};

#if HAVE_CUDA
class Device_GPU : public Device {
 public:
  Device_GPU(int device_id);
  int cuda_device_id;
};
#endif

class Device_CPU : public Device {
 public:
  Device_CPU();
};

} // namespace cnn

#endif
