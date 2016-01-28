#ifndef CNN_DEVICES_H
#define CNN_DEVICES_H

#include <string>
#include "cnn/aligned-mem-pool.h"

namespace cnn {

enum class DeviceType {CPU, GPU};

class Device {
 protected:
  Device(DeviceType t) : type(t) {}
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
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
  ~Device_GPU();
  int cuda_device_id;
};
#endif

class Device_CPU : public Device {
 public:
  Device_CPU();
  ~Device_CPU();
};

} // namespace cnn

#endif
