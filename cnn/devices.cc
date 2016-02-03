#include "cnn/devices.h"

#include "cnn/cuda.h"

namespace cnn {

Device::~Device() {}

#if HAVE_CUDA
Device_GPU::Device_GPU(int device_id) : Device(DeviceType::GPU) {
}

Device_GPU::~Device_GPU() {}
#endif

Device_CPU::Device_CPU() : Device(DeviceType::CPU) {
}

Device_CPU::~Device_CPU() {}

} // namespace cnn
