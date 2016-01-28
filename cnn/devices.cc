#include "cnn/devices.h"

#include "cnn/cuda.h"

namespace cnn {

#if HAVE_CUDA
Device_GPU::Device_GPU(int device_id) : Device(DeviceType::GPU) {
}
#endif

Device_CPU::Device_CPU() : Device(DeviceType::CPU) {
}

} // namespace cnn
