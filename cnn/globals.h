#ifndef CNN_EIGEN_RANDOM_H
#define CNN_EIGEN_RANDOM_H

#include <random>
#include <vector>

namespace cnn {

class Device;

extern std::mt19937* rndeng;
extern std::vector<Device*> devices;
extern Device* default_device;

} // namespace cnn

#endif
