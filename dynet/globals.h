#ifndef DYNET_GLOBALS_H
#define DYNET_GLOBALS_H

#include <random>
#include <vector>

namespace dynet {

class Device;
class NamedTimer;

extern std::mt19937* rndeng;
extern std::vector<Device*> devices;
extern Device* default_device;
extern NamedTimer timer; // debug timing in executors.

} // namespace dynet

#endif
