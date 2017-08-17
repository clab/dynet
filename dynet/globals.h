#ifndef DYNET_GLOBALS_H
#define DYNET_GLOBALS_H

#include <random>
#include <vector>
#include <string>
#include <unordered_map>

namespace dynet {

class Device;
class NamedTimer;

extern std::mt19937* rndeng;
extern std::vector<Device*> devices;
extern std::unordered_map<std::string, Device*> devices_map;
extern Device* default_device;
extern NamedTimer timer; // debug timing in executors.

} // namespace dynet

#endif
