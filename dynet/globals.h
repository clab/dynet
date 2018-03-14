#ifndef DYNET_GLOBALS_H
#define DYNET_GLOBALS_H

#include <random>

namespace dynet {

class Device;
class NamedTimer;

extern std::mt19937* rndeng;
extern Device* default_device;
extern NamedTimer timer; // debug timing in executors.

} // namespace dynet

#endif
