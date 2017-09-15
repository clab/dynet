#include "dynet/globals.h"
#include "dynet/devices.h"
#include "dynet/timing.h"

namespace dynet {

std::mt19937* rndeng = nullptr;
std::vector<Device*> devices;
std::unordered_map<std::string, Device*> devices_map;
Device* default_device = nullptr;
float weight_decay_lambda;
int autobatch_flag; 
int autobatch_debug_flag = 0;
NamedTimer timer;

}
