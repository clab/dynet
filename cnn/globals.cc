#include "cnn/globals.h"
#include "cnn/devices.h"

namespace cnn {

std::mt19937* rndeng = nullptr;
std::vector<Device*> devices;
Device* default_device = nullptr;
float weight_decay_lambda;

}
