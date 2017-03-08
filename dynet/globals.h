#ifndef DYNET_GLOBALS_H
#define DYNET_GLOBALS_H

#include <random>
#include <vector>
#include <sstream>
#include <stdexcept>

#define DYNET_INVALID_ARG(msg) do {             \
    std::ostringstream oss;                     \
    oss << msg;                                 \
    throw std::invalid_argument(oss.str()); }   \
  while (0);

#define DYNET_ASSERT(expr, msg) do {            \
  if(!(expr)) {                                  \
    std::ostringstream oss;                     \
    oss << msg;                                 \
    throw std::runtime_error(oss.str()); }      \
  } while (0);

#define DYNET_RUNTIME_ERR(msg) do {             \
    std::ostringstream oss;                     \
    oss << msg;                                 \
    throw std::runtime_error(oss.str()); }      \
  while (0);

namespace dynet {

class Device;

extern std::mt19937* rndeng;
extern std::vector<Device*> devices;
extern Device* default_device;

} // namespace dynet

#endif
