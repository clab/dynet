#ifndef DYNET_NODES_CONST_H_
#define DYNET_NODES_CONST_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {

// represents a simple std::vector of 0s
struct Constant : public Node {
  explicit Constant(const Dim& d, float val=0.f) : dim(d), value(val) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  Dim dim;
  float value;
};

} // namespace dynet

#endif
