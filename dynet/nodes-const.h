#ifndef DYNET_NODES_CONST_H_
#define DYNET_NODES_CONST_H_

#include "dynet/dynet.h"
#include "dynet/nodes-macros.h"

namespace dynet {

// represents a simple std::vector of 0s
struct Zeroes : public Node {
  explicit Zeroes(const Dim& d) : dim(d) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  Dim dim;
};

} // namespace dynet

#endif
