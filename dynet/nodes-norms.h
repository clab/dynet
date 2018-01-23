#ifndef DYNET_NODES_NORMS_H_
#define DYNET_NODES_NORMS_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {

// y = || x_1 ||^2
struct SquaredNorm : public Node {
  explicit SquaredNorm(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = || x_1 ||
struct L2Norm : public Node {
  explicit L2Norm(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

} // namespace dynet

#endif
