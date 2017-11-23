#ifndef DYNET_NODES_NORMALIZATION_H_
#define DYNET_NODES_NORMALIZATION_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {

// y = x_1 * x_2
struct WeightNormalization : public Node {
  explicit WeightNormalization(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return false; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

} // namespace dynet

#endif
