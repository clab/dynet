#ifndef DYNET_NODES_ARITH_SCALAR_H_
#define DYNET_NODES_ARITH_SCALAR_H_

#include "dynet/dynet.h"
#include "dynet/nodes-macros.h"

namespace dynet {

// y = x_1 + x_2  (Addition where x_2 is a scalar)
struct ScalarAdd : public Node {
  explicit ScalarAdd(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = x_1 \cdot x_2  (Hadamard product where x_1 is a scalar)
struct ScalarMultiply : public Node {
  explicit ScalarMultiply(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = x_1 / x_2  (Elementwise division where x_2 is a scalar)
struct ScalarQuotient : public Node {
  explicit ScalarQuotient(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

} // namespace dynet

#endif
