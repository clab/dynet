#ifndef DYNET_NODES_CONTRACT_H_
#define DYNET_NODES_CONTRACT_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {

//   Y_i = A_ijk * B_k
struct InnerProduct3D_1D : public Node {
  InnerProduct3D_1D(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

//   Y_i = A_ijk * B_k * C_j
struct InnerProduct3D_1D_1D : public Node {
  InnerProduct3D_1D_1D(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
};

} // namespace dynet

#endif
