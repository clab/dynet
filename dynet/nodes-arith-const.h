#ifndef DYNET_NODES_ARITH_CONST_H_
#define DYNET_NODES_ARITH_CONST_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {

// y = c + x_1
// (c is a std::vector or matrix of the constant, usually 1, but can be configured)
struct ConstantPlusX : public Node {
  explicit ConstantPlusX(const std::initializer_list<VariableIndex>& a, real o) : Node(a), c(o) {}
  virtual bool supports_multibatch() const override { return true; }
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override { Sig s(nt::plus_const); s.add_float(c); return sm.get_idx(s); }
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override { return std::vector<int>(1, 1); }  
  DYNET_NODE_DEFINE_DEV_IMPL()
  real c;
};

// y = c - x_1
// (c is a std::vector or matrix of the constant, usually 1, but can be configured)
struct ConstantMinusX : public Node {
  explicit ConstantMinusX(const std::initializer_list<VariableIndex>& a, real o) : Node(a), c(o) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
  real c;
};

// y = alpha * x_1
struct ConstScalarMultiply : public Node {
  explicit ConstScalarMultiply(const std::initializer_list<VariableIndex>& a, float alpha) : Node(a), alpha(alpha) {}
  virtual bool supports_multibatch() const override { return true; }
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override { Sig s(nt::scalar_mult); s.add_float(alpha); return sm.get_idx(s); }
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override { return std::vector<int>(1, 1); }
  DYNET_NODE_DEFINE_DEV_IMPL()
  float alpha;
};

} // namespace dynet

#endif
