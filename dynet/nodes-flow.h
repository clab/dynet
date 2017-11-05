#ifndef DYNET_NODES_FLOW_H_
#define DYNET_NODES_FLOW_H_

#include "dynet/dynet.h"
#include "dynet/nodes-macros.h"

namespace dynet {

// y = reshape(x_1, --> to)
struct Reshape : public Node {
  explicit Reshape(const std::initializer_list<VariableIndex>& a, const Dim& to) : Node(a), to(to) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  Dim to;
};

// y = x_1
struct Identity : public Node {
  explicit Identity(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override { Sig s(nt::identity); return sm.get_idx(s); }
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override { return std::vector<int>(1, 1); }  
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = x_1, but dy/dx is set to 0
struct NoBackprop : public Node {
  explicit NoBackprop(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override { Sig s(nt::nobackprop); return sm.get_idx(s); }
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override { return std::vector<int>(1, 1); }  
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = x_1, dy/dx is set to negative. 
struct FlipGradient : public Node {
  explicit FlipGradient(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override { Sig s(nt::flipgradient); return sm.get_idx(s); }
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override { return std::vector<int>(1, 1); }  
  DYNET_NODE_DEFINE_DEV_IMPL()
};  

} // namespace dynet

#endif
