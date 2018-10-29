#ifndef DYNET_NODES_ROUNDING_H_
#define DYNET_NODES_ROUNDING_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {

// round to nearest int
struct Round : public Node {
  explicit Round(const std::initializer_list<VariableIndex>& a, bool straight_through=false) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override { Sig s(nt::round); s.add_int((int)straight_through); return sm.get_idx(s); }
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override { return std::vector<int>(1, 1); }  
  DYNET_NODE_DEFINE_DEV_IMPL()
  bool straight_through;
};

// round up to int
struct Ceil : public Node {
  explicit Ceil(const std::initializer_list<VariableIndex>& a, bool straight_through=false) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override { Sig s(nt::ceiling); s.add_int((int)straight_through); return sm.get_idx(s); }
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override { return std::vector<int>(1, 1); }  
  DYNET_NODE_DEFINE_DEV_IMPL()
  bool straight_through;
};

// round down to int
struct Floor : public Node {
  explicit Floor(const std::initializer_list<VariableIndex>& a, bool straight_through=false) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override { Sig s(nt::floor); s.add_int((int)straight_through); return sm.get_idx(s); }
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override { return std::vector<int>(1, 1); }  
  DYNET_NODE_DEFINE_DEV_IMPL()
  bool straight_through;
};

} // namespace dynet

#endif
