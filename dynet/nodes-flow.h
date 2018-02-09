#ifndef DYNET_NODES_FLOW_H_
#define DYNET_NODES_FLOW_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {

// y = reshape(x_1, --> to)
struct Reshape : public Node {
  explicit Reshape(const std::initializer_list<VariableIndex>& a, const Dim& to) : Node(a), to(to) { forward_inplace_state = INPLACE_TYPE::READ; backward_inplace_state = INPLACE_TYPE::WRITE; }
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  Dim to;
};

// y = x_1
struct Identity : public Node {
  explicit Identity(const std::initializer_list<VariableIndex>& a) : Node(a) { forward_inplace_state = INPLACE_TYPE::READ; backward_inplace_state = INPLACE_TYPE::WRITE; }
  virtual bool supports_multibatch() const override { return true; }
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override { if(inplaced()) return 0; Sig s(nt::identity); return sm.get_idx(s); }
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override { return std::vector<int>(1, 1); }  
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = x_1, but dy/dx is set to 0
struct NoBackprop : public Node {
  explicit NoBackprop(const std::initializer_list<VariableIndex>& a) : Node(a) { forward_inplace_state = INPLACE_TYPE::READ; }
  virtual bool supports_multibatch() const override { return true; }
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override { if(inplaced()) return 0; Sig s(nt::nobackprop); return sm.get_idx(s); }
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override { return std::vector<int>(1, 1); }  
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = x_1, dy/dx is multiplied by lambda 
struct ScaleGradient : public Node {
  explicit ScaleGradient(const std::initializer_list<VariableIndex>& a, const float lambd) : Node(a), lambd(lambd) { forward_inplace_state = INPLACE_TYPE::READ; }
  virtual bool supports_multibatch() const override { return true; }
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override { if(inplaced()) return 0; Sig s(nt::scalegradient); return sm.get_idx(s); }
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override { return std::vector<int>(1, 1); }  
  DYNET_NODE_DEFINE_DEV_IMPL()
  float lambd;
};  

} // namespace dynet

#endif
