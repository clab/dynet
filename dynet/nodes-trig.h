#ifndef DYNET_NODES_TRIG_H_
#define DYNET_NODES_TRIG_H_

#include "dynet/dynet.h"
#include "dynet/nodes-macros.h"

namespace dynet {

// y = tanh x_1
struct Tanh : public Node {
  explicit Tanh(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override { Sig s(nt::tanh); return sm.get_idx(s); }
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override { return std::vector<int>(1, 1); }  
  DYNET_NODE_DEFINE_DEV_IMPL()
};

} // namespace dynet

#endif
