#ifndef DYNET_NODES_ACTIVATIONS_H_
#define DYNET_NODES_ACTIVATIONS_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {

// y = max(0,x)
struct Rectify : public Node {
  explicit Rectify(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override { Sig s(nt::rectify); return sm.get_idx(s); }
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override { return std::vector<int>(1, 1); }  
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = \sigma(x_1)
struct LogisticSigmoid : public Node {
  explicit LogisticSigmoid(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override { Sig s(nt::logistic); return sm.get_idx(s); }
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override { return std::vector<int>(1, 1); }  
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = x / (1 + |x|)
struct SoftSign : public Node {
  explicit SoftSign(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override { Sig s(nt::softsign); return sm.get_idx(s); }
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override { return std::vector<int>(1, 1); }  
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = erf x_1
struct Erf : public Node {
  explicit Erf(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override { Sig s(nt::erf); return sm.get_idx(s); }
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override { return std::vector<int>(1, 1); }  
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = ELU(0,x)
struct ExponentialLinearUnit : public Node {
  explicit ExponentialLinearUnit(const std::initializer_list<VariableIndex>& a, float lambda=1.f, float alpha=1.f) : Node(a), lambda(lambda), alpha(alpha) {}
  virtual bool supports_multibatch() const override { return true; }
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override { Sig s(nt::rectify); return sm.get_idx(s); }
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override { return std::vector<int>(1, 1); }  
  DYNET_NODE_DEFINE_DEV_IMPL()
  float lambda, alpha;
};

// y = SILU(x)
struct SigmoidLinearUnit : public Node {
  explicit SigmoidLinearUnit(const std::initializer_list<VariableIndex>& a, float beta=1.f) : Node(a), beta(beta) {}
  virtual bool supports_multibatch() const override { return true; }
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override { Sig s(nt::silu); return sm.get_idx(s); }
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override { return std::vector<int>(1, 1); }
  DYNET_NODE_DEFINE_DEV_IMPL()
  float beta;
};

} // namespace dynet

#endif
