#ifndef DYNET_NODES_LSTM_H_
#define DYNET_NODES_LSTM_H_

#include "dynet/dynet.h"
#include "dynet/nodes-macros.h"

namespace dynet {

struct VanillaLSTMGates : public Node {
  explicit VanillaLSTMGates(const std::initializer_list<VariableIndex>& a, real weightnoise_std) : Node(a), weightnoise_std(weightnoise_std) {}
  virtual bool supports_multibatch() const override { return true; }
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override;
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override;
  virtual void autobatch_reshape(const ComputationGraph & cg,
                                 const std::vector<VariableIndex> & batch_ids,
                                 const std::vector<int> & concat,
                                 std::vector<const Tensor*>& xs,
                                 Tensor& fx) const override {
    autobatch_reshape_concatonly(cg, batch_ids, concat, xs, fx);
  }
  real weightnoise_std;
  DYNET_NODE_DEFINE_DEV_IMPL()
};
struct VanillaLSTMC : public Node {
  explicit VanillaLSTMC(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override;
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override;
  virtual void autobatch_reshape(const ComputationGraph & cg,
                                 const std::vector<VariableIndex> & batch_ids,
                                 const std::vector<int> & concat,
                                 std::vector<const Tensor*>& xs,
                                 Tensor& fx) const override {
    autobatch_reshape_concatonly(cg, batch_ids, concat, xs, fx);
  }
  DYNET_NODE_DEFINE_DEV_IMPL()
};
struct VanillaLSTMH : public Node {
  explicit VanillaLSTMH(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override;
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override;
  virtual void autobatch_reshape(const ComputationGraph & cg,
                                 const std::vector<VariableIndex> & batch_ids,
                                 const std::vector<int> & concat,
                                 std::vector<const Tensor*>& xs,
                                 Tensor& fx) const override {
    autobatch_reshape_concatonly(cg, batch_ids, concat, xs, fx);
  }
  DYNET_NODE_DEFINE_DEV_IMPL()
};


} // namespace dynet

#endif
