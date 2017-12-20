#ifndef DYNET_NODES_SIMILARITIES_H_
#define DYNET_NODES_SIMILARITIES_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {

// y = x_1^T . x_2
struct DotProduct : public Node {
  explicit DotProduct(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = || x_1 - x_2 ||_H(d)
struct HuberDistance : public Node {
  explicit HuberDistance(const std::initializer_list<VariableIndex>& a, float d = 1.345f) : Node(a), d(d) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  float d;
};

// y = || x_1 - x_2 ||_1
struct L1Distance : public Node {
  explicit L1Distance(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = || x_1 - x_2 ||^2
struct SquaredEuclideanDistance : public Node {
  explicit SquaredEuclideanDistance(const std::initializer_list<VariableIndex>& a) : Node(a) {}
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
