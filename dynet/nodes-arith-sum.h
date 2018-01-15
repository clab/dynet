#ifndef DYNET_NODES_ARITH_SUM_H_
#define DYNET_NODES_ARITH_SUM_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {

// y = \sum_i x_i
struct Sum : public Node {
  template <typename T> explicit Sum(const T& a) : Node(a) {}
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override;
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override;
  virtual void autobatch_reshape(const ComputationGraph & cg,
                                 const std::vector<VariableIndex> & batch_ids,
                                 const std::vector<int> & concat,
                                 std::vector<const Tensor*>& xs,
                                 Tensor& fx) const override {
    if(dim.bd != 1)
      autobatch_reshape_concatonly(cg, batch_ids, concat, xs, fx);
  }
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
};

// y = \sum_i,j,... x[i,j,...]
struct SumElements : public Node {
  template <typename T> explicit SumElements(const T& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
};

//y = \sum_i x_i
struct SumDimension : public Node {
  template <typename T> explicit SumDimension(const T& a, const std::vector<unsigned> & d, bool b=false) : Node(a), dims(d), include_batch_dim(b) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
private:
  std::vector<unsigned> dims;
  bool include_batch_dim;
};

// M = x_0, v = x_1
// y = M + v (broadcasting over columns)
struct AddVectorToAllColumns : public Node {
  explicit AddVectorToAllColumns(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

} // namespace dynet

#endif
