#ifndef DYNET_NODES_PICKNEGLOGSOFTMAX_H_
#define DYNET_NODES_PICKNEGLOGSOFTMAX_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {

// z = \sum_j \exp (x_i)_j
// y = (x_1)_element - \log z
struct PickNegLogSoftmax : public Node {
  explicit PickNegLogSoftmax(const std::initializer_list<VariableIndex>& a, unsigned v) : Node(a), val(v), pval(&val), vals(), pvals() {}
  // use this constructor if you want to perform mini-batching
  explicit PickNegLogSoftmax(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>& v) : Node(a), val(), pval(), vals(v), pvals(&vals) {}
  // use these constructors if you want to change the value after the graph is constructed
  explicit PickNegLogSoftmax(const std::initializer_list<VariableIndex>& a, const unsigned* pv) : Node(a), val(), pval(pv), vals(), pvals() {}
  explicit PickNegLogSoftmax(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>* pv) : Node(a), val(), pval(), vals(), pvals(pv) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  size_t aux_storage_size() const override;
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override;
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override;
  virtual Node* autobatch_pseudo_node(const ComputationGraph & cg,
                                      const std::vector<VariableIndex> & batch_ids) const override;
  virtual void autobatch_reshape(const ComputationGraph & cg,
                                 const std::vector<VariableIndex> & batch_ids,
                                 const std::vector<int> & concat,
                                 std::vector<const Tensor*>& xs,
                                 Tensor& fx) const override {
    autobatch_reshape_concatonly(cg, batch_ids, concat, xs, fx);
  }
  unsigned val;
  const unsigned* pval;
  std::vector<unsigned> vals;
  const std::vector<unsigned>* pvals;
};

} // namespace dynet

#endif
