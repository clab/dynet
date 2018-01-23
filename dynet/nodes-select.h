#ifndef DYNET_NODES_SELECT_H_
#define DYNET_NODES_SELECT_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {

// y = select_rows(x, rows)
// x = a matrix
struct SelectRows : public Node {
  explicit SelectRows(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>& r) : Node(a), rows(r), prows(&rows) {}
  explicit SelectRows(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>* pr) : Node(a), prows(pr) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  std::vector<unsigned> rows;
  const std::vector<unsigned>* prows;
};

// y = select_cols(x, cols)
// x = a matrix
struct SelectCols : public Node {
  explicit SelectCols(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>& c) : Node(a), cols(c), pcols(&cols) {}
  explicit SelectCols(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>* pc) : Node(a), pcols(pc) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  std::vector<unsigned> cols;
  const std::vector<unsigned>* pcols;
};

// x_1 is a std::vector
// y = (x_1)_{*pval}
// this is used to implement cross-entropy training
struct PickElement : public Node {
  explicit PickElement(const std::initializer_list<VariableIndex>& a, unsigned v, unsigned d = 0) : Node(a), val(v), pval(&val), vals(), pvals(), dimension(d) {}
  // use this constructor if you want to perform mini-batching
  explicit PickElement(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>& v, unsigned d = 0) : Node(a), val(), pval(), vals(v), pvals(&vals), dimension(d) {}
  // use these constructors if you want to change the value after the graph is constructed
  explicit PickElement(const std::initializer_list<VariableIndex>& a, const unsigned* pv, unsigned d = 0) : Node(a), val(), pval(pv), vals(), pvals(), dimension(d) {}
  explicit PickElement(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>* pv, unsigned d = 0) : Node(a), val(), pval(), vals(), pvals(pv), dimension(d) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  unsigned val;
  const unsigned* pval;
  std::vector<unsigned> vals;
  const std::vector<unsigned>* pvals;
  unsigned dimension;
};

// x_1 is a tensor
// y = x_1[start:end] along dimension d
// (start inclusive, end exclusive)
struct PickRange : public Node {
  explicit PickRange(const std::initializer_list<VariableIndex>& a, unsigned s, unsigned e, unsigned d = 0) : Node(a), start(s), end(e), dim(d) {}
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override;
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override { return std::vector<int>(1, 1); }  
  virtual void autobatch_reshape(const ComputationGraph & cg,
                                 const std::vector<VariableIndex> & batch_ids,
                                 const std::vector<int> & concat,
                                 std::vector<const Tensor*>& xs,
                                 Tensor& fx) const override {
    autobatch_reshape_concatonly(cg, batch_ids, concat, xs, fx);
  }
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  unsigned start, end, dim;
};

// x is a batched tensor
// y = (x)_{[*pval]}
struct PickBatchElements : public Node {
  explicit PickBatchElements(const std::initializer_list<VariableIndex>& a, unsigned v) : Node(a), val(v), pval(&val), vals(), pvals() {}
  explicit PickBatchElements(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>& v) : Node(a), val(), pval(), vals(v), pvals(&vals) {}
  explicit PickBatchElements(const std::initializer_list<VariableIndex>& a, const unsigned* pv) : Node(a), val(), pval(pv), vals(), pvals() {}
  explicit PickBatchElements(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>* pv) : Node(a), val(), pval(), vals(), pvals(pv) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  unsigned val;
  const unsigned* pval;
  std::vector<unsigned> vals;
  const std::vector<unsigned>* pvals;
};
// x is a batched tensor
// y = (x)_{[*pval]}
struct StridedSelect : public Node {
  explicit StridedSelect(const std::initializer_list<VariableIndex>& a, const std::vector<int>& strides,
                         const std::vector<int>& from, const std::vector<int>& to) : Node(a), strides(strides), from(from), to(to) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  const std::vector<int> strides, from, to;
};

} // namespace dynet

#endif
