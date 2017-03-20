#ifndef DYNET_NODES_CONV_H_
#define DYNET_NODES_CONV_H_

#include "dynet/dynet.h"
#include "dynet/nodes-macros.h"

namespace dynet {

// with a single argument x \in R^{n x m}
// y_i = \sum_j x_i,j / m
struct AverageColumns : public Node {
  template <typename T> explicit AverageColumns(const T& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = x_1 *conv x_2
// x_1 \in R^{d x s} (input)
// x_2 \in R^{d x m} (filter)
struct Conv1DNarrow : public Node {
  explicit Conv1DNarrow(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = x_1 *conv x_2
// x_1 \in R^{d x s} (input)
// x_2 \in R^{d x m} (filter)
struct Conv1DWide : public Node {
  explicit Conv1DWide(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = x_1 *filter x_2
// x_1 \in R^{d x s} (input)
// x_2 \in R^{d x m} (filter)
struct Filter1DNarrow : public Node {
  explicit Filter1DNarrow(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
};

struct FoldRows : public Node {
  explicit FoldRows(const std::initializer_list<VariableIndex>& a, unsigned nrows) : Node(a), nrows(nrows) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  unsigned nrows;
};

struct KMaxPooling : public Node {
  explicit KMaxPooling(const std::initializer_list<VariableIndex>& a, unsigned k = 1) : Node(a), k(k) {}
  size_t aux_storage_size() const override;
  DYNET_NODE_DEFINE_DEV_IMPL()
  unsigned k;
};

// sum along a single dimension
struct SumDimension : public Node {
  template <typename T> explicit SumDimension(const T& a, unsigned d) : Node(a), dimension(d) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  unsigned dimension;
};

// 2D convolution
// TODO(Hao Zhang): move conv2d to standalone files because the code logic could be very long
// when cudnn is incorporated.
// y = x_1 *conv x_2
// x_1 \in R^{c1 x s1 x s2 x N} (input)
// x_2 \in R^{c2 x c1 x f1 x f2} (filter)
struct Conv2D: public Node {
  explicit Conv2D(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>& s,
    const bool padding_type = true)
      : Node(a), stride(s), is_valid(padding_type) { }
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
  const std::vector<unsigned> stride;
  const bool is_valid;
};

} // namespace dynet

#endif
